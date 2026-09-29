from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
from datetime import datetime

def query_similar(CLUSTER_ENDPOINT, TOKEN, vector_database, query, limit, current_ts=None, 
                  enable_threshold=False, bm25_threshold=0.0, l2_threshold=float('inf'), rrf_threshold=0.0):
    """在向量库中检索相似冲突，支持按时间上界过滤和质量阈值过滤。

    Args:
        CLUSTER_ENDPOINT (str): Milvus 连接地址
        TOKEN (str): Milvus token
        vector_database (str): collection 名
        query (str): 查询文本
        limit (int): 返回条数
        current_ts (Optional[int|float|str]): 当前冲突的时间戳，上界过滤使用。
            - 若为数值,直接用于比较；若为字符串，需与库中 chunk_timestamp 类型匹配。
            - None 时不做时间过滤，保持现有行为。
        enable_threshold (bool): 是否启用阈值过滤，默认 False
        bm25_threshold (float): BM25 最低分值阈值（越高越相关），默认 0.0
        l2_threshold (float): L2 最大距离阈值（越小越相关），默认 inf
        rrf_threshold (float): RRF 融合分值最低阈值，默认 0.0
    """

    # 云端向量数据库连接
    client = MilvusClient(
        uri=CLUSTER_ENDPOINT,
        token=TOKEN 
    )
    
    # 加载本地 CodeT5 模型
    local_model_path = "./codeT5/codet5-base"
    def download_or_load_model(local_model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path) # 尝试从本地加载模型
            model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path).to("cuda" if torch.cuda.is_available() else "cpu")
            print("使用本地模型")
        except Exception as e:
            print(f"从本地加载模型失败，错误：{e}")
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base") # 本地不存在则从网络下载
            model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to("cuda" if torch.cuda.is_available() else "cpu")
            # 保存到本地
            tokenizer.save_pretrained(local_model_path)
            model.save_pretrained(local_model_path)
            print("从网络下载并保存本地模型")
        return tokenizer, model
    tokenizer, model = download_or_load_model(local_model_path)

    # 可选时间过滤：仅检索早于 current_ts 的历史记录
    expr = None
    if current_ts is not None:
        # 根据类型构造过滤表达式：数值直接比较，字符串/时间戳需加引号
        if isinstance(current_ts, (int, float)):
            expr = f"chunk_timestamp < {current_ts}"
        else:
            # 支持 datetime 和 ISO 字符串，统一转为字符串后包裹引号
            ts_str = current_ts.isoformat() if isinstance(current_ts, datetime) else str(current_ts)
            expr = f'chunk_timestamp < "{ts_str}"'

    sparse_search_params = {"metric_type": "BM25"}
    sparse_req = AnnSearchRequest([query], "chunk_sparse_embedding", sparse_search_params, limit=limit, expr=expr)
    # 对查询使用 CodeT5 进行编码得到密集向量
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():  # 禁用梯度计算
        encoder_outputs = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        dense_embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    dense_embedding = dense_embedding[0]

    dense_search_params = {"metric_type": "L2"}
    dense_req = AnnSearchRequest([dense_embedding], "chunk_embedding", dense_search_params, limit=limit, expr=expr)

    # 分别获取原始分值（用于阈值过滤）
    bm25_results = client.search(
        collection_name=vector_database,
        data=[query],
        anns_field="chunk_sparse_embedding",
        search_params=sparse_search_params,
        limit=limit,
        filter=expr,
        output_fields=["global_id"]
    )
    
    l2_results = client.search(
        collection_name=vector_database,
        data=[dense_embedding],
        anns_field="chunk_embedding",
        search_params=dense_search_params,
        limit=limit,
        filter=expr,
        output_fields=["global_id"]
    )
    
    # 构建 id -> 原始分值的映射
    bm25_scores = {x["entity"]["global_id"]: x["distance"] for x in bm25_results[0]}
    l2_scores = {x["entity"]["global_id"]: getattr(x, "distance", float('inf')) for x in l2_results[0]}

    # 混合搜索
    #reqs = [sparse_req, dense_req]
    reqs = [dense_req]
    #rerank = RRFRanker()
    rerank =WeightedRanker(1)
    
    # 仅在需要时间过滤时请求 chunk_timestamp 字段，避免无关场景下因缺列报错
    output_fields = ["chunk_content", "chunk_resolution", "global_id"]
    if current_ts is not None:
        output_fields.append("chunk_timestamp")

    result = client.hybrid_search(vector_database, reqs, ranker=rerank, limit=limit, output_fields=output_fields)

    # 提取结果并附加原始分值
    filtered_results = []
    for x in result[0]:
        item_id = x["entity"]["global_id"]
        rrf_score = getattr(x, "score", 0)
        bm25_score = bm25_scores.get(item_id, 0)
        l2_distance = l2_scores.get(item_id, float('inf'))
        
        # 阈值过滤
        if enable_threshold:
            if bm25_score < bm25_threshold:
                continue
            if l2_distance > l2_threshold:
                continue
            if rrf_score < rrf_threshold:
                continue
        
        filtered_results.append({
            "entity": x["entity"],
            "rrf_score": rrf_score,
            "bm25_score": bm25_score,
            "l2_distance": l2_distance
        })
    
    # 提取最终结果
    related_conflict = [x["entity"]["chunk_content"] for x in filtered_results]
    related_resolution = [x["entity"]["chunk_resolution"] for x in filtered_results]
    global_id = [x["entity"]["global_id"] for x in filtered_results]

    # 仅在请求了时间字段时才读取；否则返回全 None 占位，避免 KeyError
    if current_ts is not None:
        timestamp = [x["entity"].get("chunk_timestamp") for x in filtered_results]
    else:
        timestamp = [None for _ in filtered_results]
    rrf_scores = [x["rrf_score"] for x in filtered_results]
    bm25_scores_list = [x["bm25_score"] for x in filtered_results]
    l2_distances_list = [x["l2_distance"] for x in filtered_results]
    
    print(f"检索到 {len(filtered_results)} 个结果（启用阈值过滤: {enable_threshold}）")
    if enable_threshold and len(filtered_results) > 0:
        print(f"  RRF分值范围: [{min(rrf_scores):.4f}, {max(rrf_scores):.4f}]")
        print(f"  BM25分值范围: [{min(bm25_scores_list):.4f}, {max(bm25_scores_list):.4f}]")
        print(f"  L2距离范围: [{min(l2_distances_list):.4f}, {max(l2_distances_list):.4f}]")
    
    return global_id, related_conflict, related_resolution, timestamp, rrf_scores, bm25_scores_list, l2_distances_list

"""基于全局仓库进行检索最相似历史合并冲突消解记录"""
def query_similar_inAll(CLUSTER_ENDPOINT, TOKEN, vector_database, query, limit, group_id):

    # 云端向量数据库连接
    client = MilvusClient(
        uri=CLUSTER_ENDPOINT,
        token=TOKEN 
    )
    
    # 加载本地 CodeT5 模型
    local_model_path = "./codeT5/codet5-base"
    def download_or_load_model(local_model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path) # 尝试从本地加载模型
            model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path).to("cuda" if torch.cuda.is_available() else "cpu")
            print("使用本地模型")
        except Exception as e:
            print(f"从本地加载模型失败，错误：{e}")
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base") # 本地不存在则从网络下载
            model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to("cuda" if torch.cuda.is_available() else "cpu")
            # 保存到本地
            tokenizer.save_pretrained(local_model_path)
            model.save_pretrained(local_model_path)
            print("从网络下载并保存本地模型")
        return tokenizer, model
    tokenizer, model = download_or_load_model(local_model_path)

    expr = "group_id != " + str(group_id) # 过滤表达式：过滤掉本组内的冲突后再跨仓库检索

    sparse_search_params = {"metric_type": "BM25"}
    sparse_req = AnnSearchRequest([query], "chunk_sparse_embedding", sparse_search_params, limit=limit, expr=expr)
    # 对查询使用 CodeT5 进行编码得到密集向量
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():  # 禁用梯度计算
        encoder_outputs = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        dense_embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    dense_embedding = dense_embedding[0]

    dense_search_params = {"metric_type": "L2"}
    dense_req = AnnSearchRequest([dense_embedding], "chunk_embedding", dense_search_params, limit=limit, expr=expr)
# 分别获取原始分值（用于阈值过滤）
    bm25_results = client.search(
        collection_name=vector_database,
        data=[query],
        anns_field="chunk_sparse_embedding",
        search_params=sparse_search_params,
        limit=limit,
        filter=expr,
        output_fields=["global_id"]
    )
    
    l2_results = client.search(
        collection_name=vector_database,
        data=[dense_embedding],
        anns_field="chunk_embedding",
        search_params=dense_search_params,
        limit=limit,
        filter=expr,
        output_fields=["global_id"]
    )
    
    # 构建 id -> 原始分值的映射
    bm25_scores = {x["entity"]["global_id"]: getattr(x, "distance", 0) for x in bm25_results[0]}
    l2_scores = {x["entity"]["global_id"]: getattr(x, "distance", float('inf')) for x in l2_results[0]}
    
    # 混合搜索
    reqs = [sparse_req, dense_req]
    rerank = RRFRanker()
    
    result = client.hybrid_search(vector_database, reqs, ranker=rerank, limit=limit, output_fields=["chunk_content",  "chunk_resolution", "global_id", "group_id"])

    # 提取结果并附加原始分值
    filtered_results = []
    for x in result[0]:
        item_id = x["entity"]["global_id"]
        rrf_score = getattr(x, "score", 0)
        bm25_score = bm25_scores.get(item_id, 0)
        l2_distance = l2_scores.get(item_id, float('inf'))
        
        # 阈值过滤
        if enable_threshold:
            if bm25_score < bm25_threshold:
                continue
            if l2_distance > l2_threshold:
                continue
            if rrf_score < rrf_threshold:
                continue
        
        filtered_results.append({
            "entity": x["entity"],
            "rrf_score": rrf_score,
            "bm25_score": bm25_score,
            "l2_distance": l2_distance
        })
    
    # 提取最终结果
    related_conflict = [x["entity"]["chunk_content"] for x in filtered_results]
    related_resolution = [x["entity"]["chunk_resolution"] for x in filtered_results]
    global_id = [x["entity"]["global_id"] for x in filtered_results]
    group_id_list = [x["entity"]["group_id"] for x in filtered_results]
    rrf_scores = [x["rrf_score"] for x in filtered_results]
    bm25_scores_list = [x["bm25_score"] for x in filtered_results]
    l2_distances_list = [x["l2_distance"] for x in filtered_results]
    
    print(f"检索到 {len(filtered_results)} 个结果（启用阈值过滤: {enable_threshold}）")

    return group_id_list, global_id, related_conflict, related_resolution, rrf_scores, bm25_scores_list, l2_distances_list
    return group_id, global_id, related_conflict, related_resolution