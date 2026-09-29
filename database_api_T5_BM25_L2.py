from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
from datetime import datetime

_CLIENT_CACHE = {}
_MODEL_CACHE = {}


def _get_client(cluster_endpoint, token):
    cache_key = (cluster_endpoint, token)
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = MilvusClient(uri=cluster_endpoint, token=token)
        _CLIENT_CACHE[cache_key] = client
    return client


def _get_model(local_model_path="./codeT5/codet5-base"):
    if local_model_path in _MODEL_CACHE:
        return _MODEL_CACHE[local_model_path]

    try:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("使用本地模型")
    except Exception as e:
        print(f"从本地加载模型失败，错误：{e}")
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)
        print("从网络下载并保存本地模型")

    _MODEL_CACHE[local_model_path] = (tokenizer, model)
    return tokenizer, model

def query_similar(CLUSTER_ENDPOINT, TOKEN, vector_database, query, limit, current_ts=None):
    """在向量库中检索相似冲突，支持按时间上界过滤。

    Args:
        CLUSTER_ENDPOINT (str): Milvus 连接地址
        TOKEN (str): Milvus token
        vector_database (str): collection 名
        query (str): 查询文本
        limit (int): 返回条数
        current_ts (Optional[int|float|str]): 当前冲突的时间戳，上界过滤使用。
            - 若为数值，直接用于比较；若为字符串，需与库中 chunk_timestamp 类型匹配。
            - None 时不做时间过滤，保持现有行为。
    """

    # 云端向量数据库连接
    client = _get_client(CLUSTER_ENDPOINT, TOKEN)
    
    # 加载本地 CodeT5 模型
    tokenizer, model = _get_model()

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

    reqs = [sparse_req, dense_req]
    # rerank = WeightedRanker(1, 0)
    rerank = RRFRanker()
    
    # NOTE: 部分 collection 可能没有 chunk_timestamp 字段；即便请求，也可能返回空。
    result = client.hybrid_search(
        vector_database,
        reqs,
        ranker=rerank,
        limit=limit,
        output_fields=["chunk_content", "chunk_resolution", "global_id", "chunk_timestamp"],
    )

    related_conflict = [x["entity"]["chunk_content"] for x in result[0]]
    related_resolution = [x["entity"]["chunk_resolution"] for x in result[0]]
    global_id = [x["entity"]["global_id"] for x in result[0]]
    # 使用 get 避免缺失字段触发 KeyError，当库没有 chunk_timestamp 时返回 None。
    timestamp = [x["entity"].get("chunk_timestamp") for x in result[0]]
    prompt = f"""查询：
{query}

相似的冲突块:
{related_conflict}

对应的resolution:
{related_resolution}

对应的全局id:
{global_id}
    """
    # print(prompt)
    return global_id, related_conflict, related_resolution, timestamp

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

    reqs = [sparse_req, dense_req]
    rerank = RRFRanker()
    
    result = client.hybrid_search(vector_database, reqs, ranker=rerank, limit=limit, output_fields=["chunk_content",  "chunk_resolution", "global_id", "group_id"])

    related_conflict = [x["entity"]["chunk_content"] for x in result[0]]
    related_resolution = [x["entity"]["chunk_resolution"] for x in result[0]]
    global_id = [x["entity"]["global_id"] for x in result[0]]
    group_id = [x["entity"]["group_id"] for x in result[0]]

    return group_id, global_id, related_conflict, related_resolution