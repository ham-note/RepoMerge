from pymilvus import MilvusClient
from pymilvus import connections, MilvusClient, DataType
from pymilvus import AnnSearchRequest
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
from transformers import AutoTokenizer, AutoModel
from pymilvus import WeightedRanker
import torch
import pandas as pd


def query_similar(CLUSTER_ENDPOINT, TOKEN, vector_database, local_file_path, query, limit, current_ts=None):
    """混合检索相似冲突，支持按时间戳上界过滤。

    Args:
        current_ts: 可选时间上界。
            - 若为字符串（推荐 ISO 8601），将自动加引号用于 Milvus 表达式，且 ISO 字符串的字典序与时间序一致。
            - 若为数值（秒/毫秒），将直接用于比较。
    """

    # 云端向量数据库连接
    client = MilvusClient(
        uri=CLUSTER_ENDPOINT,
        token=TOKEN 
    )

    analyzer = build_default_analyzer(language="en") 
    bm25_ef = BM25EmbeddingFunction(analyzer)

    pair = pd.read_csv(local_file_path, sep='\t', header=None).drop_duplicates()
    conflicts = pair[0].values
    bm25_ef.fit(conflicts)
    
    # 加载本地 CodeBERT 模型
    local_model_path = "./codebert/codebert-base"
    def download_or_load_model(local_model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModel.from_pretrained(local_model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            model = AutoModel.from_pretrained("microsoft/codebert-base").to("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer.save_pretrained(local_model_path)
            model.save_pretrained(local_model_path)

        return tokenizer, model

    tokenizer, model = download_or_load_model(local_model_path)

    result = bm25_ef.encode_queries([query]).reshape(1, -1)
    # 可选时间过滤表达式
    expr = None
    if current_ts is not None:
        if isinstance(current_ts, str):
            expr = f'chunk_timestamp < "{current_ts}"'
        else:
            expr = f"chunk_timestamp < {current_ts}"

    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(
        [{index: float(value) for index, value in zip(result.indices, result.data)}],
        "chunk_sparse_embedding",
        sparse_search_params,
        limit=limit,
        expr=expr,
    )

    # 对查询使用 CodeBERT 进行编码（得到密集向量）
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():  # 禁用梯度计算
        outputs = model(**inputs)
        dense_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # 确保密集向量的维度为 768
    dense_embedding = dense_embedding[0]
    dense_search_params = {"metric_type": "COSINE"}
    dense_req = AnnSearchRequest([dense_embedding], "chunk_embedding", dense_search_params, limit=limit, expr=expr)

    reqs = [sparse_req, dense_req]
    rerank = WeightedRanker(0, 1) # WeightedRanker(0.2, 0.8)
    
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
    prompt = f"""查询：
{query}

相似的冲突块:
{related_conflict}

对应的resolution:
{related_resolution}

对应的全局id:
{global_id}
    """
    print(prompt)
    return global_id, related_conflict, related_resolution