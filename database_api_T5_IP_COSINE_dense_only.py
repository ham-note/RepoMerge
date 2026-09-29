from pymilvus import MilvusClient
from pymilvus import AnnSearchRequest
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pymilvus import WeightedRanker
import torch
import pandas as pd

def query_similar(CLUSTER_ENDPOINT, TOKEN, vector_database, local_file_path, query, limit):

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

    result = bm25_ef.encode_queries([query]).reshape(1, -1)
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest([{index: float(value) for index, value in zip(result.indices, result.data)}],
                                "chunk_sparse_embedding", sparse_search_params, limit=limit)

    # 对查询使用 CodeT5 进行编码得到密集向量
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():  # 禁用梯度计算
        encoder_outputs = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        dense_embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    dense_embedding = dense_embedding[0]
    
    dense_search_params = {"metric_type": "COSINE"}
    dense_req = AnnSearchRequest([dense_embedding], "chunk_embedding", dense_search_params, limit=limit)

    reqs = [sparse_req, dense_req]
    rerank = WeightedRanker(0, 1) 
    # WeightedRanker(0.2, 0.8)
    
    result = client.hybrid_search(vector_database, reqs, ranker=rerank, limit=limit, output_fields=["chunk_content",  "chunk_resolution", "global_id"])

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
    # print(prompt)
    return global_id, related_conflict, related_resolution