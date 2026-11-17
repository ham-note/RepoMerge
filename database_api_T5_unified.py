"""Unified T5 retrieval entry point.

The script keeps the retrieval conditions identical across experiments:

* one time/group filter is applied to every search request;
* ``candidate_k`` is used before fusion and ``limit`` is used after fusion;
* sparse-only, dense-only and hybrid searches share the same encoder/model;
* fusion is explicitly selected as ``weighted`` or ``rrf``.

The default sparse implementation is Milvus BM25.  ``sparse_metric="IP"``
is available for reproducing the former local BM25-vector/IP experiment and
requires ``local_file_path``.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


Timestamp = Union[int, float, str, datetime]
VALID_MODES = {"sparse_only", "dense_only", "hybrid"}
VALID_FUSIONS = {"weighted", "rrf"}
VALID_SPARSE_METRICS = {"BM25", "IP"}
VALID_DENSE_METRICS = {"COSINE", "L2"}


def build_filter_expr(
    current_ts: Optional[Timestamp] = None,
    group_id: Optional[Union[int, str]] = None,
    extra_expr: Optional[str] = None,
) -> Optional[str]:
    """Build one Milvus filter expression shared by all retrieval routes."""
    clauses: List[str] = []
    if current_ts is not None:
        if isinstance(current_ts, datetime):
            value = current_ts.isoformat()
        elif isinstance(current_ts, (int, float)):
            value = str(current_ts)
        else:
            escaped = str(current_ts).replace("\\", "\\\\").replace('"', '\\"')
            value = f'"{escaped}"'
        clauses.append(f"chunk_timestamp < {value}")

    if group_id is not None:
        if isinstance(group_id, (int, float)):
            clauses.append(f"group_id != {group_id}")
        else:
            escaped = str(group_id).replace("\\", "\\\\").replace('"', '\\"')
            clauses.append(f'group_id != "{escaped}"')

    if extra_expr:
        clauses.append(f"({extra_expr})")
    return " and ".join(clauses) or None


class T5UnifiedRetriever:
    """Reusable, parameterized Milvus retriever for all T5 experiments."""

    def __init__(
        self,
        cluster_endpoint: str,
        token: str,
        model_path: str = "./codeT5/codet5-base",
    ) -> None:
        self.client = MilvusClient(uri=cluster_endpoint, token=token)
        self.model_path = model_path
        self.tokenizer, self.model = self._load_model(model_path)
        self.model.eval()
        self._sparse_encoder: Any = None
        self._sparse_encoder_path: Optional[str] = None

    @staticmethod
    def _load_model(model_path: str) -> Tuple[Any, Any]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            print("使用本地模型")
        except Exception as exc:
            print(f"从本地加载模型失败，错误：{exc}")
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
            model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")
            Path(model_path).mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            print("从网络下载并保存本地模型")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return tokenizer, model.to(device)

    def _encode_dense(self, query: str) -> Any:
        inputs = self.tokenizer(
            query, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

    def _encode_sparse_ip(self, local_file_path: str, query: str) -> Dict[int, float]:
        if (
            self._sparse_encoder is None
            or self._sparse_encoder_path != local_file_path
        ):
            from pymilvus.model.sparse import BM25EmbeddingFunction
            from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

            analyzer = build_default_analyzer(language="en")
            self._sparse_encoder = BM25EmbeddingFunction(analyzer)
            pair = pd.read_csv(
                local_file_path, sep="\t", header=None, usecols=[0]
            ).drop_duplicates()
            self._sparse_encoder.fit(pair[0].astype(str).values)
            self._sparse_encoder_path = local_file_path

        encoded = self._sparse_encoder.encode_queries([query]).reshape(1, -1)
        return {
            int(index): float(value)
            for index, value in zip(encoded.indices, encoded.data)
        }

    @staticmethod
    def _metric_params(metric: str) -> Dict[str, str]:
        return {"metric_type": metric}

    @staticmethod
    def _hit_value(hit: Any, key: str, default: Any = None) -> Any:
        if isinstance(hit, dict):
            return hit.get(key, default)
        return getattr(hit, key, default)

    def _search_scores(
        self,
        collection_name: str,
        data: Sequence[Any],
        field: str,
        metric: str,
        candidate_k: int,
        filter_expr: Optional[str],
    ) -> Dict[Any, float]:
        hits = self.client.search(
            collection_name=collection_name,
            data=data,
            anns_field=field,
            search_params=self._metric_params(metric),
            limit=candidate_k,
            filter=filter_expr,
            output_fields=["global_id"],
        )[0]
        scores: Dict[Any, float] = {}
        for hit in hits:
            entity = self._hit_value(hit, "entity", {}) or {}
            item_id = entity.get("global_id")
            distance = self._hit_value(hit, "distance")
            if item_id is not None and distance is not None:
                scores[item_id] = float(distance)
        return scores

    def search(
        self,
        collection_name: str,
        query: str,
        limit: int,
        *,
        mode: str = "hybrid",
        fusion: str = "rrf",
        sparse_metric: str = "BM25",
        dense_metric: str = "L2",
        sparse_weight: float = 0.2,
        dense_weight: float = 0.8,
        candidate_k: Optional[int] = None,
        current_ts: Optional[Timestamp] = None,
        group_id: Optional[Union[int, str]] = None,
        extra_expr: Optional[str] = None,
        local_file_path: Optional[str] = None,
        enable_threshold: bool = False,
        sparse_threshold: float = 0.0,
        dense_threshold: float = float("inf"),
        fusion_threshold: float = 0.0,
    ) -> Dict[str, List[Any]]:
        """Run one consistently filtered retrieval experiment.

        ``mode`` controls which routes participate.  ``candidate_k`` defaults
        to ``max(limit * 10, limit)`` so fusion is not limited to final-k.
        For L2, ``dense_threshold`` is a maximum distance; for other metrics
        ``sparse_threshold`` is a minimum score.
        """
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        if fusion not in VALID_FUSIONS:
            raise ValueError(f"fusion must be one of {sorted(VALID_FUSIONS)}")
        if sparse_metric not in VALID_SPARSE_METRICS:
            raise ValueError(f"unsupported sparse_metric: {sparse_metric}")
        if dense_metric not in VALID_DENSE_METRICS:
            raise ValueError(f"unsupported dense_metric: {dense_metric}")
        if mode == "hybrid" and abs((sparse_weight + dense_weight) - 1.0) > 1e-6:
            raise ValueError("sparse_weight + dense_weight must equal 1.0")

        candidate_k = max(limit, candidate_k or limit * 10)
        filter_expr = build_filter_expr(current_ts, group_id, extra_expr)
        dense_embedding = self._encode_dense(query)
        sparse_query: Any = [query]
        if sparse_metric == "IP":
            if not local_file_path:
                raise ValueError("local_file_path is required when sparse_metric='IP'")
            sparse_query = [self._encode_sparse_ip(local_file_path, query)]

        sparse_req = AnnSearchRequest(
            sparse_query,
            "chunk_sparse_embedding",
            self._metric_params(sparse_metric),
            limit=candidate_k,
            expr=filter_expr,
        )
        dense_req = AnnSearchRequest(
            [dense_embedding],
            "chunk_embedding",
            self._metric_params(dense_metric),
            limit=candidate_k,
            expr=filter_expr,
        )

        requests: List[AnnSearchRequest]
        if mode == "sparse_only":
            requests = [sparse_req]
        elif mode == "dense_only":
            requests = [dense_req]
        else:
            requests = [sparse_req, dense_req]

        if len(requests) == 1:
            ranker: Any = WeightedRanker(1.0)
        elif fusion == "rrf":
            ranker = RRFRanker()
        else:
            ranker = WeightedRanker(sparse_weight, dense_weight)

        fields = ["chunk_content", "chunk_resolution", "global_id"]
        if current_ts is not None:
            fields.append("chunk_timestamp")
        if group_id is not None:
            fields.append("group_id")
        hits = self.client.hybrid_search(
            collection_name,
            requests,
            ranker=ranker,
            limit=limit,
            output_fields=fields,
        )[0]

        sparse_scores: Dict[Any, float] = {}
        dense_scores: Dict[Any, float] = {}
        if mode in {"sparse_only", "hybrid"}:
            sparse_scores = self._search_scores(
                collection_name,
                sparse_query,
                "chunk_sparse_embedding",
                sparse_metric,
                candidate_k,
                filter_expr,
            )
        if mode in {"dense_only", "hybrid"}:
            dense_scores = self._search_scores(
                collection_name,
                [dense_embedding],
                "chunk_embedding",
                dense_metric,
                candidate_k,
                filter_expr,
            )

        rows: List[Dict[str, Any]] = []
        for hit in hits:
            entity = self._hit_value(hit, "entity", {}) or {}
            item_id = entity.get("global_id")
            fusion_score = float(
                self._hit_value(hit, "score", self._hit_value(hit, "distance", 0.0))
            )
            sparse_score = sparse_scores.get(item_id, 0.0)
            dense_score = dense_scores.get(item_id, float("inf"))
            if enable_threshold:
                if mode in {"sparse_only", "hybrid"} and sparse_score < sparse_threshold:
                    continue
                if (
                    mode in {"dense_only", "hybrid"}
                    and dense_metric == "L2"
                    and dense_score > dense_threshold
                ):
                    continue
                if fusion_score < fusion_threshold:
                    continue
            rows.append(
                {
                    "entity": entity,
                    "fusion_score": fusion_score,
                    "sparse_score": sparse_score,
                    "dense_score": dense_score,
                }
            )

        return {
            "global_id": [row["entity"].get("global_id") for row in rows],
            "related_conflict": [row["entity"].get("chunk_content") for row in rows],
            "related_resolution": [
                row["entity"].get("chunk_resolution") for row in rows
            ],
            "timestamp": [
                row["entity"].get("chunk_timestamp") if current_ts is not None else None
                for row in rows
            ],
            "fusion_score": [row["fusion_score"] for row in rows],
            "sparse_score": [row["sparse_score"] for row in rows],
            "dense_score": [row["dense_score"] for row in rows],
        }


def query_similar(
    CLUSTER_ENDPOINT: str,
    TOKEN: str,
    vector_database: str,
    query: str,
    limit: int,
    current_ts: Optional[Timestamp] = None,
    **kwargs: Any,
) -> Dict[str, List[Any]]:
    """Convenience wrapper for the unified hybrid entry point."""
    retriever = T5UnifiedRetriever(CLUSTER_ENDPOINT, TOKEN)
    return retriever.search(
        vector_database,
        query,
        limit,
        current_ts=current_ts,
        **kwargs,
    )


def query_similar_inAll(
    CLUSTER_ENDPOINT: str,
    TOKEN: str,
    vector_database: str,
    query: str,
    limit: int,
    group_id: Union[int, str],
    **kwargs: Any,
) -> Dict[str, List[Any]]:
    """Compatibility wrapper for cross-group retrieval."""
    retriever = T5UnifiedRetriever(CLUSTER_ENDPOINT, TOKEN)
    return retriever.search(
        vector_database,
        query,
        limit,
        group_id=group_id,
        **kwargs,
    )
