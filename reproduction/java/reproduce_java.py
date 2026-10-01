#!/usr/bin/env python3
"""Reproduce the Java retrieval stage of RepoMerge with local Milvus Lite.

The script intentionally does not require the obsolete dataset directory names
from the notebooks.  It reconstructs the 80/20 split from the IDs in the
checked-in aggregate Java vector file and the checked-in *_time.json files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


MODEL_ID = "Salesforce/codet5-base"
MODEL_REVISION = "02cd2d31bb7c6d0e4d91156167b2de044989c733"
COLLECTION = "mergebert_java_80"
DIMENSION = 768


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def epoch_seconds(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def processed_conflict(item: dict) -> str:
    text = (
        "concurrent_change\n"
        + str(item.get("a_contents", ""))
        + "\nbase_content\n"
        + str(item.get("o_contents", ""))
        + "\nincoming_change\n"
        + str(item.get("b_contents", ""))
    )
    return text.replace("\n", " _newline_ ").replace("\t", " ")


def processed_resolution(item: dict) -> str:
    return str(item.get("res_region", "")).replace("\n", " _newline_ ").replace("\t", " ")


def locate_sources(repo: Path) -> tuple[Path, Path]:
    dataset = repo / "dataset" / "mergebert"
    raw_dir = dataset / "json"
    vector = dataset / "vector" / "mergebert_80_Conflicts.txt"
    if not raw_dir.is_dir() or not vector.is_file():
        raise FileNotFoundError(
            "Expected dataset/mergebert/json/*_time.json and "
            "dataset/mergebert/vector/mergebert_80_Conflicts.txt"
        )
    return raw_dir, vector


def prepare(repo: Path, out: Path) -> dict:
    raw_dir, vector_path = locate_sources(repo)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw: dict[int, dict] = {}
    source_files = sorted(raw_dir.glob("*_time.json"))
    if not source_files:
        raise RuntimeError(f"No Java *_time.json files under {raw_dir}")
    for source in source_files:
        with source.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        for item in rows:
            gid = int(item["id"])
            if gid in raw:
                raise ValueError(f"Duplicate global ID {gid}")
            raw[gid] = item

    vector_rows: dict[int, tuple[str, str]] = {}
    with vector_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed vector row {line_no}: {len(fields)} fields")
            conflict, resolution, gid_text = fields
            gid = int(gid_text)
            if gid in vector_rows:
                raise ValueError(f"Duplicate vector global ID {gid}")
            vector_rows[gid] = (conflict, resolution)

    missing = sorted(set(vector_rows) - set(raw))
    if missing:
        raise ValueError(f"Training IDs absent from Java JSON: {missing[:20]}")

    repos = sorted({str(item["repo"]) for item in raw.values()})
    repo_to_group = {name: idx + 1 for idx, name in enumerate(repos)}
    train_ids = set(vector_rows)
    train: list[dict] = []
    test: list[dict] = []
    transformation_mismatches = 0

    for gid, item in raw.items():
        if str(item.get("res_label", "")).lower() == "null":
            continue
        repo_url = str(item["repo"])
        common = {
            "global_id": gid,
            "repo": repo_url,
            "repo_name": repo_url.rstrip("/").split("/")[-1],
            "group_id": repo_to_group[repo_url],
            "timestamp": str(item["timestamp"]),
            "timestamp_epoch": epoch_seconds(str(item["timestamp"])),
            "commit_hash": str(item.get("commitHash", "")),
            "file_name": str(item.get("file_name", "")),
        }
        generated_conflict = processed_conflict(item)
        generated_resolution = processed_resolution(item)
        if gid in train_ids:
            stored_conflict, stored_resolution = vector_rows[gid]
            if (stored_conflict, stored_resolution) != (generated_conflict, generated_resolution):
                transformation_mismatches += 1
            train.append(
                {
                    **common,
                    "conflict": stored_conflict,
                    "resolution": stored_resolution,
                }
            )
        else:
            test.append(
                {
                    **common,
                    "conflict": generated_conflict,
                    "resolution": generated_resolution,
                    "a_contents": str(item.get("a_contents", "")),
                    "o_contents": str(item.get("o_contents", "")),
                    "b_contents": str(item.get("b_contents", "")),
                }
            )

    train.sort(key=lambda x: (x["group_id"], x["timestamp_epoch"], x["global_id"]))
    test.sort(key=lambda x: (x["group_id"], x["timestamp_epoch"], x["global_id"]))
    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"
    train_path.write_text("".join(json_line(x) for x in train), encoding="utf-8")
    test_path.write_text("".join(json_line(x) for x in test), encoding="utf-8")

    counts = defaultdict(lambda: {"train": 0, "test": 0})
    for row in train:
        counts[row["repo_name"]]["train"] += 1
    for row in test:
        counts[row["repo_name"]]["test"] += 1
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_repository": str(repo.resolve()),
        "source_commit": git_commit(repo),
        "split_rule": "train IDs are exactly those in mergebert_80_Conflicts.txt; all other Java IDs are test",
        "train_count": len(train),
        "test_count": len(test),
        "total_count": len(train) + len(test),
        "transformation_mismatches": transformation_mismatches,
        "repositories": dict(sorted(counts.items())),
        "inputs": {
            str(vector_path.relative_to(repo)): sha256(vector_path),
            **{str(p.relative_to(repo)): sha256(p) for p in source_files},
        },
        "outputs": {
            "data/train.jsonl": sha256(train_path),
            "data/test.jsonl": sha256(test_path),
        },
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"prepared train={len(train)} test={len(test)} mismatches={transformation_mismatches}")
    return manifest


def git_commit(repo: Path) -> str | None:
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def select_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_records(
    records: list[dict],
    destination: Path,
    model_dir: Path,
    batch_size: int,
    requested_device: str,
) -> dict:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = select_device(torch, requested_device)
    print(f"loading CodeT5 from {model_dir} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
    model.eval().to(device)

    temp = destination.with_suffix(destination.suffix + ".partial")
    embeddings = np.lib.format.open_memmap(
        temp, mode="w+", dtype=np.float32, shape=(len(records), DIMENSION)
    )
    started = time.time()
    for start in tqdm(range(0, len(records), batch_size), desc=destination.stem):
        texts = [row["conflict"] for row in records[start : start + batch_size]]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            hidden = model.encoder(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        embeddings[start : start + len(texts)] = pooled.float().cpu().numpy()
    embeddings.flush()
    del embeddings, model
    if device == "mps":
        torch.mps.empty_cache()
    os.replace(temp, destination)
    return {
        "rows": len(records),
        "dimension": DIMENSION,
        "dtype": "float32",
        "device": device,
        "batch_size": batch_size,
        "max_tokens": 512,
        "seconds": round(time.time() - started, 3),
        "sha256": sha256(destination),
    }


def embed(out: Path, model_dir: Path, batch_size: int, device: str) -> dict:
    data_dir = out / "data"
    emb_dir = out / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(data_dir / "train.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")
    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pooling": "attention-mask-aware mean of encoder last hidden state",
        "train": embed_records(train, emb_dir / "train.npy", model_dir, batch_size, device),
        "test": embed_records(test, emb_dir / "test.npy", model_dir, batch_size, device),
    }
    (emb_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("embeddings complete")
    return metadata


def create_database(out: Path) -> dict:
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    data = read_jsonl(out / "data" / "train.jsonl")
    vectors = np.load(out / "embeddings" / "train.npy", mmap_mode="r")
    if len(data) != len(vectors):
        raise ValueError("Train record/embedding count mismatch")
    db_path = out / "milvus" / "mergebert_java.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        if db_path.is_dir():
            shutil.rmtree(db_path)
        else:
            db_path.unlink()
    client = MilvusClient(str(db_path))
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("global_id", DataType.INT64)
    schema.add_field("group_id", DataType.INT64)
    schema.add_field("chunk_timestamp", DataType.INT64)
    schema.add_field("chunk_content", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    schema.add_field("chunk_sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("chunk_embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_function(
        Function(
            name="text_bm25_emb",
            input_field_names=["chunk_content"],
            output_field_names=["chunk_sparse_embedding"],
            function_type=FunctionType.BM25,
        )
    )
    indexes = MilvusClient.prepare_index_params()
    indexes.add_index(
        field_name="chunk_embedding", metric_type="L2", index_type="AUTOINDEX", index_name="dense_l2"
    )
    indexes.add_index(
        field_name="chunk_sparse_embedding",
        metric_type="BM25",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_bm25",
        params={"drop_ratio_build": 0.2},
    )
    client.create_collection(COLLECTION, schema=schema, index_params=indexes)
    for start in range(0, len(data), 256):
        rows = []
        for offset, item in enumerate(data[start : start + 256]):
            rows.append(
                {
                    "global_id": item["global_id"],
                    "group_id": item["group_id"],
                    "chunk_timestamp": item["timestamp_epoch"],
                    "chunk_content": item["conflict"],
                    "chunk_embedding": np.asarray(vectors[start + offset], dtype=np.float32),
                }
            )
        client.insert(COLLECTION, rows)
        print(f"indexed {min(start + len(rows), len(data))}/{len(data)}")
    client.flush(COLLECTION)
    stats = client.get_collection_stats(COLLECTION)
    database_bytes = sum(p.stat().st_size for p in db_path.rglob("*") if p.is_file())
    metadata = {
        "database": str(db_path),
        "collection": COLLECTION,
        "row_count": int(stats["row_count"]),
        "dense_metric": "L2",
        "sparse_metric": "BM25",
        "ranker": "RRF (Milvus default k)",
        "database_bytes": database_bytes,
        "resolution_storage": "full resolutions remain in data/train.jsonl and are joined by global_id",
    }
    (db_path.parent / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"local Milvus complete rows={metadata['row_count']}")
    return metadata


def prompt_for(query: dict, references: list[dict]) -> str:
    refs = "\n\n".join(
        f"Historical conflict:\n{x['conflict']}\nHistorical resolution:\n{x['resolution']}"
        for x in references
    )
    return (
        "Resolve the target Java merge conflict. Return only the resolved code.\n\n"
        + refs
        + "\n\nTarget conflict:\n"
        + query["conflict"]
    )


def retrieve(out: Path, top_k: int) -> dict:
    from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
    from tqdm import tqdm

    train = read_jsonl(out / "data" / "train.jsonl")
    test = read_jsonl(out / "data" / "test.jsonl")
    train_by_id = {x["global_id"]: x for x in train}
    vectors = np.load(out / "embeddings" / "test.npy", mmap_mode="r")
    if len(test) != len(vectors):
        raise ValueError("Test record/embedding count mismatch")
    client = MilvusClient(str(out / "milvus" / "mergebert_java.db"))
    # A persisted Milvus Lite collection is released when a new process opens
    # the database; loading explicitly makes the retrieval stage independently
    # runnable instead of relying on state left by the indexing process.
    client.load_collection(COLLECTION)
    result_dir = out / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "rag_inputs.jsonl"
    partial = output_path.with_suffix(".jsonl.partial")
    no_history = 0
    no_history_by_repository: dict[str, int] = defaultdict(int)
    retrieved_count_distribution: dict[int, int] = defaultdict(int)
    with partial.open("w", encoding="utf-8") as handle:
        for idx, query in enumerate(tqdm(test, desc="hybrid retrieval")):
            expr = (
                f"group_id == {query['group_id']} and "
                f"chunk_timestamp < {query['timestamp_epoch']}"
            )
            sparse = AnnSearchRequest(
                [query["conflict"]],
                "chunk_sparse_embedding",
                {"metric_type": "BM25"},
                limit=top_k,
                expr=expr,
            )
            dense = AnnSearchRequest(
                [np.asarray(vectors[idx], dtype=np.float32)],
                "chunk_embedding",
                {"metric_type": "L2"},
                limit=top_k,
                expr=expr,
            )
            hits = client.hybrid_search(
                COLLECTION,
                [sparse, dense],
                ranker=RRFRanker(),
                limit=top_k,
                output_fields=["global_id", "group_id", "chunk_timestamp"],
            )[0]
            ids = [int(hit["entity"]["global_id"]) for hit in hits]
            references = [train_by_id[x] for x in ids]
            if not references:
                no_history += 1
                no_history_by_repository[query["repo_name"]] += 1
            retrieved_count_distribution[len(references)] += 1
            record = {
                "global_id": query["global_id"],
                "repo": query["repo"],
                "timestamp": query["timestamp"],
                "retrieved_global_ids": ids,
                "retrieved_references": [
                    {
                        "global_id": x["global_id"],
                        "timestamp": x["timestamp"],
                        "conflict": x["conflict"],
                        "resolution": x["resolution"],
                    }
                    for x in references
                ],
                "gold_resolution": query["resolution"],
                "prompt_for_llm": prompt_for(query, references[:1]),
            }
            handle.write(json_line(record))
    os.replace(partial, output_path)
    metadata = {
        "queries": len(test),
        "top_k": top_k,
        "filter": "same repository and strictly earlier timestamp",
        "no_historical_candidate": no_history,
        "no_historical_candidate_by_repository": dict(sorted(no_history_by_repository.items())),
        "retrieved_count_distribution": {
            str(k): v for k, v in sorted(retrieved_count_distribution.items())
        },
        "output": str(output_path),
        "sha256": sha256(output_path),
        "llm_status": "STOP POINT: RAG inputs are ready; no external LLM was invoked",
        "next_required_input": "external LLM API key plus an exact provider/model/version selection",
    }
    (result_dir / "retrieval_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"retrieval complete queries={len(test)} no_history={no_history}")
    return metadata


def normalized(text: str) -> str:
    return text.replace(" _newline_ ", "\n").replace("_newline_", "\n").strip()


def evaluate(out: Path) -> dict:
    train = read_jsonl(out / "data" / "train.jsonl")
    train_by_id = {x["global_id"]: x for x in train}
    predictions = read_jsonl(out / "results" / "rag_inputs.jsonl")
    by_repo = defaultdict(lambda: {"total": 0, "top1_exact": 0, "top3_oracle_exact": 0})
    top1 = top3 = 0
    for item in predictions:
        gold = normalized(item["gold_resolution"])
        retrieved = [train_by_id[x]["resolution"] for x in item["retrieved_global_ids"]]
        one = bool(retrieved) and normalized(retrieved[0]) == gold
        three = any(normalized(x) == gold for x in retrieved[:3])
        top1 += int(one)
        top3 += int(three)
        stats = by_repo[item["repo"].rstrip("/").split("/")[-1]]
        stats["total"] += 1
        stats["top1_exact"] += int(one)
        stats["top3_oracle_exact"] += int(three)
    total = len(predictions)
    for stats in by_repo.values():
        stats["top1_rate"] = stats["top1_exact"] / stats["total"] if stats["total"] else 0
        stats["top3_oracle_rate"] = stats["top3_oracle_exact"] / stats["total"] if stats["total"] else 0
    metrics = {
        "evaluated": total,
        "metric_scope": "retrieval-only exact-match diagnostic; no LLM generation",
        "top1_exact": top1,
        "top1_exact_rate": top1 / total if total else 0,
        "top3_oracle_exact": top3,
        "top3_oracle_exact_rate": top3 / total if total else 0,
        "by_repository": dict(sorted(by_repo.items())),
    }
    path = out / "results" / "metrics.json"
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def write_run_manifest(out: Path, args: argparse.Namespace, elapsed: float) -> None:
    manifest = {
        "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "arguments": vars(args),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "models/codet5-base")
    parser.add_argument("--stage", choices=["rag", "all", "prepare", "embed", "index", "retrieve", "evaluate"], default="rag")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out = args.out.resolve()
    args.repo = args.repo.resolve()
    args.model_dir = args.model_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    order = ["prepare", "embed", "index", "retrieve", "evaluate"]
    stages: Iterable[str] = order if args.stage == "all" else (order[:4] if args.stage == "rag" else [args.stage])
    for stage in stages:
        print(f"\n=== {stage.upper()} ===", flush=True)
        if stage == "prepare":
            prepare(args.repo, args.out)
        elif stage == "embed":
            embed(args.out, args.model_dir, args.batch_size, args.device)
        elif stage == "index":
            create_database(args.out)
        elif stage == "retrieve":
            retrieve(args.out, args.top_k)
        elif stage == "evaluate":
            evaluate(args.out)
    write_run_manifest(args.out, args, time.time() - started)


if __name__ == "__main__":
    main()
