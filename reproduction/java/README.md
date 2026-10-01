# Java Core Reproduction With Local Milvus

This directory contains a compact reproduction harness for the Java MergeBERT subset. It reconstructs the checked-in Java 80/20 split and runs the CodeT5 + BM25/L2 + RRF retrieval pipeline with local Milvus Lite.

The scripts do not require `dataset/after`, `dataset_split`, `dataset_all`, or Zilliz Cloud credentials.

## Run From Repository Root

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r reproduction/java/requirements.txt
.venv/bin/python reproduction/java/download_model.py
.venv/bin/python reproduction/java/reproduce_java.py \
  --repo . \
  --out reproduction-output/java \
  --stage rag \
  --batch-size 8 \
  --top-k 5
```

Expected Java split in the current repository:

| Split | Count |
| --- | ---: |
| Train | 6,674 |
| Test | 1,663 |
| Total | 8,337 |

`reproduction-output/java/results/rag_inputs.jsonl` is the hand-off artifact for the LLM stage.

## Optional DeepSeek Smoke Test

```bash
export DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"

.venv/bin/python reproduction/java/run_deepseek_sample.py \
  --input reproduction-output/java/results/rag_inputs.jsonl \
  --output reproduction-output/java/results/deepseek_flash_50.jsonl \
  --summary reproduction-output/java/results/deepseek_flash_50_summary.json \
  --sample-size 50 \
  --model deepseek-flash \
  --thinking disabled \
  --max-tokens 16384
```

The smoke test checks that the full pipeline can call an external LLM. It is not a paper-level effectiveness result.

## Outputs

The scripts write manifests for data preparation, model embeddings, Milvus indexing, retrieval, and run arguments. Generated files are ignored by Git.

```text
reproduction-output/java/data/manifest.json
reproduction-output/java/embeddings/manifest.json
reproduction-output/java/milvus/manifest.json
reproduction-output/java/results/retrieval_manifest.json
reproduction-output/java/results/rag_inputs.jsonl
reproduction-output/java/run_manifest.json
```

## Chronological Filter

Retrieval is restricted to the same repository and strictly earlier timestamps. Some `autorest` test records have no earlier candidates because many records share the same timestamp. The script keeps the strict filter to avoid unverified temporal leakage.
