# RepoMerge

> Repository-Level Code Merge Conflict Resolution Based on Historical Resolution Retrieval and Large Language Models

RepoMerge resolves repository-level merge conflicts by retrieving similar historical conflict resolutions and injecting them into an LLM prompt. The core pipeline is:

```text
checked-in MergeBERT data
  -> repository-level train/test split
  -> CodeT5 / CodeBERT dense embeddings
  -> BM25 sparse index + dense vector index in Milvus
  -> hybrid retrieval and rank fusion
  -> prompt construction
  -> external LLM generation
```

This repository now includes a lightweight Java reproduction path under `reproduction/java/`. It is the recommended first run because it uses the data already committed in this repository and a local Milvus Lite database. It does not require Zilliz Cloud credentials.

## Repository Layout

```text
RepoMerge/
├── readme.md
├── requirements.txt
├── .env.example
├── utils.py
├── openai_api.py
├── deepseek_api.py
├── alibaba_api.py
├── database_api_T5_BM25_L2.py
├── database_api_T5_BM25_L2_threshold.py
├── database_api_T5_BM25_L2_dense_only.py
├── database_api_T5_IP_COSINE.py
├── database_api_T5_IP_COSINE_dense_only.py
├── database_api_T5_IP_COSINE_sparse_only.py
├── database_api_CB_IP_COSINE.py
├── data-process.ipynb
├── vector-database-T5-BM25-L2.ipynb
├── vector-database-T5-IP-COSINE.ipynb
├── vector-database-CB-IP-COSINE.ipynb
├── reranker.ipynb
├── reranker_all.ipynb
├── dataset/
│   └── mergebert/
│       ├── json/          # Java *_time.json files
│       ├── json_cs/       # C# data
│       ├── json_js/       # JavaScript data
│       ├── json_ts/       # TypeScript data
│       └── vector/        # aggregate train/vector text files
└── reproduction/
    └── java/
        ├── README.md
        ├── requirements.txt
        ├── download_model.py
        ├── reproduce_java.py
        └── run_deepseek_sample.py
```

The older notebooks may mention generated folders such as `dataset_all/`, `dataset_split/`, `dataset_rerank/`, or paths under `dataset/after/`. Those folders are not present in the current repository. The Java quick reproduction below reconstructs the needed 80/20 split directly from the committed files under `dataset/mergebert/json/` and `dataset/mergebert/vector/`.

## Environment

For the Java quick reproduction, use Python 3.10+ because Milvus Lite requires it. Python 3.11 or 3.12 is recommended.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r reproduction/java/requirements.txt
```

The quick path installs the following core dependencies:

| Dependency | Purpose |
| --- | --- |
| `pymilvus` / `milvus-lite` | local Milvus database and hybrid search |
| `transformers` / `torch` | CodeT5 dense embeddings |
| `numpy` | embedding storage |
| `huggingface-hub` | fixed model snapshot download |
| `tqdm` | progress display |

The original top-level `requirements.txt` is kept for notebook-based experiments and remote Milvus/Zilliz workflows.

## Data

The Java data used by the quick reproduction is already committed:

| Path | Meaning |
| --- | --- |
| `dataset/mergebert/json/*_time.json` | Java conflict records with repository, timestamp, conflict parts, and resolution |
| `dataset/mergebert/vector/mergebert_80_Conflicts.txt` | Java training IDs and processed train text |
| `dataset/mergebert/vector/*_Conflicts.txt` | per-repository or aggregate processed conflict text |

The quick reproduction rebuilds the Java split as follows:

1. Load all Java records from `dataset/mergebert/json/*_time.json`.
2. Treat IDs in `dataset/mergebert/vector/mergebert_80_Conflicts.txt` as the training set.
3. Treat the remaining Java IDs as the test set.
4. Recompute the processed conflict/resolution text and compare it with the checked-in vector file.

Expected Java split for the current repository snapshot:

| Split | Count |
| --- | ---: |
| Train | 6,674 |
| Test | 1,663 |
| Total | 8,337 |
| Repositories | 11 |

Do not use complete per-repository `*_time_Conflicts.txt` files as the training database for evaluation, because those files can include records that belong to the reconstructed test set.

## Java Quick Reproduction With Local Milvus

From the repository root:

```bash
.venv/bin/python reproduction/java/download_model.py
.venv/bin/python reproduction/java/reproduce_java.py \
  --repo . \
  --out reproduction-output/java \
  --stage rag \
  --batch-size 8 \
  --top-k 5
```

This runs the reproducible RAG preprocessing stage:

1. Rebuild Java train/test JSONL files.
2. Download or load `Salesforce/codet5-base` at a fixed Hugging Face revision.
3. Encode conflicts as 768-dimensional dense vectors with attention-mask-aware mean pooling over the CodeT5 encoder hidden states.
4. Build a local Milvus Lite database.
5. Create a BM25 sparse index and an L2 dense index.
6. Run hybrid search with RRF fusion.
7. Filter candidates to the same repository and strictly earlier timestamp.
8. Write LLM-ready prompts to `reproduction-output/java/results/rag_inputs.jsonl`.

The run stops before any external LLM call. That makes the retrieval stage reproducible without API keys.

Expected output files:

```text
reproduction-output/java/data/manifest.json
reproduction-output/java/data/train.jsonl
reproduction-output/java/data/test.jsonl
reproduction-output/java/embeddings/manifest.json
reproduction-output/java/embeddings/train.npy
reproduction-output/java/embeddings/test.npy
reproduction-output/java/milvus/manifest.json
reproduction-output/java/results/retrieval_manifest.json
reproduction-output/java/results/rag_inputs.jsonl
reproduction-output/java/run_manifest.json
```

The generated output directory is ignored by Git.

## Local Milvus Lite Configuration

The quick reproduction uses Milvus Lite, so no server process, endpoint, token, or Zilliz account is required. The database is a local file under the selected output directory.

The Java collection schema is:

| Field | Type | Purpose |
| --- | --- | --- |
| `id` | `INT64` primary key | Milvus row id |
| `global_id` | `INT64` | dataset record id |
| `group_id` | `INT64` | repository/group id |
| `chunk_timestamp` | `INT64` | Unix timestamp for chronological filtering |
| `chunk_content` | `VARCHAR` with analyzer | processed conflict text and BM25 input |
| `chunk_sparse_embedding` | `SPARSE_FLOAT_VECTOR` | BM25 function output |
| `chunk_embedding` | `FLOAT_VECTOR(768)` | CodeT5 dense vector |

Indexes:

| Field | Index | Metric |
| --- | --- | --- |
| `chunk_sparse_embedding` | `SPARSE_INVERTED_INDEX` | `BM25` |
| `chunk_embedding` | `AUTOINDEX` | `L2` |

When reopening an existing Milvus Lite database in a new process, the script calls `load_collection()` before retrieval. This is necessary for independently rerunning the retrieval stage.

## Optional LLM Smoke Test

After `rag_inputs.jsonl` is produced, you can run a small DeepSeek-compatible smoke test. The API key is read only from the environment and is never written to output files.

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

The smoke test is only a pipeline check. It should not be reported as a paper-level effectiveness result unless the model, endpoint, decoding parameters, sample selection, retry policy, and evaluation protocol are fixed for the full experiment.

## Optional Zilliz Cloud / Remote Milvus

The original notebooks and `database_api_*.py` modules can be used with Zilliz Cloud or a remote Milvus deployment. Keep credentials outside source code:

```bash
export MILVUS_CLUSTER_ENDPOINT="YOUR_MILVUS_ENDPOINT"
export MILVUS_TOKEN="YOUR_MILVUS_TOKEN"
```

Do not hard-code endpoint or token values in notebooks before committing. Prefer reading them from environment variables or from a local, ignored `.env` file. If a remote collection is reused, document whether it was cleared before indexing and record the collection name, schema, row count, index configuration, and model snapshot used to generate embeddings.

## Notebook Workflow

The historical full workflow is organized as notebooks:

1. `data-process.ipynb`: data cleaning and split construction.
2. `vector-database-*.ipynb`: vector database construction for different retrieval variants.
3. `reranker.ipynb` / `reranker_all.ipynb`: retrieval, prompt construction, LLM calls, and reranking experiments.

Run notebooks from the repository root so that local imports such as `utils`, `database_api_*`, `openai_api`, `deepseek_api`, and `alibaba_api` resolve correctly.

Because some notebook cells still contain older generated path names, use the Java quick reproduction as the reference path for the currently committed dataset layout.

## Retrieval Variants

| File | Sparse retrieval | Dense retrieval | Fusion/ranking |
| --- | --- | --- | --- |
| `database_api_T5_BM25_L2.py` | BM25 | CodeT5 + L2 | RRF |
| `database_api_T5_BM25_L2_threshold.py` | BM25 | CodeT5 + L2 | RRF + threshold filtering |
| `database_api_T5_BM25_L2_dense_only.py` | none | CodeT5 + L2 | dense-only |
| `database_api_T5_IP_COSINE.py` | BM25/IP | CodeT5 + COSINE | WeightedRanker |
| `database_api_T5_IP_COSINE_dense_only.py` | none | CodeT5 + COSINE | dense-only |
| `database_api_T5_IP_COSINE_sparse_only.py` | BM25/IP | none | sparse-only |
| `database_api_CB_IP_COSINE.py` | BM25/IP | CodeBERT + COSINE | WeightedRanker |

## Model Snapshot

The Java quick reproduction fixes CodeT5 to:

```text
model: Salesforce/codet5-base
revision: 02cd2d31bb7c6d0e4d91156167b2de044989c733
```

`download_model.py` records the downloaded weight hash in `reproduction/java/models/manifest.json`. The model directory is ignored by Git.

## Known Reproduction Note

With a strict chronological filter (`chunk_timestamp < current_timestamp`), some `autorest` test records have no earlier historical candidate because many `autorest` records share the same timestamp. The quick reproduction keeps the strict filter to avoid introducing unverified temporal leakage. If the experiment uses `<=` or another tie-break policy, document that policy explicitly.

## Security

- Do not commit `.env`, API keys, Milvus/Zilliz tokens, model outputs, generated embeddings, or local Milvus databases.
- `run_deepseek_sample.py` reads `DEEPSEEK_API_KEY` from the environment and does not serialize it.
- Generated outputs under `reproduction-output/` and `reproduction/java/{models,data,embeddings,milvus,results}/` are ignored.

## Acknowledgements

This project uses MergeBERT conflict data, CodeT5/CodeBERT models, Milvus/Zilliz for vector retrieval, and OpenAI-compatible LLM APIs.
