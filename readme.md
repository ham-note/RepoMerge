# 基于历史信息的仓库级代码合并冲突消解

> Repository-Level Code Merge Conflict Resolution Based on Historical Resolution Retrieval and Large Language Models

## 目录

- [项目概述](#项目概述)
- [方法概述](#方法概述)
- [仓库结构](#仓库结构)
- [环境与依赖](#环境与依赖)
- [配置说明](#配置说明)
- [数据说明](#数据说明)
- [复现流程](#复现流程)
- [模块与接口说明](#模块与接口说明)
- [安全与密钥管理](#安全与密钥管理)
- [引用与致谢](#引用与致谢)

---

## 项目概述

在软件开发过程中，多个开发者并行修改同一代码库并合并分支时，经常会产生代码合并冲突（merge conflict）。人工消解冲突费时费力且容易出错。本项目提出一种**基于历史消解记录的仓库粒度（repository-level）合并冲突自动消解方法**：

1. 以单个代码仓库为单位，收集该仓库历史上的所有合并冲突及其人工消解结果，构建仓库专属的**数据检索源**（基于 Milvus 云端向量数据库）。
2. 对一条待消解的合并冲突，通过**稀疏检索（BM25 / 词项内积）与稠密检索（CodeT5 / CodeBERT 编码）的混合检索 + RRF / 加权重排序**，检索出最相似的历史冲突消解示例。
3. 将检索到的历史示例与当前冲突块融合，构造增强提示（prompt），调用**大语言模型**（OpenAI GPT / DeepSeek / 阿里云百炼 Qwen）生成符合仓库风格的消解代码。

该方法整体流程对应论文算法 4.1，核心组件包含两部分：**历史相似冲突消解记录的检索**（数据检索源构建 + 混合检索模块）与**大语言模型驱动的冲突消解方案生成**。

---

## 方法概述

```text
Input:
    history_tuple   —— 代码仓库中所有历史冲突元组（冲突块、解决方案、全局唯一标识）
    conflict        —— 待消解的合并冲突块
Output:
    resolution_code —— 消解后的完整代码

1. IF history_tuple 或 conflict 为空 THEN RETURN fault
2. id_list, conflict_list, resolution_list = preprocess(history_tuple)
3. source = create_retrieval_source(id_list, conflict_list, resolution_list, BM25, CodeT5)  # 历史检索源构建
4. history_conflict_pair = RRF(conflict, CodeT5, source)                                      # 混合检索
5. prompt = prepare(conflict, history_conflict_pair)                                          # 提示填充
6. resolution = LLM(prompt)                                                                   # 调用大语言模型
7. RETURN postprocess(resolution)
```

### 1. 数据检索源的构建

- 将所选数据集中的合并冲突按**仓库粒度**归类，提取每个仓库的冲突元组（冲突块 `chunk_content`、解决方案 `chunk_resolution`、全局唯一标识 `global_id`）。
- 定义向量数据库集合（collection）模式，字段包括：原生冲突块内容、对应解决方案、稀疏嵌入向量、稠密嵌入向量（768 维，由 CodeT5 编码）以及全局唯一标识。
- 插入数据时，使用预训练代码模型 **CodeT5** 对冲突块编码生成稠密向量，并使用 **BM25** 生成稀疏向量，同时为其建立索引以提升检索效率。

### 2. 文本与向量的混合检索

采用**双通道检索 + 重排序**策略：

- **稀疏检索**：基于 BM25 / 词项内积，评估关键词频率与分布的相关性；
- **稠密检索**：使用 CodeT5 / CodeBERT 将冲突转换为密集向量，度量语义相似度；
- **融合重排**：使用 **RRF（Reciprocal Rank Fusion）** 或 **WeightedRanker** 对两路结果融合排序，输出按相似度排序的历史消解记录列表。

### 3. 提示设计与 LLM 消解

提示由三部分组成：

1. **任务指示（Task Instructions）**；
2. **参考示例（Reference Example）**：检索出的最相似历史冲突消解案例（数量 n 由实验确定）；
3. **待消解冲突（Merge Conflict to be Resolved Next）**。

调用大语言模型 API 生成消解方案后，通过正则匹配提取文本中最后一个由 ` ``` ` 包裹的代码块并去除首尾空白，得到纯净的消解代码。

---

## 仓库结构

```
repoMerge/
├── README.md                           # 本文档
├── requirements.txt                    # 依赖清单
├── .env.example                        # 环境变量模板（密钥占位）
├── .gitignore                          # 忽略规则
│
├── utils.py                            # 工具库：Git 命令、diff 标记处理、结果提取、prompt 模板
├── openai_api.py                       # OpenAI（GPT-3.5 / GPT-4o）调用接口
├── deepseek_api.py                     # DeepSeek 调用接口
├── alibaba_api.py                      # 阿里云百炼（Qwen）调用接口
│
├── database_api_T5_BM25_L2.py          # 检索 API：CodeT5 + BM25(稀疏) + L2(稠密) + RRF
├── database_api_T5_BM25_L2_threshold.py    # 变体：T5-BM25-L2 + 质量阈值过滤
├── database_api_T5_BM25_L2_dense_only.py   # 变体：T5-BM25-L2 仅稠密检索（消融）
├── database_api_T5_IP_COSINE.py        # 检索 API：CodeT5 + BM25(稀疏) + COSINE(稠密) + WeightedRanker
├── database_api_T5_IP_COSINE_dense_only.py # 变体：T5-IP-COSINE 仅稠密（WeightedRanker(0,1)）
├── database_api_T5_IP_COSINE_sparse_only.py # 变体：T5-IP-COSINE 仅稀疏（WeightedRanker(1,0)）
├── database_api_CB_IP_COSINE.py        # 检索 API：CodeBERT + BM25(稀疏) + COSINE(稠密) + WeightedRanker
│
├── data-process.ipynb                  # ① 数据集处理（归类、清洗、划分）
├── vector-database-T5-BM25-L2.ipynb    # ② 构建向量检索源（T5-BM25-L2 变体）
├── vector-database-T5-IP-COSINE.ipynb  # ② 构建向量检索源（T5-IP-COSINE 变体）
├── vector-database-CB-IP-COSINE.ipynb  # ② 构建向量检索源（CB-IP-COSINE 变体）
├── reranker.ipynb                      # ③ 主流程串联（检索 + LLM 消解）与重排序实验
├── reranker_all.ipynb                  # ③ 重排序全流程实验
│
├── dataset/                            # 原始数据集（git 忽略，来源：MergeBERT / 50-repo）
├── dataset_all/                        # 仓库级合并冲突数据（全量、排序、top/bottom 切片）
├── dataset_split/                      # 划分后的数据（train / test / vector）
└── dataset_rerank/                     # 重排实验数据（train / test / vector）
```

> **说明**：`dataset/` 为原始下载数据，体积较大，默认通过 `.gitignore` 排除；模型权重（`codeT5/`、`codebert/`）在首次运行时会自动下载或从本地缓存加载，同样不入库。

---

## 环境与依赖

- Python 3.9+
- 依赖安装：

```bash
pip install -r requirements.txt
```

主要依赖：

| 依赖 | 用途 |
| --- | --- |
| `openai` | 通过 OpenAI 兼容接口调用 GPT / DeepSeek / Qwen |
| `pymilvus` | 连接 Milvus / Zilliz Cloud 向量数据库，混合检索 |
| `transformers` | 加载 CodeT5 / CodeBERT 编码模型 |
| `torch` | 深度学习推理框架 |
| `pandas` / `numpy` | 数据处理 |
| `colorama` / `tqdm` | 终端输出与进度显示 |

---

## 配置说明

所有密钥与连接信息均通过**环境变量**读取，请复制 `.env.example` 为 `.env` 并填入真实值，或在运行前于终端中导出：

```bash
# 大语言模型
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DASHSCOPE_API_KEY="..."
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# 向量数据库
export MILVUS_CLUSTER_ENDPOINT="..."
export MILVUS_TOKEN="..."
```

各代码模块中对应读取的环境变量：

| 文件 | 读取的环境变量 | 默认值 |
| --- | --- | --- |
| `openai_api.py` | `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 占位符 / `https://api.openai-proxy.org/v1` |
| `deepseek_api.py` | `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | 占位符 / `https://api.deepseek.com` |
| `alibaba_api.py` | `DASHSCOPE_API_KEY` / `DASHSCOPE_BASE_URL` | 占位符 / `https://dashscope.aliyuncs.com/compatible-mode/v1` |

> Notebook 中的 `CLUSTER_ENDPOINT` 与 `TOKEN` 变量已统一替换为占位符 `YOUR_MILVUS_ENDPOINT` / `YOUR_MILVUS_TOKEN`，运行前请替换为真实值（或从环境变量注入）。

---

## 数据说明

本项目使用的数据源自公开的代码合并冲突数据集（**MergeBERT** 与 **50-repo**），并按仓库粒度重新组织。数据分布在四个目录中：

- **`dataset/`（原始数据）**：按语言/来源组织的原始合并冲突数据，包含 `mergebert/`（`json`、`json_cs`、`json_js`、`json_ts`、`vector`）与 `50repo/`（`json`、`vector`）。
- **`dataset_all/`（仓库级全量数据）**：每个仓库一个 JSON（如 `spring_time.json`、`orientdb_time.json` 等），以及 `merged_sorted.json`、`merged_top80.json`、`merged_bottom20.json` 等聚合切片。
- **`dataset_split/`（划分数据）**：`train/`、`test/` 下的 `mergebert_20/40/60/80.json`，以及 `vector/` 下的 `*_Conflicts.txt`（制表符分隔的冲突文本，供 BM25 拟合）。
- **`dataset_rerank/`（重排实验数据）**：`train/`（`merged_top20/40/60/80.json`）、`test/`（`merged_bottom20/40/60/80.json`）与 `vector/`。

单条冲突记录的核心字段包括：冲突块内容 `chunk_content`、消解结果 `chunk_resolution`、全局唯一标识 `global_id`、时间戳 `chunk_timestamp`、组/仓库标识 `group_id` 等。

---

## 复现流程

完整实验流程由 3 个阶段组成，分别对应编号的 Notebook：

### ① 数据预处理 —— `data-process.ipynb`

- 从原始数据中按仓库提取合并冲突元组；
- 清洗冲突块（如统一 diff 标记 `<<<<<<<` / `=======` / `>>>>>>>`）；
- 生成 `dataset_all`、`dataset_split`（train/test/vector）等数据集。

### ② 构建向量检索源 —— `vector-database-*.ipynb`

- 连接 Milvus，创建 collection 模式（稀疏 + 稠密 + 全局 id 等字段）并建立索引；
- 使用 CodeT5 / CodeBERT 对冲突块编码，插入向量库；
- 提供三种检索变体：`T5-BM25-L2`、`T5-IP-COSINE`、`CB-IP-COSINE`。

### ③ 主流程串联与重排序实验 —— `reranker.ipynb` / `reranker_all.ipynb`

- 调用 `database_api_*.py` 中的 `query_similar` / `query_similar_inAll` 检索相似历史消解记录；
- 通过 `utils.py` 中的 prompt 模板构造增强提示；
- 调用 `openai_api.py` / `deepseek_api.py` / `alibaba_api.py` 生成消解代码，并提取最终结果；
- 对检索结果进行重排序、对比与评估（支持断点续跑，复用已有输出）。

> 运行 Notebook 时，请从仓库根目录启动 Jupyter，以保证 `utils`、`database_api_*`、`openai_api` 等模块可被正确导入。

---

## 模块与接口说明

### `utils.py`

| 函数 | 作用 |
| --- | --- |
| `run_git_command` / `get_parent_hashes` / `get_merge_base` / `get_commit_message` / `get_commit_time` | Git 合并提交元信息提取 |
| `execute` / `execute_with_info` | 分析合并提交，返回父提交与公共祖先的 message 与时间 |
| `rewrite_DiffMarks` | 将结果中的 diff 标记统一为不带文件地址的形式 |
| `extract_resolved_code` | 提取生成文本中最后一个 ``` ``` ``` 代码块 |
| `simplePrompt` / `prompt_message` / `prompt1` ~ `prompt5` | 各版本 prompt 模板 |

### 检索模块

| 文件 | 稀疏检索 | 稠密检索 | 重排序 |
| --- | --- | --- | --- |
| `database_api_T5_BM25_L2.py` | BM25 | CodeT5 + L2 | RRF |
| `database_api_T5_BM25_L2_threshold.py` | BM25 | CodeT5 + L2 | RRF + 质量阈值过滤 |
| `database_api_T5_BM25_L2_dense_only.py` | —（仅稠密） | CodeT5 + L2 | WeightedRanker(1) |
| `database_api_T5_IP_COSINE.py` | BM25（词项内积 IP） | CodeT5 + COSINE | WeightedRanker |
| `database_api_T5_IP_COSINE_dense_only.py` | —（仅稠密） | CodeT5 + COSINE | WeightedRanker(0,1) |
| `database_api_T5_IP_COSINE_sparse_only.py` | BM25（词项内积 IP） | —（仅稀疏） | WeightedRanker(1,0) |
| `database_api_CB_IP_COSINE.py` | BM25（词项内积 IP） | CodeBERT + COSINE | WeightedRanker |

主要函数：

- `query_similar(CLUSTER_ENDPOINT, TOKEN, vector_database, query, limit[, current_ts])`：仓库内检索相似冲突，支持按时间上界过滤。
- `query_similar_inAll(CLUSTER_ENDPOINT, TOKEN, vector_database, query, limit, group_id)`：跨仓库检索（过滤掉本组内的冲突）。

### LLM 调用模块

| 文件 | 模型 | 函数 |
| --- | --- | --- |
| `openai_api.py` | `gpt-3.5-turbo` / `gpt-4o` | `gpt_35` / `gpt_4o` |
| `deepseek_api.py` | `deepseek-v4-pro` | `deepSeek` |
| `alibaba_api.py` | `qwen-turbo` | `qwen` |

三者接口一致：`func(question, number, truth)`，返回消解后的代码列表。

---

## 实验参数配置

### 大语言模型生成参数

| 模型 | 调用入口 | temperature | max_tokens |
| --- | --- | --- | --- |
| `gpt-3.5-turbo` | `openai_api.gpt_35` | 1.99 | 4096 |
| `gpt-4o` | `openai_api.gpt_4o` | 1.99 | 4096 |
| `deepseek-v4-pro` | `deepseek_api.deepSeek` | 1.99 | 4096 |
| `qwen-turbo` | `alibaba_api.qwen` | 1.99 | 4096 |

> temperature 取值区间为 0~2，越大越随机、越小越确定。

### 检索与重排序参数

| 检索变体 | 稀疏通道 | 稠密通道 | 重排序方式 | 编码模型 |
| --- | --- | --- | --- | --- |
| T5-BM25-L2 | BM25 | CodeT5 + L2 | RRF | CodeT5 |
| T5-BM25-L2（阈值过滤） | BM25 | CodeT5 + L2 | RRF + 质量阈值过滤 | CodeT5 |
| T5-BM25-L2（仅稠密） | — | CodeT5 + L2 | WeightedRanker(1) | CodeT5 |
| T5-IP-COSINE | BM25（IP） | CodeT5 + COSINE | WeightedRanker(0.2, 0.8) | CodeT5 |
| T5-IP-COSINE（仅稠密） | — | CodeT5 + COSINE | WeightedRanker(0, 1) | CodeT5 |
| T5-IP-COSINE（仅稀疏） | BM25（IP） | — | WeightedRanker(1, 0) | CodeT5 |
| CB-IP-COSINE | BM25（IP） | CodeBERT + COSINE | WeightedRanker(0, 1) | CodeBERT |

其他关键参数：

- 稠密向量维度：**768**（CodeT5 / CodeBERT 隐层维度）
- 检索返回条数 `limit`：由各实验设定（top-k 通常为 1~6）
- 时间上界过滤 `current_ts`：数值时间戳或 ISO 8601 字符串
- 质量阈值过滤（`database_api_T5_BM25_L2_threshold.py`）：
  - `enable_threshold`：是否启用（默认 False）
  - `bm25_threshold`：BM25 最低分值（默认 0.0）
  - `l2_threshold`：L2 最大距离（默认 ∞）
  - `rrf_threshold`：RRF 融合分值最低阈值（默认 0.0）

---

## 安全与密钥管理

- 所有 API 密钥、数据库令牌（token）均**不得硬编码**在源码中，必须通过环境变量注入，参考 `.env.example`。
- `.env` 文件已被 `.gitignore` 排除，请勿将其提交到公开仓库。
- 仓库中的 `YOUR_*` 字符串均为占位符，可安全公开。

---

## 引用与致谢

本工作使用了以下公开资源：

- 数据集：MergeBERT（合并冲突数据集）、50-repo；
- 预训练模型：[Salesforce/CodeT5](https://huggingface.co/Salesforce/codet5-base)、[microsoft/CodeBERT](https://huggingface.co/microsoft/codebert-base)；
- 向量数据库：[Milvus](https://milvus.io/) / Zilliz Cloud；
- 大语言模型：OpenAI GPT、DeepSeek、阿里云百炼 Qwen。

> 若使用本项目，请按相应数据源与模型的许可协议进行引用。
