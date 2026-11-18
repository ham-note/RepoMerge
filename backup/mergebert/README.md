# MergeBERT：复现实现与审稿基线结果

公开范围为脱敏源码、汇总结果与复跑说明；完整数据、特征、逐条预测和权重不上传。

| 当前云端结果 | 匹配 | 已评估 | 缺失预测计失败 | 全量测试 | 微平均EM | 宏平均EM |
|---|---:|---:|---:|---:|---:|---:|
| 作者确认的运行 | 549 | 1591 | 72 | 1663 | 33.01% | 33.43% |

分仓库表与统计口径见 [RESULTS.md](RESULTS.md)，中文回复见 [REVIEW_RESPONSE_ZH.md](REVIEW_RESPONSE_ZH.md)。云端使用本地上传脚本，因GPU限制分两次执行，不是独立随机种子重复实验。

本公开包只报告当前云端汇总，不附旧本地实验结果。下述命令是公开适配实现的复跑说明，不替代私有云端运行记录。实现差异见 [PROTOCOL.md](PROTOCOL.md)。

## 公开内容

- `colab_reference/mergebert.ipynb`：去除输出、执行计数和附加元数据的参考notebook。
- `colab_reference/cell_*.py`：参考单元源码；模型定义和核心对齐/推理逻辑保持一致。
- `colab_reference/scripts/`：缓存预处理、全量九分类标签、训练、全量评估、核验和数据导入脚本。
- `reports/cloud_confirmed_1663_20261114/`：当前云端汇总；无逐样本数据。
- `examples/synthetic_raw.json`：手工编写的合成格式样例，不来自实验数据。

## 环境

Python 3.11，Git 2.45.2（含`merge-file --diff-algorithm=histogram`），可用CUDA的PyTorch。原实验使用PyTorch 2.6.0+cu124，batch8、AMP、梯度检查点。安装适合本机的PyTorch后，可安装`requirements.txt`中其余依赖。本目录不需要外部LLM服务或API密钥。

以下命令从`backup/mergebert/`执行：

```bash
python -m pip install -r requirements.txt
python colab_reference/scripts/download_model.py
```

下载的是公开的`huggingface/CodeBERTa-small-v1`预训练编码器，脚本核对五个文件的内容哈希；不是本地训练后的模型。如果已有同一编码器，可设置`MERGEBERT_MODEL_PATH`直接使用，不必下载。Git不在PATH时可设置`MERGEBERT_GIT`。

## 导入自己的私有数据并复跑

数据格式见合成样例：JSON数组，每条具有`uid`、`repository`和`raw`；`raw`须有`a_contents`、`b_contents`、`o_contents`、`res_region`字符串，可选`res_label`。导入器不删样本、不自动去重；同一子集内UID必须唯一。

```bash
python colab_reference/scripts/import_data.py --split sort_by_prerepo --train /path/to/prerepo_train.json --test /path/to/prerepo_test.json
python colab_reference/scripts/prepare_all.py --split sort_by_prerepo
python colab_reference/scripts/run_colab.py --split sort_by_prerepo
```

请使用已有的正确划分，不要把合成样例当实验数据。该导入入口不重新推断时间划分；原实验的时间协议和数量见PROTOCOL.md。需要原实验数据才能复核原数值；本PR不声称仅凭公开样例就能重现表中准确率。

默认输出在`runtime/colab_reproduction_sort_by_prerepo`；共享缓存位于`runtime/shared/`。可设置`MERGEBERT_RUN_ROOT`指定独立运行目录。新实验应使用空运行目录；入口会跳过已存在的模型或指标，避免无意覆盖。

每套保存格式化训练/测试数据、逐条标签审计、训练模型与逐条预测。测试集也进行分词和合并预处理，但不生成监督分类标签，真值仅用于最终计分。`verify_final.py`复算全量数量和EM并核验最终模型哈希。

## 公开前检查

```bash
python -m unittest discover -s tests -v
```

隐私范围和扫描说明见 [PRIVACY.md](PRIVACY.md)。完整私有实验不作为本PR的提交附件。
