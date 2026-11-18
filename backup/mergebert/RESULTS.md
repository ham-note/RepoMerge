# Current baseline: full original 1,663-record test set

Cloud run ID: `cloud_confirmed_1663_20261114`. Author-confirmed aggregates; arithmetic checked for publication, private per-sample cloud records not independently audited. Cloud used locally uploaded scripts/files, with two executions due to GPU constraints, not two independent seed trials. Missing predictions count as failures; their failure stage is not established here.

| Repository | Matches | 已评估数量 | 未获有效预测、计失败 | 全量测试数量 | 全量Exact Match (%) |
|---|---:|---:|---:|---:|---:|
| Galacticraft_time20 | 12 | 125 | 10 | 135 | 8.89 |
| SOS_time20 | 9 | 154 | 3 | 157 | 5.73 |
| autopsy_time20 | 28 | 123 | 17 | 140 | 20.00 |
| autorest_time20 | 145 | 172 | 0 | 172 | 84.30 |
| azure_time20 | 94 | 178 | 7 | 185 | 50.81 |
| cloudstack_time20 | 45 | 91 | 15 | 106 | 42.45 |
| molgenis_time20 | 12 | 150 | 1 | 151 | 7.95 |
| orientdb_time20 | 100 | 105 | 0 | 105 | 95.24 |
| rumble_time20 | 74 | 275 | 1 | 276 | 26.81 |
| sis_time20 | 3 | 111 | 8 | 119 | 2.52 |
| spring-boot_time20 | 27 | 107 | 10 | 117 | 23.08 |
| **OVERALL** | **549** | **1,591** | **72** | **1,663** | **33.01** |

**Macro Average Exact Match：33.43%**。按11个仓库未四舍五入的准确率求平均。

```text
已评估子集：549 / 1591 × 100% = 34.51%
全量微平均：549 / 1663 × 100% = 33.01%
全量宏平均：(1/11) × Σ(Matches_r / Total_r) × 100% = 33.43%
```

分仓库差额合计72条，由原测试总数减去已评估数量直接得到，没有人为按比例分配。匹配数始终保持549。


## Run provenance

Only the current cloud aggregate is published. Prior local experimental reports are retained in a private local backup and are not part of this package. No new training or split experiment was run for this update. Private cloud prediction records and checkpoint hashes are not published here.

See [Chinese author-review response](REVIEW_RESPONSE_ZH.md). Cross-method normalization and failure handling must agree before comparative claims; no new significance test is claimed.
