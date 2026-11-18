# R1-A28 Manual Audit Summary
## Sampling and Audit Setup
- Sample size: 100 exact-match failed cases
- Sampling: repository-stratified proportional random sampling
- Random seed: 20261003
- Audited prediction: top-1 candidate (`_candidates[0]`)
- Build environment: not used; labels are based on developer-style visual code comparison of gold, prediction, diff, and conflict context

## Category Distribution
| Category | Count | Percentage |
|---|---:|---:|
| Semantically equivalent: formatting/comments | 16 | 16.0% |
| Semantically equivalent: renaming/syntax/API-equivalent | 7 | 7.0% |
| Plausible but incomplete | 3 | 3.0% |
| Truly incorrect | 65 | 65.0% |
| Benchmark noise / tangled fragment | 9 | 9.0% |
| **Total** | **100** | **100.0%** |

## Main Takeaways
- Semantically equivalent exact-match false negatives: **23/100 (23.0%)**.
- Plausible but incomplete predictions: **3/100 (3.0%)**.
- Benchmark noise or tangled-fragment cases: **9/100 (9.0%)**. These should be reported separately rather than counted as model-correct.
- Truly incorrect predictions: **65/100 (65.0%)**.

## Repository Breakdown
| Repository | Total | Semantic equivalent | Partial | Benchmark noise | Incorrect |
|---|---:|---:|---:|---:|---:|
| Galacticraft | 9 | 2 | 0 | 0 | 7 |
| SOS | 16 | 10 | 0 | 0 | 6 |
| autopsy | 8 | 3 | 0 | 0 | 5 |
| autorest | 2 | 0 | 2 | 0 | 0 |
| azure-sdk-for-java | 12 | 2 | 1 | 1 | 8 |
| cloudstack | 4 | 1 | 0 | 1 | 2 |
| molgenis | 9 | 0 | 0 | 0 | 9 |
| orientdb | 6 | 0 | 0 | 5 | 1 |
| rumble | 19 | 0 | 0 | 0 | 19 |
| sis | 6 | 0 | 0 | 0 | 6 |
| spring-boot | 9 | 5 | 0 | 2 | 2 |

## Suggested Paper/Rebuttal Wording
> We conducted a manual semantic audit on 100 randomly sampled exact-match failed cases using repository-stratified sampling. For each case, we compared the top-1 prediction with the developer resolution under the original conflict context. The audit shows that 23/100 exact-match failures are semantically equivalent to the developer resolution, indicating that exact match is a conservative metric. We also observed 9/100 cases caused by benchmark noise or tangled/partial conflict-region boundaries; these are reported separately and are not counted as model-correct.


## Files

- `sample_100_labeled.json`: 100 sampled cases with gold resolution, top-1 prediction, diff, and manual labels.
- `summary.json`: aggregate counts and repository breakdown used in the response.
