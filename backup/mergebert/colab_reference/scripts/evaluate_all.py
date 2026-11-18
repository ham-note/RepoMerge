from common import *
import argparse,types,time,collections,csv,torch
from transformers import AutoTokenizer

def summarize(rows):
 hits=sum(x['exact_match'] for x in rows);return {'matches':hits,'total':len(rows),'em':hits/len(rows) if rows else None,'statuses':dict(collections.Counter(r['status'] for r in rows))}
def main():
 p=argparse.ArgumentParser();p.add_argument('split',choices=['sort_by_prerepo','sort_by_realtime']);args=p.parse_args();root=BASE/('colab_reproduction_'+args.split)
 enable_fast_alignment();torch.set_num_threads(4)
 global infer_ref
 infer_ref=module("colab_inference_full",REFERENCE/"cell_2.py");infer_ref.compute=fast_compute
 ref=types.ModuleType('colab_eval_reference');ref.__dict__.update(infer_ref.__dict__);ref.__dict__['__name__']='colab_eval_reference'
 exec(compile((REFERENCE/'cell_7.py').read_text(encoding='utf-8'),str(REFERENCE/'cell_7.py'),'exec'),ref.__dict__)
 model=infer_ref.MergeBERTModel(str(MODEL),9);model.load_state_dict(torch.load(root/'checkpoints/mergebert.pt',map_location='cpu',weights_only=True));model.cuda().eval();tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 observed={}
 def hook(m,inputs,output):
  observed['label']=int(output['logits'].argmax(-1));observed['probabilities']=output['logits'].softmax(-1)[0].detach().cpu().tolist()
  observed['inference_features']=[x[0].detach().cpu().tolist() for x in inputs]
 model.register_forward_hook(hook)
 raw=json.loads((root/'data/raw_20.json').read_text(encoding='utf-8'));records=json.loads((root/'data/data_test.json').read_text(encoding='utf-8'));by_uid={r['uid']:r for r in records};results=[];start=time.time()
 with (root/'reports/predictions_20.jsonl').open('w',encoding='utf-8') as f:
  for i,item in enumerate(raw):
   record=by_uid[item['uid']];c=json.loads((FEATURE_CACHE/(record['input_sha256']+'.json')).read_text(encoding='utf-8'));hunks=deserialize_hunks(c['hunks']);observed.clear()
   def cached_merge(a,b,o):
    assert a==c['a_tokens'] and b==c['b_tokens'] and o==c['o_tokens'];return hunks
   ref.token_level_merge=cached_merge
   answers=ref.get_topk_predictions(model,tokenizer,torch.device('cuda'),record['a_text'],record['b_text'],record['o_text'],k=1)
   prediction=answers[0];gold=item['raw']['res_region'];status='classifier' if c['conflict_count']==1 else 'clean_merge' if c['conflict_count']==0 else 'multiple_conflict_colab_fallback'
   if not any(c['token_lengths']):status='empty_input'
   row={'uid':item['uid'],'repository':item['repository'],'answers':answers,'prediction':prediction,'gold':gold,'prediction_normalized':normalize(prediction),'gold_normalized':normalize(gold),'exact_match':normalize(prediction)==normalize(gold),'colab_whitespace_exact_match':''.join(prediction.split())==''.join(gold.split()),'raw_exact_match':prediction==gold,'status':status,'conflict_count':c['conflict_count'],**observed}
   results.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
   if (i+1)%100==0:print('EVAL',i+1,'/',len(raw),'seconds',round(time.time()-start),flush=True)
 repos=sorted({r['repository'] for r in results}|{r['repository'] for r in json.loads((root/'data/raw_80.json').read_text(encoding='utf-8'))});metrics={repo:summarize([r for r in results if r['repository']==repo]) for repo in repos};metrics['OVERALL']=summarize(results)
 for repo in repos:
  subset=[r for r in results if r['repository']==repo];name=repo.replace('https://github.com/','').replace('/','__');(root/'reports'/('predictions_'+name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset),encoding='utf-8')
 (root/'reports/metrics_by_repository.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
 with (root/'reports/metrics_by_repository.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['repository','matches','total','EM_percent'])
  for repo,m in metrics.items():w.writerow([repo,m['matches'],m['total'],None if m['em'] is None else m['em']*100])
 history=json.loads((root/'reports/training_history.json').read_text());cfg=json.loads((root/'reports/run_config.json').read_text());stats=json.loads((root/'reports/preparation_stats.json').read_text());macro=sum(metrics[r]['em'] for r in repos if metrics[r]['total'])/sum(bool(metrics[r]['total']) for r in repos)
 lines=['# Colab MergeBERT 全量复现：'+args.split,'','以用户提供的 `colab_reference/mergebert.ipynb` 为主要参考。原件和逐单元源码位于 source/。','', '## 分仓库 Exact Match','','| Repository | Matches | Total | EM (%) |','|---|---:|---:|---:|']
 for repo,m in metrics.items():
  value='N/A' if m['em'] is None else f"{m['em']*100:.2f}";lines.append(f"| {repo.replace('https://github.com/','')} | {m['matches']} | {m['total']} | {value} |")
 lines+=['',f'Macro Average EM：{macro*100:.2f}%。OVERALL 为全量测试集的微平均；零测试样本仓库不纳入宏平均。','', '## 训练与标签','', '全部 80 数据均参与每轮训练，不执行 backup 的长度、可表达性、冲突数量筛选。使用 Colab 的九分类、四路编辑操作输入、CodeBERTa-small-v1、batch 8、AMP、10 epochs、AdamW 5e-5 和无 warmup 线性衰减。','', '用户已确认新增近似标签规则：九候选中选择与真值归一化后字符 Levenshtein 距离最小的操作，距离为零标记精确标签，否则标记近似标签；并列取最小标签编号。单 token 冲突时按 Colab 的冲突局部重建完整候选；零个/多个冲突时使用完整 A/B/O 的九操作作为训练代理候选。后者并非 Colab 已定义的训练规则，不能声称完全复现了缺失的标签脚本。详见逐条 label_audit.jsonl。','', '为满足全部 80 数据训练，未额外划出验证集，固定使用第 10 轮模型；不同于 Colab 的验证集最优模型选择。训练准确率仅为训练监控，不能当作验证或测试成绩。','', '## 推理与指标','', '实际调用 notebook cell 7 的 get_topk_predictions，保留其训练/推理输入不一致、A/B 分支解析顺序、无冲突直接返回合并结果、多冲突处理异常后返回空串的行为。所有测试样本保留；错误返回空串仍按 notebook 输出计分，因此空真值可能匹配。预测状态已逐条记录。','', '主指标按用户要求删除 `<s>` 与全部空白后逐字符比较完整答案；不使用 notebook 的编辑距离四舍五入作为严格 EM。另保留仅去空白和原文精确匹配标记。TOP_K=1，只报告 Top-1，不将同一候选包装成 Top-3/5。','', '## 保存的产物','','- data/raw_80.json、raw_20.json：全量划分原件。','- data/data_train.json：全部训练记录、标签及四路512输入；data_test.json：全部测试记录。','- reports/label_audit.jsonl：候选距离、精确/近似标签、候选作用范围。','- checkpoints/mergebert.pt：固定第10轮最终权重。本地训练还会保存逐轮权重和续训状态；这些文件不属于公开提交。','- reports/predictions_20.jsonl 及分仓库 JSONL：预测、真值、状态和 EM。','- 共享 runtime/shared/feature_cache 与 merge_artifacts：确定性 token、编辑序列、合并输入输出，记录包含来源路径。','- 包内 colab_reference/ 参考 notebook；scripts/ 和共享 scripts/：本地适配源码；logs/：执行日志。','', '## 配置与统计','','```json',json.dumps(cfg,ensure_ascii=False,indent=2),'```','','```json',json.dumps(stats,ensure_ascii=False,indent=2),'```','','```json',json.dumps(history,indent=2),'```','']
 (root/'reports/RESULTS.md').write_text('\n'.join(lines),encoding='utf-8');print('EVALUATION_COMPLETE',json.dumps(metrics['OVERALL']),flush=True)
if __name__=='__main__':main()