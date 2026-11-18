from common import *
import argparse,collections
from rapidfuzz.distance import Levenshtein

def main():
 p=argparse.ArgumentParser();p.add_argument('split');args=p.parse_args();root=BASE/('colab_reproduction_'+args.split)
 raw_train=json.loads((root/'data/raw_80.json').read_text(encoding='utf-8'));train=json.loads((root/'data/data_train.json').read_text(encoding='utf-8'));raw_test=json.loads((root/'data/raw_20.json').read_text(encoding='utf-8'));test=[json.loads(l) for l in (root/'reports/predictions_20.jsonl').read_text(encoding='utf-8').splitlines()];audits=[json.loads(l) for l in (root/'reports/label_audit.jsonl').read_text(encoding='utf-8').splitlines()]
 assert len(train)==len(raw_train)==len(audits);assert {r['uid'] for r in train}=={r['uid'] for r in raw_train};assert len(test)==len(raw_test)==len({r['uid'] for r in test});assert {r['uid'] for r in test}=={r['uid'] for r in raw_test}
 by_raw={r['uid']:r for r in raw_test}
 for r in test:
  assert r['gold']==by_raw[r['uid']]['raw']['res_region'];assert r['exact_match']==(normalize(r['prediction'])==normalize(r['gold']))
 for r,a in zip(train,audits):
  assert r['uid']==a['uid'] and r['numeric_label']==min(range(9),key=lambda i:(a['all_distances'][i],i));assert a['label_exact']==(a['distance']==0);assert Levenshtein.distance(normalize(a['chosen_candidate']),normalize(a['resolution']))==a['distance'];assert len(r['train_features'])==4 and all(len(x)==512 for x in r['train_features'])
 history=json.loads((root/'reports/training_history.json').read_text());assert len(history)==10 and all(r['samples_seen']==len(train) for r in history)
 metrics=json.loads((root/'reports/metrics_by_repository.json').read_text())
 for repo,m in metrics.items():
  subset=test if repo=='OVERALL' else [r for r in test if r['repository']==repo];assert m['total']==len(subset) and m['matches']==sum(r['exact_match'] for r in subset)
 def digest(p):
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 final_model=root/'checkpoints/mergebert.pt'
 final_info=json.loads((root/'reports/final_model_sha256.json').read_text(encoding='utf-8'));assert final_info['epoch']==10 and digest(final_model)==final_info['sha256']
 manifest={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':digest(p)} for folder in ['source','scripts','checkpoints'] for p in (root/folder).glob('*') if p.is_file()}
 (root/'reports/artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');report={'all_training_samples_used':len(train),'all_test_samples_evaluated':len(test),'every_epoch_used_all_training_samples':True,'label_rule_verified':True,'exact_match_recomputed':True,'final_model_sha256_verified':True,'epoch10_provenance':'hash recorded by train_all.py after final epoch 10','metrics':metrics['OVERALL']};(root/'reports/final_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('FINAL_VALIDATION_PASSED',json.dumps(report),flush=True)
if __name__=='__main__':main()