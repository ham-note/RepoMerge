import os
os.environ['USE_TORCH']='0'
from common import *
import concurrent.futures,collections,time
from rapidfuzz.distance import Levenshtein
TOK=None
CACHE=FEATURE_CACHE
def init():
 from transformers import AutoTokenizer
 global TOK
 enable_fast_alignment();TOK=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
def encode(seq):
 ids=TOK.convert_tokens_to_ids(seq);return ids[:512]+[TOK.pad_token_id]*max(0,512-len(ids))
def process(item):
 raw=item['raw'];texts=[raw[k] for k in ['a_contents','b_contents','o_contents']];key=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest();path=CACHE/(key+'.json')
 if path.exists():return key
 a,b,o=[TOK.tokenize(t) for t in texts]
 four=[train_ref.get_edit_sequence(x,y)[0] for x,y in [(o,a),(a,o),(o,b),(b,o)]]
 train_features=[encode(x) for x in four]
 directory=MERGE_ARTIFACTS/key
 hunks=token_merge(a,b,o,directory);conflicts=[h for h in hunks if isinstance(h,tuple)]
 if len(conflicts)==1:
  idx=next(i for i,h in enumerate(hunks) if isinstance(h,tuple));prefix=[t for h in hunks[:idx] for t in h];suffix=[t for h in hunks[idx+1:] for t in h];ca,cb,co=conflicts[0]
  candidates=[TOK.convert_tokens_to_string(prefix+infer_ref.apply_pattern(p,ca,cb,co)+suffix) for p in PATTERNS];scope='inference_conflict_hunk'
 else:
  candidates=[TOK.convert_tokens_to_string(infer_ref.apply_pattern(p,a,b,o)) for p in PATTERNS];scope='whole_region_training_surrogate'
 record={'input_sha256':key,'token_lengths':[len(a),len(b),len(o)],'a_tokens':a,'b_tokens':b,'o_tokens':o,'edit_sequences':four,'train_features':train_features,'hunks':serialize_hunks(hunks),'conflict_count':len(conflicts),'label_candidates':candidates,'label_scope':scope,'merge_artifacts':directory.relative_to(BASE).as_posix()}
 path.write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8');return key

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument("--split",choices=["sort_by_prerepo","sort_by_realtime"],default="sort_by_prerepo");args=parser.parse_args()
 parity=json.loads((REFERENCE/'runtime_parity.json').read_text());assert parity['alignment_exact'] and parity['merge_mismatch_count']==0
 CACHE.mkdir(parents=True,exist_ok=True);roots=[BASE/('colab_reproduction_'+s) for s in [args.split]]
 unique={}
 for root in roots:
  for split in ['80','20']:
   for r in json.loads((root/'data'/f'raw_{split}.json').read_text(encoding='utf-8')):
    texts=[r['raw'][k] for k in ['a_contents','b_contents','o_contents']];key=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest();unique[key]=r
 pending=[r for k,r in unique.items() if not (CACHE/(k+'.json')).exists()];print('UNIQUE_INPUTS',len(unique),'PENDING',len(pending),flush=True);start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
  for i,k in enumerate(pool.map(process,pending)):
   if (i+1)%100==0:print('FEATURES',i+1,'/',len(pending),'seconds',round(time.time()-start),flush=True)
 for root in roots:
  audits=[];statistics={}
  for split in ['80','20']:
   records=json.loads((root/'data'/f'raw_{split}.json').read_text(encoding='utf-8'));rows=[];counts=collections.Counter()
   for r in records:
    raw=r['raw'];texts=[raw[k] for k in ['a_contents','b_contents','o_contents']];key=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest();c=json.loads((CACHE/(key+'.json')).read_text(encoding='utf-8'))
    row={'uid':r['uid'],'repository':r['repository'],'a_text':texts[0],'b_text':texts[1],'o_text':texts[2],'resolution':raw['res_region'],'descriptive_label':raw.get('res_label','unknown'),'input_sha256':key,'feature_cache':(CACHE/(key+'.json')).relative_to(BASE).as_posix(),'train_features':c['train_features']}
    counts['total']+=1;counts['conflict_count_'+str(c['conflict_count'])]+=1
    if split=='80':
     target=normalize(raw['res_region']);distances=[Levenshtein.distance(normalize(x),target) for x in c['label_candidates']];label=min(range(9),key=lambda i:(distances[i],i));exact=distances[label]==0
     row.update({'numeric_label':label,'label_exact':exact,'label_distance':distances[label],'label_scope':c['label_scope']});counts['exact_label' if exact else 'approximate_label']+=1;counts['label_'+str(label)]+=1
     audits.append({'uid':r['uid'],'repository':r['repository'],'label':label,'pattern':PATTERNS[label],'label_exact':exact,'distance':distances[label],'all_distances':distances,'label_scope':c['label_scope'],'chosen_candidate':c['label_candidates'][label],'resolution':raw['res_region'],'conflict_count':c['conflict_count']})
    rows.append(row)
   output=root/'data'/('data_train.json' if split=='80' else 'data_test.json');output.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8');statistics[split]=dict(counts)
   assert len(rows)==len(records)
  (root/'reports/label_audit.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in audits),encoding='utf-8')
  (root/'reports/preparation_stats.json').write_text(json.dumps(statistics,indent=2),encoding='utf-8');print('PREPARED',root.name,json.dumps(statistics),flush=True)
 print('ALL_FULL_DATA_PREPARED',flush=True)
if __name__=='__main__':main()