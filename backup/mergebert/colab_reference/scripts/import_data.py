from common import BASE
import argparse,json,pathlib

def main():
 p=argparse.ArgumentParser(description="Import privately held, already split data without filtering")
 p.add_argument('--split',required=True,choices=['sort_by_prerepo','sort_by_realtime']);p.add_argument('--train',required=True,type=pathlib.Path);p.add_argument('--test',required=True,type=pathlib.Path);a=p.parse_args()
 root=BASE/('colab_reproduction_'+a.split)
 for d in ['data','reports','logs','checkpoints','scripts','source']:(root/d).mkdir(parents=True,exist_ok=True)
 for label,path in [('80',a.train),('20',a.test)]:
  rows=json.loads(path.read_text(encoding='utf-8'));assert isinstance(rows,list) and rows,'Expected nonempty JSON array'
  assert len({r['uid'] for r in rows})==len(rows),'UIDs must be unique within each split'
  for r in rows:
   assert isinstance(r['repository'],str)
   for key in ['a_contents','b_contents','o_contents','res_region']:assert isinstance(r['raw'][key],str)
  (root/'data'/('raw_'+label+'.json')).write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
  print(a.split,label,len(rows))
if __name__=='__main__':main()
