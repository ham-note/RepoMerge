from common import *
import argparse,random,torch
from transformers import AutoTokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument('split');args=p.parse_args();root=BASE/('colab_reproduction_'+args.split)
 ref=module('colab_dataset_original',REFERENCE/'cell_1.py');tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);ds=ref.MergeDataset(str(root/'data/data_train.json'),tokenizer)
 indices=list(range(len(ds)));random.Random(42).shuffle(indices);checked=[]
 for i in indices:
  row=ds.samples[i]
  lengths=[len(tokenizer.tokenize(row[k])) for k in ['a_text','b_text','o_text']]
  if max(lengths)>180:continue
  actual=ds[i];expected=row['train_features'];keys=['a_o_ids','o_a_ids','b_o_ids','o_b_ids']
  assert all(actual[k].tolist()==expected[j] for j,k in enumerate(keys));assert int(actual['labels'])==row['numeric_label'];checked.append(row['uid'])
  if len(checked)==30:break
 assert len(checked)==30
 report={'original_colab_dataset_items_compared':len(checked),'all_four_features_equal':True,'uids':checked,'uses_unmodified_original_compute':True}
 (root/'reports/feature_parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('FEATURE_PARITY_PASSED',len(checked),flush=True)
if __name__=='__main__':main()