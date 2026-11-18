from common import *
train_ref=module("colab_train_full",REFERENCE/"cell_1.py")
import argparse,time,random,math,torch
from torch.utils.data import Dataset,DataLoader
from transformers import get_linear_schedule_with_warmup

class CachedDataset(Dataset):
 def __init__(self,rows):self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];return tuple(torch.tensor(x,dtype=torch.long) for x in r['train_features'])+(torch.tensor(r['numeric_label'],dtype=torch.long),)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('split',choices=['sort_by_prerepo','sort_by_realtime']);parser.add_argument('--smoke',action='store_true');parser.add_argument('--checkpointing',action='store_true',default=True);parser.add_argument('--benchmark',action='store_true');args=parser.parse_args()
 root=BASE/('colab_reproduction_'+args.split);torch.set_num_threads(4);seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 rows=json.loads((root/'data/data_train.json').read_text(encoding='utf-8'))
 if args.smoke:rows=rows[:8]
 loader=DataLoader(CachedDataset(rows),batch_size=8,shuffle=True,num_workers=0,pin_memory=True)
 model=train_ref.MergeBERTModel(str(MODEL),9).cuda()
 if args.checkpointing:model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5);scheduler=get_linear_schedule_with_warmup(optimizer,0,len(loader)*10)
 cfg={'source':'mergebert.ipynb cell 1','epochs':10,'batch_size':8,'learning_rate':5e-5,'precision':'AMP float16 with float32 weights','seed':seed,'gradient_checkpointing':args.checkpointing,'train_count':len(rows),'external_test_used_for_selection':False,'selection':'fixed final epoch 10, all 80 samples trained; no validation holdout','label_policy':'min normalized character Levenshtein distance across nine candidates, tie smallest label index','training_inputs':'four edit sequences O->A, A->O, O->B, B->O; original tokenizer IDs, padding512; no extra tokens or masks'}
 if not (args.smoke or args.benchmark):(root/'reports/run_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
 print('TRAIN_CONFIG',json.dumps(cfg),flush=True);start=time.time();history=[];start_epoch=1;resume=root/'checkpoints/latest_state.pt'
 if resume.exists() and not (args.smoke or args.benchmark):
  state=torch.load(resume,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);start_epoch=state['epoch']+1;history=state['history'];del state
 for epoch in range(start_epoch,11):
  model.train();scaler=torch.amp.GradScaler('cuda');loss_sum=0.;seen=0;correct=0
  for step,batch in enumerate(loader,1):
   features=[x.cuda(non_blocking=True) for x in batch[:4]];labels=batch[4].cuda(non_blocking=True);optimizer.zero_grad()
   with torch.autocast('cuda',dtype=torch.float16):out=model(*features,labels=labels);loss=out['loss']
   assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();scheduler.step()
   seen+=labels.numel();loss_sum+=loss.item();correct+=(out['logits'].argmax(-1)==labels).sum().item()
   if step%50==0 or args.smoke or args.benchmark:print('STEP',json.dumps({'epoch':epoch,'step':step,'steps':len(loader),'seen':seen,'loss':loss_sum/step,'training_accuracy':correct/seen,'elapsed_s':round(time.time()-start),'gpu_peak_mb':round(torch.cuda.max_memory_allocated()/2**20)}),flush=True)
   if args.benchmark and step==10:
    print('BENCHMARK_COMPLETE',flush=True);return
   if args.smoke:
    assert model.encoder.embeddings.word_embeddings.weight.grad is not None and optimizer.state
    print('SMOKE_OK',flush=True);return
  record={'epoch':epoch,'samples_seen':seen,'loss_mean_batches':loss_sum/len(loader),'training_class_accuracy':correct/seen,'elapsed_s':round(time.time()-start)};assert seen==len(rows);history.append(record)
  torch.save(model.state_dict(),root/'checkpoints'/f'epoch_{epoch:02d}.pt')
  state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'config':cfg}
  tmp=root/'checkpoints/latest_state.tmp';torch.save(state,tmp);tmp.replace(resume)
  (root/'reports/training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');print('EPOCH_COMPLETE',json.dumps(record),flush=True)
 torch.save(model.state_dict(),root/'checkpoints/mergebert.pt')
 with (root/'checkpoints/mergebert.pt').open('rb') as f:final_hash=hashlib.file_digest(f,'sha256').hexdigest()
 (root/'reports/final_model_sha256.json').write_text(json.dumps({'sha256':final_hash,'epoch':10}),encoding='utf-8')
 print('TRAINING_COMPLETE',root.name,flush=True)
if __name__=='__main__':main()