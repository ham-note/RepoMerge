import os,sys,pathlib,importlib.util,json,hashlib,subprocess,tempfile,shutil,numpy as np
PACKAGE=pathlib.Path(__file__).resolve().parents[2]
BASE=pathlib.Path(os.environ.get('MERGEBERT_RUN_ROOT', str(PACKAGE/'runtime'))).resolve()
REFERENCE=PACKAGE/'colab_reference'
FEATURE_CACHE=BASE/'shared/feature_cache'
MERGE_ARTIFACTS=BASE/'shared/merge_artifacts'
MODEL=pathlib.Path(os.environ.get('MERGEBERT_MODEL_PATH', str(BASE/'models/CodeBERTa-small-v1'))).resolve()
GIT=os.environ.get('MERGEBERT_GIT') or shutil.which('git') or 'git'
os.environ['GIT_PYTHON_GIT_EXECUTABLE']=GIT
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTHONUTF8']='1'
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def lightweight_reference(name,file,names):
 import ast,types
 from typing import Callable,List
 tree=ast.parse(file.read_text(encoding='utf-8'));nodes=[]
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:nodes.append(n)
  elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets):nodes.append(n)
 m=types.ModuleType(name);m.__dict__.update({'np':np,'Callable':Callable,'List':List})
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),m.__dict__);return m
alignment_names={'edit_seq_tokens','OffsetRange','SequenceDiff','compute','get_edit_sequence'}
train_ref=lightweight_reference('colab_train_alignment',REFERENCE/'cell_1.py',alignment_names)
infer_ref=lightweight_reference('colab_inference_alignment',REFERENCE/'cell_2.py',alignment_names|{'REVERSE_LABEL_MAP','apply_pattern','parse_conflict_markers','align_and_get_edit_sequence'})
original_compute=train_ref.compute

def fast_compute(sequence1,sequence2,equalityScore=None):
 if equalityScore is not None:return original_compute(sequence1,sequence2,equalityScore)
 n,m=len(sequence1),len(sequence2)
 if not n or not m:return [train_ref.SequenceDiff(train_ref.OffsetRange(0,n),train_ref.OffsetRange(0,m))]
 vocab={};a=np.array([vocab.setdefault(x,len(vocab)) for x in sequence1],dtype=np.int64);b=np.array([vocab.setdefault(x,len(vocab)) for x in sequence2],dtype=np.int64)
 directions=np.empty((n,m),dtype=np.uint8);prev=np.zeros(m);plen=np.zeros(m);pdir=np.zeros(m,dtype=np.uint8)
 for i in range(n):
  diag=np.empty(m);diag[0]=0;diag[1:]=prev[:-1]
  dlen=np.empty(m);dlen[0]=0;dlen[1:]=plen[:-1]
  ddir=np.empty(m,dtype=np.uint8);ddir[0]=0;ddir[1:]=pdir[:-1]
  ext=np.where(a[i]==b,diag+1+np.where(ddir==3,dlen,0),-1)
  current=np.maximum.accumulate(np.maximum(prev,ext))
  dirs=np.where(current==ext,3,np.where(current==prev,1,2)).astype(np.uint8)
  plen=np.where(dirs==3,dlen+1,0);directions[i]=dirs;prev=current;pdir=dirs
 result=[];i,j,last_i,last_j=n-1,m-1,n,m
 while i>=0 and j>=0:
  d=directions[i,j]
  if d==3:
   if i+1!=last_i or j+1!=last_j:result.append(train_ref.SequenceDiff(train_ref.OffsetRange(i+1,last_i),train_ref.OffsetRange(j+1,last_j)))
   last_i,last_j=i,j;i-=1;j-=1
  elif d==1:i-=1
  else:j-=1
 if last_i!=0 or last_j!=0:result.append(train_ref.SequenceDiff(train_ref.OffsetRange(0,last_i),train_ref.OffsetRange(0,last_j)))
 result.reverse();return result

def enable_fast_alignment():
 train_ref.compute=fast_compute;infer_ref.compute=fast_compute

def normalize(s):
 import re
 return re.sub(r'\s+','',s.replace('<s>',''))

def token_merge(a,b,o,directory):
 directory=pathlib.Path(directory);directory.mkdir(parents=True,exist_ok=True)
 for name,tokens in [('a',a),('b',b),('o',o)]:
  (directory/(name+'.tokens')).write_text(''.join(t+'\n' for t in tokens),encoding='utf-8',newline='')
 args=[GIT,'-c','core.autocrlf=false','merge-file','--diff-algorithm=histogram','--diff3','-p',str(directory/'b.tokens'),str(directory/'o.tokens'),str(directory/'a.tokens')]
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 (directory/'merge.stdout').write_bytes(p.stdout);(directory/'merge.stderr').write_bytes(p.stderr)
 (directory/'command.json').write_text(json.dumps({'args':args,'returncode':p.returncode}),encoding='utf-8')
 if p.returncode<0 or p.returncode>127:raise RuntimeError(p.stderr.decode('utf-8',errors='replace'))
 text=p.stdout.decode('utf-8');hunks=[text.splitlines()] if p.returncode==0 else infer_ref.parse_conflict_markers(text)
 return hunks

def serialize_hunks(hunks):return [{'conflict':list(h)} if isinstance(h,tuple) else {'clean':h} for h in hunks]
def deserialize_hunks(hunks):return [tuple(h['conflict']) if 'conflict' in h else h['clean'] for h in hunks]
PATTERNS=[infer_ref.REVERSE_LABEL_MAP[i] for i in range(9)]
