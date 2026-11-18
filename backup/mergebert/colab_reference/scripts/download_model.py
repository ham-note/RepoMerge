"""Download the public pretrained encoder; verify the five public upstream files."""
import argparse,pathlib,os,json,hashlib

def main():
 package=pathlib.Path(__file__).resolve().parents[2]
 run=pathlib.Path(os.environ.get('MERGEBERT_RUN_ROOT',str(package/'runtime'))).resolve()
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,default=pathlib.Path(os.environ.get('MERGEBERT_MODEL_PATH',str(run/'models/CodeBERTa-small-v1'))));a=p.parse_args()
 manifest=json.loads((package/'resources/pretrained_encoder_manifest.json').read_text())
 os.environ.pop('HF_HUB_OFFLINE',None)
 from huggingface_hub import snapshot_download
 snapshot_download('huggingface/CodeBERTa-small-v1',local_dir=a.output,allow_patterns=list(manifest))
 for name,expected in manifest.items():
  path=a.output/name
  assert path.stat().st_size==expected['bytes'],name
  with path.open('rb') as f:
   if name=='pytorch_model.bin':digest=hashlib.file_digest(f,'sha256').hexdigest()
   else:
    h=hashlib.sha1();h.update(('blob '+str(path.stat().st_size)+'\0').encode());h.update(f.read());digest=h.hexdigest()
  assert digest==expected['hash'],'Upstream content differs from recorded model: '+name
 print('Verified all five pretrained files')
if __name__=='__main__':main()
