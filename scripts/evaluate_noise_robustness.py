"""Label-preserving ECG robustness evaluation at fixed SNR levels."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from scipy.io import loadmat
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.mitbih import MITBIHDataset
from models.ecg_cnn import ECGCNN1D
from preprocessing.ecg import ECGPreprocessor
from evaluation.metrics import compute_metrics

def main():
 p=argparse.ArgumentParser(); p.add_argument('--max-windows',type=int,default=2000); p.add_argument('--checkpoint',default='models/MODEL_V1.pt'); p.add_argument('--output',default='reports/noise_robustness_report.json'); a=p.parse_args()
 ds=MITBIHDataset('data/raw/mitbih'); model,_=ECGCNN1D.load_checkpoint(a.checkpoint,'cpu'); prep=ECGPreprocessor(); windows=[]
 for rec in ds.load_all_records():
  if not rec.is_valid: continue
  sig,fs,beats,labels=ds.load_signal_with_annotations(rec.record_id); out=prep.process(sig,int(fs),rec.record_id,'mitbih').signal; pos=(beats*250/fs).astype(int)
  for start in range(0,len(out)-2500+1,1250):
   usable=[x for x in labels[(pos>=start)&(pos<start+2500)] if x!='EXCLUDE']
   if usable: windows.append((out[start:start+2500],int(sum(x=='ABNORMAL' for x in usable)>sum(x=='NORMAL' for x in usable))))
   if len(windows)>=a.max_windows: break
  if len(windows)>=a.max_windows: break
 x=np.stack([w[0] for w in windows]); y=np.array([w[1] for w in windows]); rng=np.random.RandomState(42); report={'dataset':'MIT-BIH labelled windows + supplied NST noise','n_windows':len(y),'snr_db':{}}
 noise_files=list(Path('data/raw/mitbih_noise_stress').glob('*e*.mat'))
 for snr in [24,18,12,6,0,-6]:
  candidates=[f for f in noise_files if f'e{snr:02d}.mat' in f.name or (snr==-6 and 'e-6.mat' in f.name)]
  source=loadmat(candidates[0],squeeze_me=True,struct_as_record=False)['data'].noisy_ecg[:,0].astype(np.float32)
  noisy=[]
  for clean in x:
   offset=rng.randint(0,len(source)-2500); noise=source[offset:offset+2500]; noise=noise/(np.std(noise)+1e-8)
   target=np.std(clean)/(10**(snr/20)); noisy.append(clean+noise*target)
  tensor=torch.tensor(np.asarray(noisy),dtype=torch.float32).unsqueeze(1); probs=[]
  for batch in torch.split(tensor,128): probs.extend(model.predict_proba(batch)[:,1].tolist())
  probs=np.array(probs); report['snr_db'][str(snr)]=compute_metrics(y,probs>=.5,probs,split='robustness',model_name='CENTRAL_ECG_MODEL_V1',dataset='MITBIH_NST').to_dict()
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
