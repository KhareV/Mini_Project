"""Evaluate frozen ECG MODEL_V1 on MIT-BIH; never trains or tunes on it."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.mitbih import MITBIHDataset
from models.ecg_cnn import ECGCNN1D
from preprocessing.ecg import ECGPreprocessor
from evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw/mitbih')
    parser.add_argument('--checkpoint', default='models/MODEL_V1.pt')
    parser.add_argument('--output', default='reports/mitbih_external_report.json')
    args = parser.parse_args()
    ds, model, preprocessor = MITBIHDataset(args.data_dir), None, ECGPreprocessor()
    model, _ = ECGCNN1D.load_checkpoint(args.checkpoint, device='cpu')
    labels, probabilities, failures = [], [], []
    for record in ds.load_all_records():
        if not record.is_valid: continue
        try:
            signal, fs, beats, beat_labels = ds.load_signal_with_annotations(record.record_id)
            prep = preprocessor.process(signal, int(fs), record.record_id, 'mitbih')
            for start in range(0, len(prep.signal) - 2500 + 1, 1250):
                # map annotations to the canonical 250 Hz timebase
                positions = (beats * (250.0 / fs)).astype(int)
                usable = beat_labels[(positions >= start) & (positions < start + 2500)]
                usable = [label for label in usable if label != 'EXCLUDE']
                if not usable: continue
                y = int(sum(label == 'ABNORMAL' for label in usable) > sum(label == 'NORMAL' for label in usable))
                x = torch.tensor(prep.signal[start:start+2500], dtype=torch.float32).view(1, 1, -1)
                probabilities.append(float(model.predict_proba(x)[0, 1]))
                labels.append(y)
        except Exception as exc:
            failures.append({'record_id': record.record_id, 'error': str(exc)})
    y, p = np.asarray(labels), np.asarray(probabilities)
    result = compute_metrics(y, p >= .5, p, split='external', model_name='CENTRAL_ECG_MODEL_V1', dataset='mitbih', notes='External-only: no MIT-BIH training or tuning.').to_dict()
    payload = {'metrics': result, 'n_records': len(ds.load_all_records()), 'n_windows': len(y), 'failures': failures[:25], 'threshold': .5, 'checkpoint': args.checkpoint}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

if __name__ == '__main__': main()
