import argparse
import json
import os
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

DATA_DIR    = "./"
WEIGHT      = "rnn_4.pt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ───────────────────────────────────────────────────────────────────

class VentilatorDataset(Dataset):
    def __init__(self, df):
        if "pressure" not in df.columns:
            df["pressure"] = 0

        self.df = df.groupby("breath_id").agg(list).reset_index()
        self._prepare_data()

    def __len__(self):
        return self.df.shape[0]

    def _prepare_data(self):
        self.pressures = np.array(self.df["pressure"].values.tolist())

        rs    = np.array(self.df["R"].values.tolist())
        cs    = np.array(self.df["C"].values.tolist())
        u_ins = np.array(self.df["u_in"].values.tolist())
        self.u_outs = np.array(self.df["u_out"].values.tolist())

        self.inputs = np.concatenate([
            rs[:, None],
            cs[:, None],
            u_ins[:, None],
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None],
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p":     torch.tensor(self.pressures[idx], dtype=torch.float),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class RNNModel(nn.Module):
    def __init__(self, input_dim=5, lstm_dim=512, dense_dim=512, logit_dim=512, num_classes=1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_dim * 2),
            nn.ReLU(),
            nn.Linear(dense_dim * 2, dense_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, num_classes),
        )

    def forward(self, x):
        features = self.mlp(x)
        features, _ = self.lstm(features)
        return self.logits(features)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_accuracy(tp, tn, total):
    return (tp + tn) / total if total > 0 else 0


def compute_f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


# ── Entry point ───────────────────────────────────────────────────────────────

NOISE_LEVEL = 0.1
THRESHOLD   = 100.0

FOLD = 4
K    = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', nargs='+', choices=['accuracy', 'f1'],
                        default=['accuracy', 'f1'],
                        help='출력할 지표 선택 (기본값: accuracy f1)')
    parser.add_argument('--json', action='store_true',
                        help='결과를 JSON 형식으로 출력')
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(DATA_DIR, "pressure_data.csv"))

    gkf      = GroupKFold(n_splits=K)
    splits   = list(gkf.split(X=df, y=df, groups=df["breath_id"]))
    _, val_idx = splits[FOLD]
    df_val   = df.iloc[val_idx].copy().reset_index(drop=True)

    dataset  = VentilatorDataset(df_val)

    print(f"Device          : {DEVICE}")
    print(f"Val size (fold {FOLD}): {len(dataset)} breaths")
    print(f"Loading {WEIGHT} ...")

    model = RNNModel().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT, map_location=DEVICE))
    model.eval()

    tp, fp, fn, tn = 0, 0, 0, 0

    for i in tqdm(range(len(dataset))):
        original_input    = dataset[i]["input"]
        original_pressure = dataset[i]["p"]

        noisy_input = original_input.clone()
        noisy_input += NOISE_LEVEL * torch.randn_like(noisy_input)

        with torch.no_grad():
            orig_pred  = model(original_input[None].to(DEVICE)).cpu().squeeze()
            noisy_pred = model(noisy_input[None].to(DEVICE)).cpu().squeeze()

        orig_diff  = torch.abs(orig_pred  - original_pressure).sum().item()
        noisy_diff = torch.abs(noisy_pred - original_pressure).sum().item()

        gt_pos    = orig_diff  <= THRESHOLD
        pred_pos  = noisy_diff <= THRESHOLD

        if gt_pos and pred_pos:
            tp += 1
        elif not gt_pos and pred_pos:
            fp += 1
        elif gt_pos and not pred_pos:
            fn += 1
        else:
            tn += 1

    results = {}
    if 'f1' in args.metrics:
        results['f1'] = compute_f1_score(tp, fp, fn)
    if 'accuracy' in args.metrics:
        results['accuracy'] = compute_accuracy(tp, tn, len(dataset))

    if args.json:
        print(json.dumps(results))
    else:
        if 'f1' in results:
            print(f"F1 Score : {results['f1']:.4f}")
        if 'accuracy' in results:
            print(f"Accuracy : {results['accuracy']:.4f}")
