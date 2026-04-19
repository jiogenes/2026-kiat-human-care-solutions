import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from train import RespiratoryClassifier, extract_features, LABELS, AUDIO_PATH

LABELS_DICT = {l: i for i, l in enumerate(LABELS)}

# noise_factor=0.005
def noise_injection(x, noise_factor=0.8):
    return x + np.random.randn(*x.shape) * noise_factor


def time_shift(x, shift=20):
    return np.roll(x, shift, axis=-1)


class RespiratoryAugDataset(Dataset):
    def __init__(self, aug_type, audio_path=AUDIO_PATH):
        assert aug_type in ('noise', 'shift')
        self.aug_type = aug_type
        self.audio_path = audio_path
        file_label_df = pd.read_csv('file_label_df.csv')
        filenames = file_label_df['filename'].tolist()
        labels = [LABELS_DICT[d] for d in file_label_df['Diagnosis']]
        _, test_filenames, _, test_labels = train_test_split(
            filenames, labels, test_size=0.2, random_state=42
        )
        self.filenames = test_filenames
        self.labels = test_labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_file = self.audio_path + self.filenames[idx] + '.wav'
        feat = extract_features(audio_file)[np.newaxis, :, :]  # (1, 60, T)

        if self.aug_type == 'noise':
            feat = noise_injection(feat)
        elif self.aug_type == 'shift':
            feat = time_shift(feat)

        return torch.tensor(feat, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def compute_accuracy(all_labels, all_preds):
    return sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)


def compute_f1_score(all_labels, all_preds):
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate(model, loader, device, compute_acc=True, compute_f1=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    results = {}
    if compute_acc:
        results['accuracy'] = compute_accuracy(all_labels, all_preds)
    if compute_f1:
        results['f1'] = compute_f1_score(all_labels, all_preds)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', nargs='+', choices=['accuracy', 'f1'],
                        default=['accuracy', 'f1'])
    parser.add_argument('--json', action='store_true',
                        help='결과를 JSON 형식으로 출력')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = RespiratoryClassifier().to(device)
    model.load_state_dict(torch.load('respiratory_bn_delta.pth', map_location=device, weights_only=True))
    print('Loaded respiratory_bn_delta.pth\n')

    noise_ds = RespiratoryAugDataset(aug_type='noise')
    shift_ds = RespiratoryAugDataset(aug_type='shift')
    combined_ds = ConcatDataset([noise_ds, shift_ds])
    combined_loader = DataLoader(combined_ds, batch_size=1, pin_memory=True)

    print(f'Evaluation samples: {len(combined_ds)}')
    results = evaluate(model, tqdm(combined_loader), device,
                       compute_acc='accuracy' in args.metrics,
                       compute_f1='f1' in args.metrics)
    if args.json:
        print(json.dumps(results))
    else:
        if 'accuracy' in results:
            print(f'\nAccuracy : {results["accuracy"]:.4f}')
        if 'f1' in results:
            print(f'F1 Score : {results["f1"]:.4f}')


if __name__ == '__main__':
    main()
