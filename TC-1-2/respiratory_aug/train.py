import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torchvision.models as models
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

LABELS = ["COPD", "Pneumonia", "Healthy", "URTI", "Bronchiectasis", "Bronchiolitis", "LRTI", "Asthma"]
LABELS_DICT = {l: i for i, l in enumerate(LABELS)}
AUDIO_PATH = '/data/moon/kiat/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/'
SR = 16000
DURATION = 5
N_MFCC = 20
EPOCHS = 50
BATCH_SIZE = 32
torch.manual_seed(42)

def extract_features(audio_file):
    x, _ = librosa.load(audio_file, sr=SR)
    max_len = DURATION * SR
    if x.shape[0] < max_len:
        x = np.pad(x, (0, max_len - x.shape[0]))
    else:
        x = x[:max_len]
    mfcc = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, delta, delta2], axis=0)  # (60, T)


class RespiratoryDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_train_dataset(audio_path=AUDIO_PATH):
    cache_data = 'features_cache.npy'
    cache_labels = 'labels_cache.npy'

    if os.path.exists(cache_data) and os.path.exists(cache_labels):
        print('Loading from cache...')
        all_data = np.load(cache_data)
        all_labels = np.load(cache_labels)
    else:
        file_label_df = pd.read_csv('file_label_df.csv')
        all_data, all_labels = [], []
        for i in tqdm(range(len(file_label_df)), desc='Loading audio'):
            audio_file = audio_path + file_label_df['filename'][i] + '.wav'
            all_data.append(extract_features(audio_file))
            all_labels.append(LABELS_DICT[file_label_df['Diagnosis'][i]])
        all_data = np.array(all_data)[:, np.newaxis, :, :]  # (N, 1, 60, T)
        all_labels = np.array(all_labels)
        np.save(cache_data, all_data)
        np.save(cache_labels, all_labels)
        print('Cache saved.')

    x_train, _, y_train, _ = train_test_split(
        all_data, all_labels, test_size=0.2, random_state=42
    )

    # Use standard balanced class weights (softened) for robust minority class prediction
    class_weights = compute_class_weight('balanced', classes=np.arange(len(LABELS)), y=y_train)
    class_weights = np.sqrt(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Return pure training data without augmentation
    return RespiratoryDataset(x_train, y_train), class_weights


class RespiratoryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        # Modify the first convolutional layer to accept 1-channel MFCC input instead of 3-channel RGB
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final fully connected layer to output exactly 8 respiratory classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 8)

    def forward(self, x):
        return self.resnet(x)


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    train_ds, class_weights = load_train_dataset()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = RespiratoryClassifier().to(device)
    
    # AdamW with specific learning rate for pretrained fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Standard CrossEntropy with smoothed balanced weights works best without augmentation
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    print(f'\nTraining for {EPOCHS} epochs...')
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += (torch.argmax(out, 1) == y).float().mean().item()
        
        scheduler.step()
        epoch_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch:3d} | loss: {epoch_loss:.4f} | acc: {total_acc/len(train_loader):.4f}')

    torch.save(model.state_dict(), 'respiratory_bn_delta.pth')
    print('Model saved → respiratory_bn_delta.pth')


if __name__ == '__main__':
    main()
