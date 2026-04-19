from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import noise_injection, time_shift, spec_mask

labels_dict = {
    "COPD": 0,
    "Pneumonia": 1,
    "Healthy": 2,
    "URTI": 3,
    "Bronchiectasis": 4,
    "Bronchiolitis": 5,
    "LRTI": 6,
    "Asthma": 7
    }

class Respiratory_Sound_Aug(Dataset):
    def __init__(self, audio_path='/data/moon/kiat/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/', aug_type=''):
        self.sr = 16000
        self.aug_type = aug_type
        self.audio_path = audio_path
        self.file_label_df = pd.read_csv('file_label_df.csv')
        self.labels = [self.file_label_df['Diagnosis'][i] for i in range(len(self.file_label_df['filename']))]
        self.filenames = [self.file_label_df['filename'][i] for i in range(len(self.file_label_df['filename']))]

    def preprocessing(self, audio_file):
        # we want to resample audio to 16 kHz
        x, sr = librosa.load(audio_file, sr=self.sr)
        
        max_len = 5 * self.sr
        if x.shape[0] < max_len:
            pad_width = max_len - x.shape[0]
            x = np.pad(x, (0, pad_width))
        elif x.shape[0] > max_len:
            # truncated
            x = x[:max_len]
        return x

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audiofile = self.audio_path + self.filenames[idx] + '.wav'
        label = self.labels[idx]
        x = self.preprocessing(audiofile)[None, ...]

        if self.aug_type == 'noise':
            x = librosa.feature.mfcc(y=x, sr=self.sr)
            x_aug = noise_injection(x)
        elif self.aug_type == 'shift':
            x = librosa.feature.mfcc(y=x, sr=self.sr)
            x_aug = time_shift(x)
        elif self.aug_type == 'mask':
            S = librosa.feature.melspectrogram(y=x, sr=self.sr, n_mels=128, fmax=8000)
            x_aug = spec_mask(librosa.power_to_db(S))
        
        return (x_aug, labels_dict[label])

class Respiratory_Sound(Dataset):
    def __init__(self, audio_path='/data/moon/kiat/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/', mode='train'):
        self.mode = mode
        self.file_label_df = pd.read_csv('file_label_df.csv')
        self.labels = []
        self.preprocessed_data = []
        for i in tqdm(range(len(self.file_label_df['filename']))):
            self.labels.append(self.file_label_df['Diagnosis'][i])
            audio_file = audio_path + self.file_label_df['filename'][i] + '.wav'
            data = self.preprocessing(audio_file, mode = 'mfcc')
            self.preprocessed_data.append(data)
        self.preprocessed_data = np.array(self.preprocessed_data) # (920, 20, 157) (num, H, W)
        self.preprocessed_data = self.preprocessed_data.reshape(-1, 1, 20, 157) # (920, 1, 20, 157) (num, C, H, W)
        self.labels = np.array(self.labels)
        self.labels_categorical = [labels_dict[l] for l in self.labels]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.preprocessed_data, self.labels_categorical, test_size=0.2, random_state=42)
        self.x_train = list(zip(self.x_train, self.y_train))
        self.x_test = list(zip(self.x_test, self.y_test))

    def preprocessing(self, audio_file, mode):
        # we want to resample audio to 16 kHz
        sr_new = 16000 # 16kHz sample rate
        x, sr = librosa.load(audio_file, sr=sr_new)
        # padding sound 
        # because duration of sound is dominantly 20 s and all of sample rate is 22050
        # we want to pad or truncated sound which is below or above 20 s respectively
        max_len = 5 * sr_new  # length of sound array = time x sample rate
        if x.shape[0] < max_len:
            # padding with zero
            pad_width = max_len - x.shape[0]
            x = np.pad(x, (0, pad_width))
        elif x.shape[0] > max_len:
            # truncated
            x = x[:max_len]
        if mode == 'mfcc':
            feature = librosa.feature.mfcc(y=x, sr=sr_new)
        
        elif mode == 'log_mel':
            feature = librosa.feature.melspectrogram(y=x, sr=sr_new, n_mels=128, fmax=8000)
            feature = librosa.power_to_db(feature, ref=np.max)
        return feature

    def __len__(self):
        if self.mode == 'train':
            num_sample = len(self.x_train)
        else:
            num_sample = len(self.x_test)
        return num_sample

    def __getitem__(self, idx):
        if self.mode == 'train':
            x, y = self.x_train[idx]
        else:
            x, y = self.x_test[idx]
        return x, y