import pandas as pd
import matplotlib.pyplot as plt
import librosa
import numpy as np


# Code copied and edited from https://www.kaggle.com/code/davids1992/specaugment-quick-implementation
def spec_augment(original_melspec,
                 freq_masking_max_percentage = 0.15, 
                 time_masking_max_percentage = 0.3):

    augmented_melspec = original_melspec.squeeze().copy()
    all_frames_num, all_freqs_num = augmented_melspec.shape

    # Frequency masking
    freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
    num_freqs_to_mask = int(freq_percentage * all_freqs_num)
    f0 = int(np.random.uniform(low = 0.0, high = (all_freqs_num - num_freqs_to_mask)))
    
    augmented_melspec[:, f0:(f0 + num_freqs_to_mask)] = 0

    # Time masking
    time_percentage = np.random.uniform(0.0, time_masking_max_percentage)
    num_frames_to_mask = int(time_percentage * all_frames_num)
    t0 = int(np.random.uniform(low = 0.0, high = (all_frames_num - num_frames_to_mask)))
    
    augmented_melspec[t0:(t0 + num_frames_to_mask), :] = 0
    
    return augmented_melspec[None, ...]

def noise_injection(x, noise_factor=0.005):
    white_noise = np.random.randn(len(x)) * noise_factor
    x_aug = x + white_noise
    return x_aug
    
def time_shift(x):
    x_aug = np.roll(x, 3000)
    return x_aug

def spec_mask(x):
    aug_spec = spec_augment(x, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3)
    x_aug = librosa.power_to_db(aug_spec)
    return x_aug
