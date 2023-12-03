import numpy as np
import librosa

from torch.utils.data import random_split

def process_audio_data(audio_path):
    # Load audio data
    audio, _ = librosa.load(audio_path, sr=None)
    # Compute the STFT
    stft = librosa.core.stft(audio, n_fft=2048, window='hann', hop_length=1024)
    # Compute the magnitude spectrogram
    stft_abs = np.abs(stft)
    # Convert the magnitude spectrogram to dB scale
    spec_db = librosa.amplitude_to_db(stft_abs, ref=np.max)
    
    return spec_db

def split_dataset(dataset, train_ratio=0.7, valid_ratio=0.15):
    # Get the size of the dataset
    total_size = len(dataset)
    
    # Compute the sizes of the training, validation, and test sets
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size
    
    return random_split(dataset, [train_size, valid_size, test_size])
