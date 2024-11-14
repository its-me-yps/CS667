import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint

def load_audio(file_path, target_length):
    """
    Loads an audio file and centers it with zero-padding if needed to match the target length.
    """
    audio_data, sample_rate = librosa.load(file_path)
    padded_audio = librosa.util.pad_center(audio_data, size=target_length)
    return padded_audio, sample_rate

def compute_spectrogram(file_path, target_length=220500):
    """
    Computes a spectrogram from an audio file.
    """
    try:
        audio_data, sample_rate = load_audio(file_path, target_length)
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        return spectrogram
    except Exception as e:
        print(f"Error computing spectrogram for {file_path}: {e}")
        return None

def compute_mel_spectrogram(file_path, target_length=220500):
    """
    Computes a mel-spectrogram from an audio file.
    """
    try:
        audio_data, sample_rate = load_audio(file_path, target_length)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=256)
        mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db
    except Exception as e:
        print(f"Error computing mel-spectrogram for {file_path}: {e}")
        return None

def export_spectrograms(num_spectrograms=10):
    """
    Generates and saves spectrograms and mel-spectrograms for inspection.
    """
    dataset = pd.read_csv('data/dataset_2.csv')

    for idx, row in dataset.iterrows():
        if idx >= num_spectrograms:
            break

        audio_path = f"data/audioset_audios/{row['ytid']}_{row['start']}_{row['stop']}_cut.mp3"
        spectrogram = compute_spectrogram(audio_path)
        mel_spectrogram = compute_mel_spectrogram(audio_path)

        if spectrogram is not None:
            save_spectrogram(spectrogram, row['ytid'], row['label'], spectrogram_type="Spectrogram")
        if mel_spectrogram is not None:
            save_spectrogram(mel_spectrogram, row['ytid'], row['label'], spectrogram_type="Mel-Spectrogram")

def save_spectrogram(spectrogram, ytid, label, spectrogram_type="Spectrogram"):
    """
    Saves a spectrogram as an image file for a given type.
    """
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    img = librosa.display.specshow(spectrogram, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(f"{spectrogram_type} for {ytid} ({label})", fontsize=20)
    plt.colorbar(img, ax=ax, format='%0.2f')
    output_path = f"data/spectrograms/{ytid}-{label}_{spectrogram_type.lower().replace(' ', '-')}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {spectrogram_type} for {ytid} ({label}) to {output_path}")

def apply_random_frequency_mask(spectrogram, num_masks=1, max_width=10):
    """
    Applies random frequency masks to a spectrogram, muting frequencies within a random range.
    """
    for _ in range(num_masks):
        mask_width = randint(1, max_width)
        mask_start = randint(0, spectrogram.shape[0] - mask_width)
        spectrogram[mask_start:mask_start + mask_width, :] = 0
    return spectrogram

if __name__ == '__main__':
    export_spectrograms()
