"""Feature extraction from audio chunks."""

import numpy as np
import librosa
import config


def extract_features(audio: np.ndarray) -> np.ndarray:
    """Extract MFCC features from an audio chunk.
    
    Returns a 1D feature vector summarizing the audio.
    """
    mfccs = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=config.SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
    
    # Summarize temporal dimension: mean + std per coefficient
    features = np.concatenate([
        np.mean(mfccs, axis=1),
        np.std(mfccs, axis=1),
    ])
    
    return features
