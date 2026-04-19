import librosa
import numpy as np

def preprocess_audio(file_path, img_size=(128, 128)):
    # Load audio
    y, sr = librosa.load(file_path, duration=30)

    # Convert to Mel Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to log scale (important!)
    log_spectrogram = librosa.power_to_db(spectrogram)

    # Normalize
    log_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / (np.max(log_spectrogram) - np.min(log_spectrogram))

    # Resize to match model input
    log_spectrogram = np.resize(log_spectrogram, img_size)

    # Add channel dimension (for CNN)
    log_spectrogram = np.stack((log_spectrogram,)*3, axis=-1)

    return log_spectrogram