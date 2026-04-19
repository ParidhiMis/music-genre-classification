# main.py

import librosa
import numpy as np
import joblib
from collections import Counter
from tensorflow.keras.models import load_model

# ----------------------------
# Load Saved Model + Scaler
# ----------------------------
model = load_model("ann/ann_model.h5")
scaler = joblib.load("ann/scaler.pkl")

# Genre Names
genres = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# ----------------------------
# Feature Extraction Function
# ----------------------------
def extract_features(y, sr):
    features = []

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma))
    features.append(np.var(chroma))

    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.var(rms))

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spec_cent))
    features.append(np.var(spec_cent))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spec_bw))
    features.append(np.var(spec_bw))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.var(zcr))

    harmony = librosa.effects.harmonic(y)
    features.append(np.mean(harmony))
    features.append(np.var(harmony))

    perceptr = librosa.effects.percussive(y)
    features.append(np.mean(perceptr))
    features.append(np.var(perceptr))

    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    features.append(tempo)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    for i in range(20):
        features.append(np.mean(mfcc[i]))
        features.append(np.var(mfcc[i]))

    return np.array(features)


# ----------------------------
# Prediction Function
# ----------------------------
def predict_song(file_path):
    y, sr = librosa.load(file_path, duration=30)

    chunk_samples = 3 * sr
    predictions = []

    for i in range(10):
        start = i * chunk_samples
        end = start + chunk_samples

        chunk = y[start:end]

        if len(chunk) < chunk_samples:
            continue

        features = extract_features(chunk, sr)
        features = features.reshape(1, -1)

        # Apply same scaler used in training
        features = scaler.transform(features)

        pred = model.predict(features, verbose=0)
        index = np.argmax(pred)

        predictions.append(genres[index])

    final_genre = Counter(predictions).most_common(1)[0][0]

    print("Chunk Predictions:", predictions)
    print("Final Genre:", final_genre)


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    predict_song("test_audio/test_audio2.wav")