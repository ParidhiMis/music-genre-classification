import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_audio

# Load model
model = load_model("cnn+transfer_learning/music_genre_transfer_learning.h5")

# Genre labels (CHANGE based on your dataset order)
genres = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

def predict_genre(file_path):
    # Preprocess
    spectrogram = preprocess_audio(file_path)

    # Add batch dimension
    spectrogram = np.expand_dims(spectrogram, axis=0)

    # Predict
    prediction = model.predict(spectrogram)

    predicted_index = np.argmax(prediction)
    predicted_genre = genres[predicted_index]

    return predicted_genre

# Test
if __name__ == "__main__":
    file_path = "test_audio/test_audio1.wav"  # put your audio here
    genre = predict_genre(file_path)
    print("Predicted Genre:", genre)