import keras
import numpy as np

from audio_preprocessing import generate_spectrogram

MODEL_PATH = 'models/mobileNetV2_modified.keras'
AUDIO_FILE_PATH = 'data/temp_cut.mp3'

# Load model
model_playfulness = keras.models.load_model(MODEL_PATH)

# Audio preprocessing
spectrogram = generate_spectrogram(AUDIO_FILE_PATH)

# Predict
y = np.array([spectrogram])
prediction = model_playfulness.predict(y)[0]  # Retrieve the first prediction

# Format output with floating point (not scientific notation) and highlight max
emotions = ['Happy', 'Aggressive', 'Sad']
max_index = np.argmax(prediction)
output = "\n".join(
    f"{emotions[i]}: {'{:.6f}'.format(prediction[i])}{' <-- highest' if i == max_index else ''}"
    for i in range(len(prediction))
)

print(output)
