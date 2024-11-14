import os
from datetime import datetime
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import Callback
from tqdm.keras import TqdmCallback
from audio_preprocessing import compute_spectrogram  

# --- Configurations ---
LOG_DIR = "logs"
MODEL_DIR = "models"
DATA_PATH = 'data/dataset_2.csv'
AUDIO_DIR = 'data/audioset_audios'
BATCH_SIZE = 16
EPOCHS = 10
CLASS_NAMES = ['Happy', 'Aggressive', 'Sad']

# Setup log file
os.makedirs(LOG_DIR, exist_ok=True)
log_file = f"{LOG_DIR}/{os.path.basename(__file__)}-{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.txt"
sys.stdout = open(log_file, 'w')

# Setup GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Using GPU for training.")
else:
    print("No GPU detected, training will proceed on CPU.")

# --- Load and Process Dataset ---
df = pd.read_csv(DATA_PATH)
df = df[df['label'].isin(CLASS_NAMES)]

# Apply spectrogram generation to each file
df['spectrogram'] = df.apply(lambda row: compute_spectrogram(f"{AUDIO_DIR}/{row['ytid']}_{int(row['start'])}_{int(row['stop'])}_cut.mp3"), axis=1)
print(f"Dropped {df['spectrogram'].isna().sum()} entries due to missing audio files.")
df = df.dropna(subset=['spectrogram'])

# Encode labels and split data
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
X = np.array(df['spectrogram'].tolist())
y = to_categorical(df['label_encoded'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# --- Define ResNet50 Model ---
model = Sequential([
    Conv2D(3, (3, 3), input_shape=X_train.shape[1:]),  # Convert grayscale to RGB
    ResNet50(include_top=False, pooling='avg', weights='imagenet'),
    Dense(512, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])
model.layers[1].trainable = False  # Freeze pre-trained layers

# Compile model with F1 score metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Custom Callback ---
class StopTrainingOnHighAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.99:
            self.model.stop_training = True

# --- Train Model ---
history = model.fit(
    X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[TqdmCallback(verbose=1), StopTrainingOnHighAccuracy()]
)

# --- Evaluation ---
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report and Metrics
print("Classification Report:\n", classification_report(y_true, y_pred))
print(f"Cohen's Kappa Score: {cohen_kappa_score(y_true, y_pred)}")

# --- Plot Training History ---
fig, ax = plt.subplots(2, 3, figsize=(18, 8))
metrics = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
for idx, metric in enumerate(metrics):
    ax[idx // 2, idx % 2].plot(history.history[metric])
    ax[idx // 2, idx % 2].set_title(f'{metric.capitalize()}')
    ax[idx // 2, idx % 2].set_xlabel('Epochs')
    ax[idx // 2, idx % 2].set_ylabel(metric.capitalize())
plt.show()

# --- Save Model ---
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, 'resNet50.keras')
model.save(model_path)
print(f"Model saved to {model_path}")

# --- Close Log File ---
sys.stdout.close()

