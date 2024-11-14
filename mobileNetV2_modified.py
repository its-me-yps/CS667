import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from audio_preprocessing import compute_spectrogram

# --- Configurations ---
LOG_DIR = "logs"
MODEL_DIR = "models"
DATA_PATH = 'data/dataset_2.csv'
AUDIO_DIR = 'data/audioset_audios'
BATCH_SIZE = 8
EPOCHS = 10
CLASS_NAMES = ['Happy', 'Aggressive', 'Sad']

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)
log_file = f"{LOG_DIR}/{os.path.basename(__file__)}-{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.txt"
sys.stdout = open(log_file, 'w')

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Using GPU for training.")
else:
    print("No GPU detected; training on CPU.")

# Load and preprocess dataset
df = pd.read_csv(DATA_PATH)
df = df[df.label.isin(CLASS_NAMES)]

# Generate spectrograms for each audio file
df['spectrogram'] = df.apply(lambda row: compute_spectrogram(f"{AUDIO_DIR}/{row['ytid']}_{int(row['start'])}_{int(row['stop'])}_cut.mp3"), axis=1)
missing_files = df['spectrogram'].isna().sum()
print(f"{missing_files} entries dropped due to missing audio files.")

# Drop rows with missing spectrograms
df.dropna(subset=['spectrogram'], inplace=True)
print(f"Data distribution:\n{df['label'].value_counts()}\n")

# Label encoding and data preparation
label_encoder = LabelEncoder()
df['emotion_label_encoded'] = label_encoder.fit_transform(df['label'])
X = np.array(df['spectrogram'].tolist())
y = to_categorical(df['emotion_label_encoded'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Define MobileNetV2 model
model = Sequential([
    Conv2D(3, (3, 3), input_shape=X_train.shape[1:]),  # Convert grayscale to RGB
    MobileNetV2(include_top=False, pooling='avg', weights='imagenet'),
    Dense(512, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

# Freeze all but the last few layers of MobileNetV2
model.layers[1].trainable = True
for layer in model.layers[1].layers[:-4]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# --- Custom Callback ---
class StopTrainingOnHighAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.99:
            self.model.stop_training = True

# --- Callbacks and Class Weights ---
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.0001)
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weights = {i: weight for i, weight in enumerate(class_weights)}

# --- Train Model ---
history = model.fit(
    X_train, y_train, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    validation_data=(X_test, y_test), 
    class_weight=class_weights,
    callbacks=[StopTrainingOnHighAccuracy(), lr_reduction]
)

# --- Predictions and Evaluations ---
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_true_labels, y_pred_labels), annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:\n", classification_report(y_true_labels, y_pred_labels))
print(f"Cohen's Kappa: {cohen_kappa_score(y_true_labels, y_pred_labels)}")

# Evaluation metrics
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# --- Training History Plots ---
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
metrics = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
for i, metric in enumerate(metrics):
    ax[i//2, i%2].plot(history.history[metric])
    ax[i//2, i%2].set_title(f'Model {metric}')
    ax[i//2, i%2].set_xlabel('Epochs')
    ax[i//2, i%2].set_ylabel(metric)
plt.tight_layout()
plt.show()

# --- Save Model ---
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(f"{MODEL_DIR}/mobileNetV2_modified.keras")
print(f"Model saved to {MODEL_DIR}/mobileNetV2_modified.keras")

# --- Close Log File ---
sys.stdout.close()
