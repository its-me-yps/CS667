import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from audio_preprocessing import compute_spectrogram

# --- Configurations ---
LOG_DIR = "logs"
MODEL_DIR = "models"
DATA_PATH = 'data\\dataset_2.csv'
AUDIO_DIR = 'data\\audioset_audios'
BATCH_SIZE = 8
EPOCHS = 10
CLASS_NAMES = ['Happy', 'Aggressive', 'Sad']

# Setup log file
os.makedirs(LOG_DIR, exist_ok=True)
log_file = f"{LOG_DIR}\\{os.path.basename(__file__)}-{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.txt"
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
df = df[df.label.isin(CLASS_NAMES)]
df['spectrogram'] = df.apply(lambda row: compute_spectrogram(f"{AUDIO_DIR}\\{row['ytid']}_{int(row['start'])}_{int(row['stop'])}_cut.mp3"), axis=1)
print(f"{df['spectrogram'].isna().sum()} entries dropped because no audio file found.")
df = df.dropna(subset=['spectrogram'])

# Encode emotion labels and split data
label_encoder = LabelEncoder()
df['emotion_label_encoded'] = label_encoder.fit_transform(df['label'])
data = np.array(df['spectrogram'].tolist())
labels = to_categorical(df['emotion_label_encoded'])
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# --- Define MobileNetV2 Model ---
def build_mobilenet_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(3, (3, 3), input_shape=input_shape),  # Convert grayscale to RGB
        MobileNetV2(include_top=False, pooling='avg', weights='imagenet'),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.layers[1].trainable = True
    for layer in model.layers[1].layers[:-4]:  # Freeze all but the last few layers
        layer.trainable = False
    return model

mobilenet_model = build_mobilenet_model(X_train.shape[1:], labels.shape[1])
mobilenet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mobilenet_model.summary()

# --- Custom Callbacks ---
class StopTrainingOnHighAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.99:
            self.model.stop_training = True

lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.0001)

# --- Class Weights for Imbalance ---
class_weights_array = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

# --- Train Model ---
history = mobilenet_model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[StopTrainingOnHighAccuracy(), lr_reduction]
)

# --- Evaluation ---
y_pred = mobilenet_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report and Metrics
class_report = classification_report(y_true_labels, y_pred_labels)
cohens_kappa = cohen_kappa_score(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

print("Classification Report:")
print(class_report)
print(f"Cohen's Kappa: {cohens_kappa}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Plot Training History
fig, ax = plt.subplots(2, 3, figsize=(20, 6))
ax = ax.ravel()
for i, metric in enumerate(['accuracy', 'loss', 'val_accuracy', 'val_loss']):
    ax[i].plot(history.history[metric])
    ax[i].set_title(f'Model {metric}')
    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel(metric)
    ax[i].legend(['train'])
plt.show()

# --- Save Model ---
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, 'mobileNetV2.keras')
mobilenet_model.save(model_path)
print(f"Model saved to {model_path}")

# Close log file
sys.stdout.close()

