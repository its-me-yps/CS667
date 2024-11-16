# Smart Collar for Pet Dogs: Audio-Based Emotional Analysis

This project focuses on the development of a **smart collar** for pet dogs, integrating **Internet of Things (IoT)** and **Machine Learning (ML)** technologies to monitor pet health effectively. Initially, the collar will perform **audio-based emotional analysis**, with plans to expand into physiological monitoring as suitable datasets become available.

---

## Key Features

### **Audio Emotion Classification**
- **Vocalization Monitoring**: The collar records audio samples of the dog's vocalizations every 5–10 minutes.
- **Emotion Detection**: The audio recordings are processed using an ML model trained to classify the dog's emotional state (e.g., *happy*, *angry*, *sad*, etc.).
- **Daily Summary**: The results are aggregated to provide a daily emotional pattern overview.

---

## How to Run the Project

### 1.Download and Process Audio Data
Next, download the audio files and preprocess them. This will cut them into smaller parts and prepare the data for analysis.
```bash
python audioset_download.py

### 2.Preprocess the Audio Data and Visualize Spectrograms
After downloading the data, run the preprocessing script to visualize some spectrograms:
```bash
python audio_preprocessing.py

### 3. Train the Model
Now, run the appropriate model training script. For example, to train using MobileNetV2, run:
```bash
python mobileNetV2.py

### 4. Test the Trained Model
```bash
python model_tester.py

### 5. Run the Simulation
```bash
python simulation.py
