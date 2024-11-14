import tkinter as tk
from tkinter import filedialog, messagebox
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from keras.models import load_model
from audio_preprocessing import generate_spectrogram
import time

# Load model - replace with the model path that outputs [happy, aggressive, sad] probabilities
MODEL_PATH = 'models/mobileNetV2_modified.keras'
model_mood = load_model(MODEL_PATH)

# Global variables to store the results
timestamps = []
mood_predictions = []

# Function to process audio and predict mood
def process_audio_live(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        duration = len(audio)
        
        # Clear previous data
        timestamps.clear()
        mood_predictions.clear()

        # Initialize the plot
        fig, ax = plt.subplots()
        ax.set_title("Mood Prediction Over Time")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Mood Probability")
        
        # Plot lines for each mood
        line_happy, = ax.plot([], [], label="Happy", color="green")
        line_aggressive, = ax.plot([], [], label="Aggressive", color="red")
        line_sad, = ax.plot([], [], label="Sad", color="blue")
        ax.legend()
        
        def update_plot(frame):
            if frame < len(mood_predictions):
                # Update data
                times = timestamps[:frame + 1]
                happy_vals = [pred[0] for pred in mood_predictions[:frame + 1]]
                aggressive_vals = [pred[1] for pred in mood_predictions[:frame + 1]]
                sad_vals = [pred[2] for pred in mood_predictions[:frame + 1]]
                
                line_happy.set_data(times, happy_vals)
                line_aggressive.set_data(times, aggressive_vals)
                line_sad.set_data(times, sad_vals)
                ax.relim()
                ax.autoscale_view()

            return line_happy, line_aggressive, line_sad

        # Process 10-second clips and make predictions
        for start in range(0, duration, 10000):  # 10 seconds in milliseconds
            end = min(start + 10000, duration)
            segment = audio[start:end]
            
            # Export segment as temporary file
            segment_path = "temp_segment.mp3"
            segment.export(segment_path, format="mp3")
            
            # Generate spectrogram and predict
            spectrogram = generate_spectrogram(segment_path)
            y = np.array([spectrogram])
            prediction = model_mood.predict(y)[0]  # Output should be [happy, aggressive, sad]
            
            # Append predictions and timestamp
            mood_predictions.append(prediction)
            timestamps.append(start / 1000)  # Convert ms to seconds

        # Animate the plot
        ani = FuncAnimation(fig, update_plot, frames=len(timestamps), interval=500, repeat=False)
        plt.show()

        # Calculate mood percentages
        calculate_mood_percentage()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to calculate and display mood percentages
def calculate_mood_percentage():
    mood_totals = np.sum(mood_predictions, axis=0)
    mood_percentages = (mood_totals / np.sum(mood_totals)) * 100
    mood_names = ["Happy", "Aggressive", "Sad"]
    percentages_str = "\n".join(f"{name}: {percent:.6f}%" for name, percent in zip(mood_names, mood_percentages))
    
    messagebox.showinfo("Mood Analysis", f"Mood Percentages:\n{percentages_str}")

# Function to open file dialog
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
    if file_path:
        process_audio_live(file_path)

# Create GUI window with animation
def create_initial_ui():
    # Setup the main window
    root = tk.Tk()
    root.title("Audio Mood Predictor")

    # Initial animation canvas
    canvas = tk.Canvas(root, width=400, height=300)
    canvas.pack()

    # Draw an animated circle to represent a "pulse" animation
    circle = canvas.create_oval(150, 100, 250, 200, outline='blue', width=2)
    def animate_circle():
        for size in range(5, 30, 5):
            canvas.coords(circle, 150-size, 100-size, 250+size, 200+size)
            root.update()
            time.sleep(0.05)
        for size in range(30, 5, -5):
            canvas.coords(circle, 150-size, 100-size, 250+size, 200+size)
            root.update()
            time.sleep(0.05)
        root.after(1000, animate_circle)  # Loop the animation

    animate_circle()

    # Title label
    title_label = tk.Label(root, text="Barking Emotion Predictor", font=("Helvetica", 16))
    title_label.pack(pady=10)

    # Description label
    desc_label = tk.Label(root, text="Choose an audio file to analyze its mood over time.", font=("Helvetica", 12))
    desc_label.pack(pady=10)

    # Browse file button
    browse_button = tk.Button(root, text="Choose Audio File", command=open_file)
    browse_button.pack(pady=20)

    root.mainloop()

# Run the GUI with initial UI animation
create_initial_ui()

