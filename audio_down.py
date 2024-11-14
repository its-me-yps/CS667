import os
import yt_dlp as ytdlp
import subprocess
from pydub import AudioSegment

def download_audio(video_id, output_path):
    full_audio_path = f"{output_path}/temp_full.mp3"
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f"{output_path}/temp_full.%(ext)s",
            'quiet': False,
        }
        print(f"Downloading {video_id}")
        with ytdlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])

        downloaded_file_path = next(
            (f"{output_path}/temp_full.{ext}" for ext in ['webm', 'm4a', 'opus', 'ogg']
             if os.path.isfile(f"{output_path}/temp_full.{ext}")), None
        )
        if not downloaded_file_path:
            raise FileNotFoundError(f"Audio file not found for {video_id}.")

        print(f"Converting {video_id} to MP3")
        subprocess.run([
            'ffmpeg', '-i', downloaded_file_path, '-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k', full_audio_path
        ], check=True)
        os.remove(downloaded_file_path)
    except Exception as e:
        print(f"Error with video {video_id}: {e}")
        return None
    return full_audio_path

def cut_audio(video_id, start_time, end_time, output_path):
    full_audio_path = f"{output_path}/temp_full.mp3"
    export_path = f"{output_path}/temp_cut.mp3"
    if os.path.isfile(export_path):
        print(f"Cut audio {export_path} already exists, skipping.")
        return export_path
    try:
        audio = AudioSegment.from_file(full_audio_path)
        cut_audio = audio[start_time * 1000:end_time * 1000]
        cut_audio.export(export_path, format="mp3")
        print(f"Exported cut audio to {export_path}")
    except Exception as e:
        print(f"Error processing cut audio for video {video_id}: {e}")
        return None
    return export_path

if __name__ == "__main__":
    video_id = input("Enter YouTube video ID: ")
    start_time = int(input("Enter start time in seconds: "))
    end_time = int(input("Enter end time in seconds: "))
    output_path = 'data'
    full_audio_path = download_audio(video_id, output_path)
    if full_audio_path:
        cut_audio(video_id, start_time, end_time, output_path)
    else:
        print("Audio download failed.")
