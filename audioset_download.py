import os
import yt_dlp as ytdlp
import subprocess
import pandas as pd
from pydub import AudioSegment

csv_file_path = './data/dataset_2.csv'
df = pd.read_csv(csv_file_path)
df['start'] = df['start'].astype(int)
df['stop'] = df['stop'].astype(int)
output_path = 'data/audioset_audios'
failed_downloads = []

def download_audio(row, output_path):
    video_id = row['ytid']
    if os.path.isfile(f"{output_path}/{video_id}.mp3"):
        print(f"Video {video_id} already downloaded, skipping.")
        return True
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f"{output_path}/{video_id}.%(ext)s",
            'quiet': False,
        }
        print(f"Downloading {video_id}")
        with ytdlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        downloaded_file_path = next(
            (f"{output_path}/{video_id}.{ext}" for ext in ['webm', 'm4a', 'opus', 'ogg']
             if os.path.isfile(f"{output_path}/{video_id}.{ext}")), None
        )
        if not downloaded_file_path:
            raise FileNotFoundError(f"Downloaded audio file not found for {video_id}.")
        converted_file_path = f"{output_path}/{video_id}.mp3"
        print(f"Converting {video_id} to MP3")
        subprocess.run([
            'ffmpeg', '-i', downloaded_file_path, '-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k', converted_file_path
        ], check=True)
        os.remove(downloaded_file_path)
    except Exception as e:
        print(f"Error downloading or processing video {video_id}: {e}")
        return False
    return True

def cut_audio(row, output_path, full_audio_path):
    video_id = row['ytid']
    start_time = int(row['start'])
    end_time = int(row['stop'])
    export_path = f"{output_path}/{video_id}_{start_time}_{end_time}_cut.mp3"
    if os.path.isfile(export_path):
        print(f"Video {export_path} already cut, skipping.")
        return None
    try:
        audio_file_path = f"{full_audio_path}/{video_id}.mp3"
        audio = AudioSegment.from_file(audio_file_path)
        cut_audio = audio[start_time * 1000:end_time * 1000]
        cut_audio.export(export_path, format="mp3")
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")

def process_audio(row, output_path):
    if download_audio(row, f"{output_path}/full_video"):
        cut_audio(row, output_path, f"{output_path}/full_video")
    else:
        print(row.name)
        failed_downloads.append(row.name)

if __name__ == "__main__":
    df.apply(lambda row: process_audio(row, output_path), axis=1)
    if failed_downloads:
        print(failed_downloads)
        print(f"Removing {len(failed_downloads)} failed downloads from CSV.")
        df.drop(failed_downloads, inplace=True)
        df.to_csv(csv_file_path, index=False)
        print("Updated CSV file saved.")
    else:
        print("All downloads completed successfully.")
