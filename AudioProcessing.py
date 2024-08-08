#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pydub import AudioSegment
import speech_recognition as sr
import numpy as np
import librosa

# Helper function to split audio into chunks
def split_audio(file_path, segments, output_folder):
    audio = AudioSegment.from_wav(file_path)
    os.makedirs(output_folder, exist_ok=True)
    
    for i, (start, end) in enumerate(segments):
        start_ms = start * 1000  # Convert to milliseconds
        end_ms = end * 1000      # Convert to milliseconds
        segment = audio[start_ms:end_ms]
        segment.export(os.path.join(output_folder, f'speaker_{i+1}.wav'), format='wav')

# Dummy segmentation function
def get_segments(file_path):
    # Placeholder: Replace with actual segmentation
    # Segments are tuples of (start_time_in_seconds, end_time_in_seconds)
    return [(0, 30), (30, 60)]  # Example: Two segments, each 30 seconds long

# Transcribe audio
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to process audio
def process_audio(file_path, output_folder):
    # Get speaker segments (replace this with actual diarization logic)
    segments = get_segments(file_path)
    
    # Split audio into segments
    split_audio(file_path, segments, output_folder)
    
    # Transcribe each segment
    transcriptions = []
    for i in range(len(segments)):
        segment_file = os.path.join(output_folder, f'speaker_{i+1}.wav')
        transcription = transcribe_audio(segment_file)
        transcriptions.append(f"Speaker {i+1} said: {transcription}")
    
    return transcriptions

# Paths
file_path = 'path_to_your_file' #info removed to hide personal information
output_folder = 'path_to_your_file' #info removed to hide personal information

# Process the audio and get transcriptions
transcriptions = process_audio(file_path, output_folder)

# Print transcriptions
for transcription in transcriptions:
    print(transcription)

