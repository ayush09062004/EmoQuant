#!/usr/bin/env python
# coding: utf-8

# In[3]:


import speech_recognition as sr
from nrclex import NRCLex
import pyaudio

def record_audio(duration=600):
    # Initialize recognizer and microphone
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    with microphone as source:
        print("Recording...")
        audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
        print("Recording complete.")
    
    return audio

def transcribe_audio(audio):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcription: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def analyze_sentiment(text):
    lexicon = NRCLex(text)
    emotions = lexicon.affect_frequencies
    print(f"Emotions: {emotions}")
    
    # Determine the predominant emotion
    predominant_emotion = max(emotions, key=emotions.get)
    print(f"Predominant Emotion: {predominant_emotion}")

def main():
    audio = record_audio()
    text = transcribe_audio(audio)
    if text:
        analyze_sentiment(text)

if __name__ == "__main__":
    main()


# In[ ]:




