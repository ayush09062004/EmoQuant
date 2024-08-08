#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install textblob')
from textblob import TextBlob

# Function to perform sentiment analysis
def sentiment_analysis(file_path):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Create a TextBlob object
    blob = TextBlob(text)
    
    # Perform sentiment analysis
    sentiment = blob.sentiment
    
    # Print the results
    print(f"Sentiment Analysis of the Text:\n")
    print(f"Polarity (range -1 to 1): {sentiment.polarity}")
    print(f"Subjectivity (range 0 to 1): {sentiment.subjectivity}")

# Provide the path to your text file
file_path = 'path_to_your_file' #info hide to protect personal information

# Perform sentiment analysis
sentiment_analysis(file_path)


# In[4]:


import nltk
from nrclex import NRCLex

# Download necessary NLTK data
nltk.download('punkt')

# Function to perform detailed emotion analysis
def detailed_emotion_analysis(file_path):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Perform emotion analysis using NRCLex
    emotion_analysis = NRCLex(text)
    
    # Get emotion scores
    emotions = emotion_analysis.raw_emotion_scores
    
    # Print the results
    print(f"Detailed Emotion Analysis of the Text:\n")
    for emotion, score in emotions.items():
        print(f"{emotion.capitalize()}: {score}")

# Provide the path to your text file
file_path = 'path_to_your_file' #info hide to protect personal information

# Perform detailed emotion analysis
detailed_emotion_analysis(file_path)


# In[1]:


import nltk
from nrclex import NRCLex

# Download necessary NLTK data
nltk.download('punkt')

# Function to perform detailed emotion analysis
def detailed_emotion_analysis(file_path):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Perform emotion analysis using NRCLex
    emotion_analysis = NRCLex(text)
    
    # Get emotion scores
    emotions = emotion_analysis.raw_emotion_scores
    
    # Print the results
    print(f"Detailed Emotion Analysis of the Text:\n")
    for emotion, score in emotions.items():
        print(f"{emotion.capitalize()}: {score}")

# Provide the path to your text file
file_path = 'path_to_your_file' #info hide to protect personal information

# Perform detailed emotion analysis
detailed_emotion_analysis(file_path)


# In[2]:


import nltk
from nrclex import NRCLex

# Download necessary NLTK data
nltk.download('punkt')

# Function to perform detailed emotion analysis
def detailed_emotion_analysis(file_path):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Perform emotion analysis using NRCLex
    emotion_analysis = NRCLex(text)
    
    # Get emotion scores
    emotions = emotion_analysis.raw_emotion_scores
    
    # Print the results
    print(f"Detailed Emotion Analysis of the Text:\n")
    for emotion, score in emotions.items():
        print(f"{emotion.capitalize()}: {score}")

# Provide the path to your text file
file_path = 'path_to_your_file' #info hide to protect personal information
# Perform detailed emotion analysis
detailed_emotion_analysis(file_path)


# In[3]:


from nrclex import NRCLex

# Function to perform detailed emotion analysis on patient's statements
def detailed_emotion_analysis(patient_text):
    # Perform emotion analysis using NRCLex
    emotion_analysis = NRCLex(patient_text)
    
    # Get emotion scores
    emotions = emotion_analysis.raw_emotion_scores
    
    # Print the results
    print(f"Detailed Emotion Analysis of the Patient's Statements:\n")
    for emotion, score in emotions.items():
        print(f"{emotion.capitalize()}: {score}")

# Provide the conversation text
conversation = """
Therapist: How have you been feeling since our last session?
Patient: Honestly, I’ve been feeling really overwhelmed. It’s like there’s this heavy cloud hanging over me, and I can’t see any light at the end of the tunnel. Everything feels so pointless.
Therapist: I’m sorry to hear that. Can you tell me more about what’s been weighing on you?
Patient: I just feel like I’m stuck in this endless loop of negativity. I wake up every day with this dread, and it feels like I’m just going through the motions. Nothing seems to matter anymore—work, friends, even things I used to enjoy.
Therapist: It sounds like you’re feeling really disconnected and hopeless right now. Have you been able to talk to anyone else about these feelings?
Patient: No, not really. I’ve tried, but I don’t want to burden anyone with my problems. It’s like no one would understand even if I did try to explain. They’d just tell me to cheer up or that things will get better, but it doesn’t feel that way.
Therapist: I can hear that you’re feeling isolated and like others might not fully grasp what you’re going through. It’s okay to feel this way, and I’m here to listen and help however I can. What would you say has been the hardest part for you lately?
Patient: The hardest part is this constant emptiness. I’ve lost interest in everything. Even the simplest tasks feel like a mountain to climb. I can’t remember the last time I felt genuinely happy. It’s like I’m just existing, not living.
Therapist: That emptiness you’re describing must be incredibly tough to deal with. It’s important to acknowledge how hard you’re working just to get through each day. Have there been any moments, no matter how small, where you felt even a slight relief or comfort?
Patient: Sometimes, when I’m with my dog, I feel a little better. It’s like he understands, even if he can’t say anything. But those moments don’t last long. As soon as I’m alone, everything comes rushing back, and I’m drowning in it all over again.
Therapist: It’s good that you have your dog to provide some comfort, even if just for a little while. It shows that there’s still some connection to the things around you. Have you thought about what it is about those moments with your dog that help you feel a bit better?
Patient: I guess it’s just that he doesn’t expect anything from me. I don’t have to pretend with him. He’s just there, and that’s enough. But when he’s not around, it’s like I’m left alone with all these dark thoughts, and they’re so overwhelming.
Therapist: That sense of unconditional acceptance from your dog seems to be really meaningful for you. It’s understandable that being alone with your thoughts can feel overwhelming. What are those thoughts usually like when they come?
Patient: It’s a lot of self-blame and hopelessness. I keep thinking about all the things I’ve failed at, how I’m not good enough, and how things will never change. It’s exhausting, and I don’t see any way out of this cycle.
Therapist: Those thoughts sound very harsh and painful, and it’s clear that they’re taking a heavy toll on you. Challenging these thoughts can be difficult, but it’s an important part of finding a way out of this dark place. What would you say to yourself if you could look at your situation from a different perspective?
Patient: I don’t know. It’s hard to see things any other way. But maybe… maybe I’d tell myself that it’s okay to feel this way, that it doesn’t make me weak. I guess I’d try to remind myself that it’s not always going to be this dark, even if it feels like it right now.
Therapist: That’s a powerful insight. It shows that despite the darkness you’re feeling, there’s still a part of you that wants to hold on to hope. Recognizing that it’s okay to feel this way and that it won’t always be like this is a big step. How can we build on that glimmer of hope together?
Patient: I don’t know. I just wish I could see some progress, some sign that things might get better. I’m tired of feeling like I’m stuck in this endless cycle.
Therapist: I understand that desire for progress, and it’s something we can work on together. Small steps can lead to significant changes, even if they’re hard to see at first. Let’s explore ways to break down this cycle and create moments of relief and hope, just like the ones you feel with your dog. How does that sound?
Patient: It sounds like a start. I’m not sure how to do it, but I’m willing to try. I just don’t want to feel like this forever.
Therapist: That willingness to try is a great start. We’ll take it one step at a time, and I’ll be here to support you through this. You don’t have to go through it alone.
"""

# Extract only patient's statements for analysis
patient_statements = []
for line in conversation.split('\n'):
    if line.strip().startswith("Patient:"):
        patient_statements.append(line.replace("Patient:", "").strip())

# Join all patient statements into one text
patient_text = ' '.join(patient_statements)

# Perform detailed emotion analysis on the patient's text
detailed_emotion_analysis(patient_text)


# In[5]:


from nrclex import NRCLex

# Function to perform detailed emotion analysis on patient's statements
def detailed_emotion_analysis(file_path):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Extract only patient's statements
    patient_statements = []
    for line in lines:
        if line.strip().startswith("Patient:"):
            patient_statements.append(line.replace("Patient:", "").strip())

    # Join all patient statements into one text
    patient_text = ' '.join(patient_statements)
    
    # Perform emotion analysis using NRCLex
    emotion_analysis = NRCLex(patient_text)
    
    # Get emotion scores
    emotions = emotion_analysis.raw_emotion_scores
    
    # Print the results
    print(f"Detailed Emotion Analysis of the Patient's Statements:\n")
    for emotion, score in emotions.items():
        print(f"{emotion.capitalize()}: {score}")

# Provide the path to your text file
file_path = 'path_to_your_file' #info hide to protect personal information

# Perform detailed emotion analysis
detailed_emotion_analysis(file_path)


# In[ ]:




