import os
import nltk
import streamlit as st
import pandas as pd
import assemblyai as aai
import sounddevice as sd
import numpy as np
import wave
from scipy.io.wavfile import write
from nltk.sentiment import SentimentIntensityAnalyzer
from gtts import gTTS
import tempfile

# Download VADER if not available
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Set AssemblyAI API Key
aai.settings.api_key = "-----"

def preprocess_data(file_path):
    """Loads and cleans the dataset."""
    df = pd.read_csv(file_path, delimiter=",", quotechar='"', engine="python")
    if df.shape[1] == 1:  # If incorrectly formatted, split manually
        df = df.iloc[:, 0].str.split(",", expand=True)
        df.columns = ["Text", "Sentiment", "Source", "Date/Time", "User ID", "Location", "Confidence Score"]
    df.dropna(inplace=True)  # Remove missing values
    return df

def analyze_text_sentiment(text):
    """Analyzes sentiment of text input."""
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def transcribe_audio_assemblyai(audio_path):
    """Converts speech to text using AssemblyAI."""
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path)
    return transcript.text

def analyze_audio_sentiment(audio_path):
    """Analyzes sentiment of transcribed audio text."""
    text = transcribe_audio_assemblyai(audio_path)
    return analyze_text_sentiment(text)

def text_to_speech(text):
    """Converts sentiment result to speech using gTTS."""
    tts = gTTS(text=text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def record_audio(filename, duration=5, fs=44100):
    """Records audio from the microphone and saves it to a file."""
    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype=np.int16)
    sd.wait()
    write(filename, fs, recording)
    st.write("Recording finished.")

# Streamlit App
st.title("Voice & Text Sentiment Analysis")

option = st.selectbox("Choose Input Type:", ["Text", "Audio Upload", "Live Audio", "Dataset"])

if option == "Text":
    user_input = st.text_area("Enter customer feedback:")
    if st.button("Analyze Sentiment"):
        result = analyze_text_sentiment(user_input)
        st.write(f"Sentiment: {result}")
        
        # Convert to speech
        tts_file = text_to_speech(f"The sentiment is {result}")
        st.audio(tts_file, format='audio/mp3')
        st.download_button("Download Audio", open(tts_file, "rb"), file_name="sentiment.mp3")

elif option == "Audio Upload":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file:
        temp_audio = "temp_audio.wav"
        with open(temp_audio, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Analyze Sentiment"):
            sentiment = analyze_audio_sentiment(temp_audio)
            st.write(f"Sentiment: {sentiment}")
            
            # Convert to speech
            tts_file = text_to_speech(f"The sentiment is {sentiment}")
            st.audio(tts_file, format='audio/mp3')
            st.download_button("Download Audio", open(tts_file, "rb"), file_name="sentiment.mp3")

elif option == "Live Audio":
    if st.button("Record and Analyze Sentiment"):
        temp_audio = "live_audio.wav"
        record_audio(temp_audio)
        sentiment = analyze_audio_sentiment(temp_audio)
        st.write(f"Sentiment: {sentiment}")
        
        # Convert to speech
        tts_file = text_to_speech(f"The sentiment is {sentiment}")
        st.audio(tts_file, format='audio/mp3')
        st.download_button("Download Audio", open(tts_file, "rb"), file_name="sentiment.mp3")

elif option == "Dataset":
    uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_csv:
        df_clean = preprocess_data(uploaded_csv)
        df_clean['Predicted Sentiment'] = df_clean['Text'].apply(analyze_text_sentiment)
        st.write(df_clean)
        
        # Generate TTS for results
        summary_text = "Summary of Sentiment Analysis: " + ", ".join(df_clean['Predicted Sentiment'].unique())
        tts_file = text_to_speech(summary_text)
        st.audio(tts_file, format='audio/mp3')
        st.download_button("Download Audio", open(tts_file, "rb"), file_name="dataset_sentiment.mp3")
