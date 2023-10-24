"""
Streamlit app for basic signal processing operations applied on a signal that a user uploads.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.io import wavfile
import seaborn as sns
import streamlit.components.v1 as components
import mpld3
from scipy import signal as sg
from utils import *

st.title("Signal Processing Functions")
st.markdown("##### Author: [Yannis Kalfas](https://github.com/kalfasyan)")
st.markdown("***Credits to [Tim Sainburg](https://github.com/timsainb) and [Kyle Kastner](https://github.com/kastnerkyle) for their gists***")

uploaded_file = st.file_uploader("Upload a signal (.wav)...", type=["wav"])
if uploaded_file is None: st.info("Please upload a signal."); st.stop()
st.success("Signal uploaded successfully!")

raw_data = st.audio(uploaded_file, format="audio/wav")
signal_bytes = BytesIO(uploaded_file.read())
fs, data = wavfile.read(signal_bytes)


# Apply some filters that the user can choose from
filters = st.selectbox("Choose a filter to apply on the signal:", 
                       ("None", "Low-pass filter", "High-pass filter", 
                        "Band-pass filter"))

if filters == "None":
    st.info("No filter applied.")
elif filters == "Low-pass filter":
    lowcut = st.slider("Low cut", min_value=0, max_value=500, value=50, step=10) # Hz 
    data = butter_lowpass_filter(data, lowcut, fs, order=5)
    st.info("Low-pass filter applied.")
elif filters == "High-pass filter":
    highcut = st.slider("High cut", min_value=0, max_value=5000, value=500, step=10) # Hz
    data = butter_highpass_filter(data, highcut, fs, order=5)
    st.info("High-pass filter applied.")
elif filters == "Band-pass filter":
    lowcut = st.slider("Low cut", min_value=0, max_value=500, value=50, step=10) # Hz 
    highcut = st.slider("High cut", min_value=0, max_value=5000, value=500, step=10) # Hz
    data = butter_bandpass_filter(data, 50, 250, fs, order=5)
    st.info("Band-pass filter applied.")

# Plotting the signal
st.subheader("Signal plot")
st.write("The signal is plotted below. You can interact with the plot by zooming in and out, panning, etc.")
fig = plt.figure()
plt.plot(data)
plt.title(f"Sampling frequency: {fs} Hz, Signal length: {len(data)} samples", fontsize=10)
sns.despine()
# st.pyplot(fig)
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500, width=700)

# Make the spectrogram

st.subheader("Spectrograms")

fft_size = st.slider("FFT size", min_value=64, max_value=4096, value=512, step=64) 
default_step = fft_size/16 # distance to slide along the window (in time)
step_size = st.slider("Step size", min_value=64, max_value=2048, value=int(default_step), step=8) 
spec_thresh = st.slider("Spectrogram threshold", min_value=0.0, max_value=10.0, value=0.5, step=0.1) # threshold for spectrograms (lower filters out more noise) 

wav_spectrogram = pretty_spectrogram(data.astype('float64'), 
                                     fft_size = fft_size,
                                     step_size=step_size,
                                        log = True,
                                        thresh=spec_thresh)

# Make two columns for the plots
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(4, 4))
    cax = ax.matshow(
        np.transpose(wav_spectrogram),
        interpolation='nearest',
        aspect='auto',
        cmap=plt.cm.afmhot,
        origin='lower')
    fig.colorbar(cax)
    plt.title('Spectrogram', fontsize=10)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=500, width=700)

# Invert the spectrogram
recovered_audio_orig = invert_pretty_spectrogram(wav_spectrogram, fft_size = fft_size,
                                            step_size = step_size, log = True, n_iter = 10)
# Make a new spectrogram of the inverted signal
inverted_spectrogram = pretty_spectrogram(
    recovered_audio_orig.astype('float64'), 
    fft_size = fft_size, 
    step_size = step_size, 
    log = True, 
    thresh = spec_thresh)
with col2:
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(4, 4))
    cax = ax.matshow(
        np.transpose(inverted_spectrogram),
        interpolation='nearest',
        aspect='auto',
        cmap=plt.cm.afmhot,
        origin='lower')
    fig.colorbar(cax)
    plt.title('Inverted Spectrogram', fontsize=10)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=500, width=700)

