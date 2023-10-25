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
import librosa

st.title("Signal Processing Functions")
st.markdown("##### Author: [Yannis Kalfas](https://github.com/kalfasyan)")
st.markdown("***Credits to [Tim Sainburg](https://github.com/timsainb) and [Kyle Kastner](https://github.com/kastnerkyle) for their gists***")

uploaded_file = st.file_uploader("Upload a signal (.wav)...", type=["wav"])
if uploaded_file is None: st.info("Please upload a signal."); st.stop()
st.success("Signal uploaded successfully!")

raw_data = st.audio(uploaded_file, format="audio/wav")
signal_bytes = BytesIO(uploaded_file.read())
fs, data = wavfile.read(signal_bytes)


st.subheader("Signal plot")

# Apply some filters that the user can choose from
filters = st.selectbox("Choose a filter to apply on the signal:", 
                       ("None", "Low-pass filter", "High-pass filter", 
                        "Band-pass filter"))

if filters == "None":
    st.info("No filter applied.")
elif filters == "Low-pass filter":
    lowcut = st.number_input("Low cut", min_value=0, max_value=500, value=50, step=5) # Hz
    data = butter_lowpass_filter(data, lowcut, fs, order=4)
    st.info("Low-pass filter applied.")
elif filters == "High-pass filter":
    highcut = st.number_input("High cut", min_value=0, max_value=5000, value=500, step=5) # Hz
    data = butter_highpass_filter(data, highcut, fs, order=4)
    st.info("High-pass filter applied.")
elif filters == "Band-pass filter":
    lowcut = st.number_input("Low cut", min_value=0, max_value=500, value=50, step=5) # Hz
    highcut = st.number_input("High cut", min_value=0, max_value=5000, value=500, step=5) # Hz
    data = butter_bandpass_filter(data, 50, 250, fs, order=4)
    st.info("Band-pass filter applied.")

# Plotting the signal
st.write("You can interact with the plot by zooming in and out, panning, etc.")
fig = plt.figure()
plt.plot(data)
plt.title(f"Sampling frequency: {fs} Hz, Signal length: {len(data)} samples", fontsize=10)
sns.despine()
# st.pyplot(fig)
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500, width=700)

# Make the spectrogram

st.subheader("Spectrograms")

fft_size = st.number_input("FFT size", min_value=64, max_value=8192, value=512, step=64) # window size for the FFT
noverlap = st.number_input("Noverlap", min_value=0., max_value=float(fft_size), value=fft_size-fft_size/6, step=1.) # overlap between windows
default_step = fft_size/6 # distance to slide along the window (in time)
hop_length = st.number_input("Step size", min_value=1, max_value=fft_size, value=int(default_step), step=1)
win_length = st.number_input("Window length", min_value=0, max_value=100, value=5, step=5)
colormap = st.selectbox("Colormap", 
                        ("viridis", "plasma", "inferno", "magma", "cividis", "Greys", 
                         "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr", 
                         "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", 
                         "YlGnBu", "PuBuGn", "BuGn", "YlGn"))

# Compute the spectrogram using librosa 
fig, ax = plt.subplots()
X = librosa.stft(data.astype(np.float16), n_fft=fft_size, hop_length=hop_length, win_length=win_length)
Xdb = librosa.amplitude_to_db(np.abs(X))
librosa.display.specshow(Xdb, sr=fs, hop_length=hop_length, x_axis='time', y_axis='hz', cmap=colormap)
cbar = plt.colorbar()
cbar.ax.set_ylabel('dB')    

plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500, width=700)

with st.expander("Learn more about the settings", expanded=True):
    st.markdown("**Breakdown of what the Short-Time Fourier Transform (STFT) does:**\
                The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.\
                    The STFT is computed by sliding a window along the signal, multiplying the signal values with a window function, and computing the DFT of the windowed signal.\
                        The window is usually chosen to be a Hann window (also known as Hanning window).\
                            The DFT of the windowed signal is computed using the fast Fourier transform (FFT) algorithm.\
                                    The FFT is computed on each windowed signal, and the result is called a frame.\
                                        The STFT is the collection of frames.\
                                            The STFT is a complex-valued matrix whose rows represent the frequency bins and whose columns represent the time bins.\
                                                The magnitude of the STFT is called the spectrogram.\
                                                    The spectrogram is a real-valued matrix whose rows represent the frequency bins and whose columns represent the time bins.\
                                                        The spectrogram is usually plotted with a colormap that represents the magnitude of the spectrogram with colors.\
                                                                To learn more about the STFT, please visit the [librosa documentation](https://librosa.org/doc/latest/generated/librosa.stft.html).")
    st.markdown("**Sampling frequency**: The number of samples of a signal that are used per second to represent the signal.")
    st.markdown("**FFT size**: The number of data points used in each block for the DFT. A larger number implies a better frequency resolution, but a worse time resolution.\
                The default value is 512. Must be a power of 2. To get a better frequency resolution, you can try to increase the FFT size.\
                    To get a better time resolution, you can try to decrease the FFT size.\
                        If the FFT size is too small compared to the window length, the spectrogram will be mostly empty.\
                            If the FFT size is too large compared to the window length, the spectrogram will be very dense.\
                                To learn more about the FFT size, please visit the [librosa documentation](https://librosa.org/doc/latest/generated/librosa.stft.html).")
    st.markdown("**Noverlap**: The number of points of overlap between blocks. The default value is 0, which means no overlap.\
                Should be smaller than `fft_size`.\
                    If `noverlap` is 0, the STFT matrix is square.\
                        If `noverlap` is smaller than `fft_size`, the STFT matrix is taller than it is wide.\
                            To learn more about the overlap, please visit the [librosa documentation](https://librosa.org/doc/latest/generated/librosa.stft.html).")
    st.markdown("**Step size**: The distance to slide along the window (in time). Default value is 1.\
                If the step size is smaller than the window length, the spectrogram will be very dense.\
                    If the step size is larger than the window length, the spectrogram will be mostly empty.\
                        To learn more about the step size, please visit the [librosa documentation](https://librosa.org/doc/latest/generated/librosa.stft.html).")
    st.markdown("**Window length**: The size of the window function. Default value is 5. \
                The window will be centered on each frame.\
                    The window is applied in the time domain to each frame, and the Fourier transform is computed on each windowed frame.\
                        So the number of rows in the STFT matrix is (1 + `fft_size`/2).\
                            The number of columns depends on the length of the signal and the `step_size` and on the `fft_size`.\
                                With the default values, the spectrogram will have 1 + `fft_size`/2 rows and `len(data)`/`step_size` columns.\
                                    To learn more about the window length, please visit the [librosa documentation](https://librosa.org/doc/latest/generated/librosa.stft.html).")
    st.markdown("**Colormap**: The colormap to use. Default value is viridis.")
    st.markdown("For more information, please visit the [librosa documentation](https://librosa.org/doc/latest/generated/librosa.display.specshow.html).")
        