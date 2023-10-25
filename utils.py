from scipy.signal import butter, lfilter
import librosa

# Most of the Spectrograms and Inversion are taken from: https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(lowcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut/nyq
    b, a = butter(order, low, btype='low')
    return b, a

def butter_lowpass_filter(data, lowcut, fs, order=4):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(highcut, fs, order=4):
    nyq = 0.5 * fs
    high = highcut/nyq
    b, a = butter(order, high, btype='high')
    return b, a

def butter_highpass_filter(data, highcut, fs, order=4):
    b, a = butter_highpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
