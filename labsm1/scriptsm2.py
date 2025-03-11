import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.fftpack


def plotAudio(Signal, Fs, TimeMargin=[0, 0.02], fsize=2 ** 10):
    # Time domain plot
    plt.subplot(2, 1, 1)
    start_sample = int(TimeMargin[0] * Fs)
    end_sample = int(TimeMargin[1] * Fs)
    time_axis = np.arange(start_sample, end_sample) / Fs
    plt.plot(time_axis, Signal[start_sample:end_sample])
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Time Domain')
    plt.xlim(TimeMargin[0], TimeMargin[1])
    plt.grid()
    plt.xlim(TimeMargin)

    # Frequency domain plot
    yf = scipy.fftpack.fft(Signal, fsize)
    freq_axis = np.linspace(0, Fs / 2, fsize // 2)
    magnitude = 20 * np.log10(np.abs(yf[:fsize // 2]))

    plt.subplot(2, 1, 2)
    plt.plot(freq_axis, magnitude)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Frequency Domain')
    plt.grid()
    plt.tight_layout()


# Example usage with sin440Hz.wav
if __name__ == "__main__":
    data, fs = sf.read('sin_440Hz.wav')
    plt.figure(figsize=(10, 6))
    plotAudio(data, fs, [0, 0.02])
    plt.show()