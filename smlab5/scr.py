import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os


def get_companding_constants():
    return 87.6, 255


def a_law_compress(signal):
    A, _ = get_companding_constants()
    signal = np.clip(signal, -1, 1)
    abs_signal = np.abs(signal)
    compressed = np.where(abs_signal < 1 / A,
                          (A * abs_signal) / (1 + np.log(A)),
                          (1 + np.log(A * abs_signal)) / (1 + np.log(A)))
    return np.sign(signal) * compressed



def a_law_expand(compressed_signal):
    A, _ = get_companding_constants()
    abs_y = np.abs(compressed_signal)
    expanded = np.where(abs_y < 1 / (1 + np.log(A)),
                        abs_y * (1 + np.log(A)) / A,
                        np.exp(abs_y * (1 + np.log(A)) - 1) / A)
    return np.sign(compressed_signal) * expanded


def mu_law_compress(signal):
    _, mu = get_companding_constants()
    signal = np.clip(signal, -1, 1)
    return np.sign(signal) * np.log1p(mu * np.abs(signal)) / np.log1p(mu)


def mu_law_expand(compressed_signal):
    _, mu = get_companding_constants()
    return np.sign(compressed_signal) * (1 / mu) * np.expm1(np.abs(compressed_signal) * np.log1p(mu))


def quantize(data, bit):
    if not 2 <= bit <= 32:
        raise ValueError("Bit depth must be between 2 and 32")

    if np.issubdtype(data.dtype, np.floating):
        m = -1
        n = 1
    elif np.issubdtype(data.dtype, np.integer):
        m = np.iinfo(data.dtype).min
        n = np.iinfo(data.dtype).max
    else:
        raise TypeError("Data type must be integer or float")

    d = 2**bit - 1
    
    data_normalized = (data.astype(float) - m) / (n - m)  
    data_quantized_float = np.round(data_normalized * d) / d
    data_quantized = (data_quantized_float * (n - m) + m).astype(data.dtype)

    return data_quantized


def DPCM_compress_no_pred(x, bit):
    y = np.zeros(x.shape)
    e = 0
    for i in range(0, x.shape[0]):
        y[i] = quantize(x[i] - e, bit)
        e += y[i]
    return y


def DPCM_compress_pred(x, bit, predictor, n):
    y = np.zeros(x.shape)
    xp = np.zeros(x.shape)
    e = 0
    for i in range(0, x.shape[0]):
        y[i] = quantize(x[i] - e, bit)
        xp[i] = y[i] + e
        idx = np.arange(i - n, i, 1, dtype=int) + 1
        idx = idx[idx >= 0]
        e = predictor(xp[idx])
    return y


def no_pred(X):
    return X[-1] if len(X) > 0 else 0


def DPCM_decompress_no_pred(y, bit):
    x = np.zeros(y.shape)
    e = 0
    for i in range(0, y.shape[0]):
        x[i] = y[i] + e
        e += y[i]
    return x


def DPCM_decompress_pred(y, bit, predictor, n):
    x = np.zeros(y.shape)
    xp = np.zeros(y.shape)
    e = 0
    for i in range(0, y.shape[0]):
        x[i] = y[i] + e
        xp[i] = x[i]
        idx = np.arange(i - n, i, 1, dtype=int) + 1
        idx = idx[idx >= 0]
        e = predictor(xp[idx])
    return x


def display_compression(signal, bits, title_suffix=""):
    title = f"Analiza kompresji sygnału ({bits} bitów) {title_suffix}"

    alaw_compressed = a_law_compress(signal)
    alaw_quantized = quantize(alaw_compressed, bits)

    mulaw_compressed = mu_law_compress(signal)
    mulaw_quantized = quantize(mulaw_compressed, bits)

    dpcm_no_pred_q = DPCM_compress_no_pred(signal, bits)
    dpcm_pred_q = DPCM_compress_pred(signal, bits, no_pred, n=1)

    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(title)

    axes[0].plot(signal, label="Oryginalny sygnał")
    axes[0].set_title("Oryginalny sygnał")

    axes[1].plot(alaw_quantized, label="Kompresja A-law")
    axes[1].set_title("Kompresja A-law")

    axes[2].plot(mulaw_quantized, label="Kompresja μ-law")
    axes[2].set_title("Kompresja μ-law")

    axes[3].plot(dpcm_no_pred_q, label="DPCM bez predykcji")
    axes[3].set_title("Kompresja DPCM bez predykcji")

    axes[4].plot(dpcm_pred_q, label="DPCM z predykcją")
    axes[4].set_title("Kompresja DPCM z predykcją")

    for ax in axes:
        ax.grid(True)
        ax.set_xlim(0, len(signal))
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"compression_{bits}_bits_{title_suffix}.png")
    plt.close()


def display_decompression(signal, bits, title_suffix=""):
    title = f"Analiza dekompresji sygnału ({bits} bitów) {title_suffix}"

    alaw_decoded = a_law_expand(quantize(a_law_compress(signal), bits))
    mulaw_decoded = mu_law_expand(quantize(mu_law_compress(signal), bits))
    dpcm_no_pred_dec = DPCM_decompress_no_pred(DPCM_compress_no_pred(signal, bits), bits)
    dpcm_pred_dec = DPCM_decompress_pred(DPCM_compress_pred(signal, bits, no_pred, n=1), bits, no_pred, n=1)

    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(title)

    axes[0].plot(signal, label="Oryginalny sygnał")
    axes[0].set_title("Oryginalny sygnał")

    axes[1].plot(alaw_decoded, label=f"A-law Dekompresja ({bits} bitów)")
    axes[1].set_title("Dekompresja A-law")

    axes[2].plot(mulaw_decoded, label=f"μ-law Dekompresja ({bits} bitów)")
    axes[2].set_title("Dekompresja μ-law")

    axes[3].plot(dpcm_no_pred_dec, label="DPCM bez predykcji")
    axes[3].set_title("Dekompresja DPCM bez predykcji")

    axes[4].plot(dpcm_pred_dec, label="DPCM z predykcją")
    axes[4].set_title("Dekompresja DPCM z predykcją")

    for ax in axes:
        ax.grid(True)
        ax.set_xlim(0, len(signal))
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"decompression_{bits}_bits_{title_suffix}.png")
    plt.close()


def process_audio(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не найден")
    data, samplerate = sf.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]  # Mono only
    data = data / np.max(np.abs(data))  # Normalization

    short_data = data[:1000]  # Use only the first 1000 samples for clarity

    for bits in range(2, 9):
        print(f"\nProcessing '{os.path.basename(filepath)}' with Depth {bits} bit:")
        display_compression(short_data, bits, title_suffix=f"[{os.path.basename(filepath)}]")
        display_decompression(short_data, bits, title_suffix=f"[{os.path.basename(filepath)}]")


if __name__ == "__main__":
    process_audio("sing_high2.wav")
    process_audio("sing_medium2.wav")
    process_audio("sing_low2.wav")
 