import numpy as np
import soundfile as sf
import pygame

def alaw_compress(x, A=87.6):
    x = np.clip(x, -1, 1)
    y = np.zeros_like(x)
    abs_x = np.abs(x)
    mask = abs_x < (1 / A)
    y[mask] = np.sign(x[mask]) * (A * abs_x[mask]) / (1 + np.log(A))
    y[~mask] = np.sign(x[~mask]) * (1 + np.log(A * abs_x[~mask])) / (1 + np.log(A))
    return y

def alaw_decompress(y, A=87.6):
    y = np.clip(y, -1, 1)
    x = np.zeros_like(y)
    abs_y = np.abs(y)
    mask = abs_y < (1 / (1 + np.log(A)))
    x[mask] = np.sign(y[mask]) * (abs_y[mask] * (1 + np.log(A))) / A
    x[~mask] = np.sign(y[~mask]) * (np.exp(abs_y[~mask] * (1 + np.log(A)) - 1)) / A
    return x

def ulaw_compress(x, mu=255):
    x = np.clip(x, -1, 1)
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

def ulaw_decompress(y, mu=255):
    y = np.clip(y, -1, 1)
    return np.sign(y) * (1 / mu) * (np.power(1 + mu, np.abs(y)) - 1)

def quantize(x, bits, range_min=-1, range_max=1):
    levels = 2 ** bits
    x = np.clip(x, range_min, range_max)
    x_norm = (x - range_min) / (range_max - range_min)
    q = np.round(x_norm * (levels - 1))
    return q.astype(np.int8), range_min, range_max

def dequantize(q, bits, range_min, range_max):
    levels = 2 ** bits
    x_norm = q / (levels - 1)
    return x_norm * (range_max - range_min) + range_min

def dpcm_compress_no_pred(x, bits):
    y = np.zeros(x.shape, dtype=np.int8)
    e = 0
    for i in range(x.shape[0]):
        d = x[i] - e
        q, range_min, range_max = quantize(np.array([d]), bits, range_min=-0.5, range_max=0.5)
        y[i] = q[0]
        d_quantized = dequantize(q, bits, range_min, range_max)[0]
        e = e + d_quantized
        e = np.clip(e, -1, 1)
    return y

def dpcm_decompress_no_pred(y, bits):
    x = np.zeros_like(y, dtype=float)
    for i in range(len(y)):
        if i == 0:
            x[i] = dequantize(np.array([y[i]]), bits, -0.5, 0.5)[0]
        else:
            d_quantized = dequantize(np.array([y[i]]), bits, -0.5, 0.5)[0]
            x[i] = x[i - 1] + d_quantized
        x[i] = np.clip(x[i], -1, 1)
    return x

def no_pred(X):
    return X[-1]

def dpcm_compress_with_pred(x, bits, predictor=no_pred, n=3):
    y = np.zeros(x.shape, dtype=np.int8)
    xp = np.zeros(x.shape, dtype=float)
    e = 0
    for i in range(x.shape[0]):
        d = x[i] - e
        q, range_min, range_max = quantize(np.array([d]), bits, range_min=-0.5, range_max=0.5)
        y[i] = q[0]
        d_quantized = dequantize(q, bits, range_min, range_max)[0]
        xp[i] = d_quantized + e
        idx = np.arange(max(0, i - n + 1), i + 1, 1, dtype=int)
        e = predictor(xp[idx]) if i > 0 else 0
        xp[i] = np.clip(xp[i], -1, 1)
    return y

def dpcm_decompress_with_pred(y, bits, predictor=no_pred, n=3):
    x = np.zeros_like(y, dtype=float)
    xp = np.zeros_like(y, dtype=float)
    for i in range(len(y)):
        if i == 0:
            x[i] = dequantize(np.array([y[i]]), bits, -0.5, 0.5)[0]
            xp[i] = x[i]
        else:
            idx = np.arange(max(0, i - n), i, 1, dtype=int)
            e = predictor(xp[idx])
            d_quantized = dequantize(np.array([y[i]]), bits, -0.5, 0.5)[0]
            x[i] = d_quantized + e
            xp[i] = x[i]
        x[i] = np.clip(x[i], -1, 1)
    return x

def mean_predictor(X):
    return np.mean(X)

def test_dpcm():
    print("Testing DPCM with example from instructions:")
    X = np.array([15, 16, 20, 14, 5, 10, 15, 13, 11, 7, 10, 11, 20, 1, 23], dtype=float)
    X_min, X_max = X.min(), X.max()
    X_normalized = 2 * (X - X_min) / (X_max - X_min) - 1
    y_dpcm = dpcm_compress_no_pred(X_normalized, 2)
    X_dpcm_restored = dpcm_decompress_no_pred(y_dpcm, 2)
    X_dpcm_restored = (X_dpcm_restored + 1) / 2 * (X_max - X_min) + X_min
    print("DPCM without prediction (restored signal):", X_dpcm_restored)
    y_dpcm_pred = dpcm_compress_with_pred(X_normalized, 2, predictor=mean_predictor, n=3)
    X_dpcm_pred_restored = dpcm_decompress_with_pred(y_dpcm_pred, 2, predictor=mean_predictor, n=3)
    X_dpcm_pred_restored = (X_dpcm_pred_restored + 1) / 2 * (X_max - X_min) + X_min
    print("DPCM with prediction (restored signal):", X_dpcm_pred_restored)

def play_audio(filename):
    pygame.mixer.init()
    try:
        sound = pygame.mixer.Sound(filename)
        print(f"Playing: {filename}")
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
    except pygame.error as e:
        print(f"Error playing {filename}: {e}")
    finally:
        pygame.mixer.quit()

def process_audio(filename, bits_list=[8, 7, 6, 5, 4, 3, 2]):
    print(f"\nProcessing file: {filename}")
    data, samplerate = sf.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data / np.max(np.abs(data))
    quality_table = []
    for bits in bits_list:
        print(f"  Bit depth: {bits}")
        y_alaw = alaw_compress(data)
        q_alaw, _, _ = quantize(y_alaw, bits)
        y_alaw_deq = dequantize(q_alaw, bits, -1, 1)
        data_alaw = alaw_decompress(y_alaw_deq)
        alaw_filename = f"{filename}_alaw_{bits}bits.wav"
        sf.write(alaw_filename, data_alaw, samplerate)

        y_ulaw = ulaw_compress(data)
        q_ulaw, _, _ = quantize(y_ulaw, bits)
        y_ulaw_deq = dequantize(q_ulaw, bits, -1, 1)
        data_ulaw = ulaw_decompress(y_ulaw_deq)
        ulaw_filename = f"{filename}_ulaw_{bits}bits.wav"
        sf.write(ulaw_filename, data_ulaw, samplerate)

        y_dpcm = dpcm_compress_no_pred(data, bits)
        data_dpcm = dpcm_decompress_no_pred(y_dpcm, bits)
        dpcm_nopred_filename = f"{filename}_dpcm_nopred_{bits}bits.wav"
        sf.write(dpcm_nopred_filename, data_dpcm, samplerate)

        y_dpcm_pred = dpcm_compress_with_pred(data, bits, predictor=mean_predictor, n=3)
        data_dpcm_pred = dpcm_decompress_with_pred(y_dpcm_pred, bits, predictor=mean_predictor, n=3)
        dpcm_pred_filename = f"{filename}_dpcm_pred_{bits}bits.wav"
        sf.write(dpcm_pred_filename, data_dpcm_pred, samplerate)

        quality_table.append({
            "bits": bits,
            "A-law": alaw_filename,
            "μ-law": ulaw_filename,
            "DPCM_no_pred": dpcm_nopred_filename,
            "DPCM_pred": dpcm_pred_filename
        })
    return quality_table

if __name__ == "__main__":
    test_dpcm()
    audio_files = {
        "low": "low.wav",
        "medium": "medium.wav",
        "high": "high.wav"
    }
    quality_results = {}
    for quality, filename in audio_files.items():
        try:
            quality_results[quality] = process_audio(filename)
            print(f"\nPlaying audio files for {quality}.wav:")
            for row in quality_results[quality]:
                for method in ["A-law", "μ-law", "DPCM_no_pred", "DPCM_pred"]:
                    audio_file = row[method]
                    play_audio(audio_file)
                    input(f"Finished playing {audio_file}. Press Enter to continue...")
        except FileNotFoundError:
            print(f"File {filename} not found. Please add low.wav, medium.wav, high.wav to the directory.")
    print("\nSound Quality Table:")
    for quality, results in quality_results.items():
        print(f"\nFile: {quality}.wav")
        print("| Bit Depth | A-law | μ-law | DPCM No Prediction | DPCM With Prediction |")
        print("|-----------|-------|-------|--------------------|----------------------|")
        for row in results:
            print(f"| {row['bits']} | {row['A-law']} | {row['μ-law']} | {row['DPCM_no_pred']} | {row['DPCM_pred']} |")
    print("\nListen to the generated files and note your observations on sound quality for each file and bit depth.")