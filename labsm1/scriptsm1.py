import soundfile as sf
import numpy as np

# Read original file
data, fs = sf.read('sound1.wav', dtype='float32')

# Split channels
left_channel = data[:, 0]
right_channel = data[:, 1]

# Create mono mix
mono_mix = (left_channel + right_channel) / 2

# Save files
sf.write('sound_L.wav', left_channel, fs)
sf.write('sound_R.wav', right_channel, fs)
sf.write('sound_mix.wav', mono_mix, fs)

print("Files created successfully: sound_L.wav, sound_R.wav, sound_mix.wav")