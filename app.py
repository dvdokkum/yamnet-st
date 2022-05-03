import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile

st.title('Sound ID Demo')

# Load the model.
@st.cache
def load_model():
	  return hub.load('https://tfhub.dev/google/yamnet/1')

model = load_model()

process = False

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

#st.dataframe(class_names)

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

def process_audio(wave_data):
    waveform = wav_data / tf.int16.max
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    return waveform, infered_class

uploaded_file = st.sidebar.file_uploader("Upload a .wav file to get started.", type='wav')
st.sidebar.write("or")


if uploaded_file is not None:
    with st.expander("file info"):
        sample_rate, wav_data = wavfile.read(uploaded_file, 'rb')
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
        duration = len(wav_data)/sample_rate
        st.audio(uploaded_file)
        st.write(f'Sample rate: {sample_rate} Hz')
        st.write(f'Total duration: {duration:.2f}s')
        st.write(f'Size of the input: {len(wav_data)}')
    process = st.button("analyze audio")
else: st.write("<<< Please select some audio to process.")

if process == True:
    with st.spinner("processing"):
        waveform = wav_data / tf.int16.max
        scores, embeddings, spectrogram = model(waveform)
        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()
        infered_class = class_names[scores_np.mean(axis=0).argmax()]
        st.write(infered_class)
        
        viz = plt.figure(figsize=(10, 6))
        plt.subplot(3,1,1)
        plt.plot(waveform)
        plt.xlim([0, len(waveform)])

        plt.subplot(3, 1, 2)
        plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')

        mean_scores = np.mean(scores, axis=0)
        top_n = 10
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
        plt.subplot(3, 1, 3)
        plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

        # # patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
        # values from the model documentation
        patch_padding = (0.025 / 2) / 0.01
        plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
        # Label the top_N classes.
        yticks = range(0, top_n, 1)
        plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
        _ = plt.ylim(-0.5 + np.array([top_n, 0]))

        st.pyplot(viz)
        