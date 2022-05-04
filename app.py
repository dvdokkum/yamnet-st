import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
from pathlib import Path
import time
import pydub

TMP_DIR = Path('temp')
if not TMP_DIR.exists():
    TMP_DIR.mkdir(exist_ok=True, parents=True)

MEDIA_STREAM_CONSTRAINTS = {
    "video": False,
    "audio": {
        "echoCancellation": False,  # don't turn on else it would reduce wav quality
        "noiseSuppression": False,
        "autoGainControl": True,
    },
}

st.title('Sound ID Demo')

# Load the model.
@st.cache
def load_model():
	  return hub.load('https://tfhub.dev/google/yamnet/1')

model = load_model()

process = False
file_to_process = ""

# Find the name of the class with the top score when mean-aggregated across frames.
@st.cache
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

#print 520 possible sound identifiers
#st.dataframe(class_names)

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

def process_audio(wav_data):
    waveform = wav_data / tf.int16.max
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    return waveform, infered_class

#recoder component
def save_frames_from_audio_receiver(wavpath):
    with st.sidebar:
        webrtc_ctx = webrtc_streamer(
            key="sendonly-audio",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
        )

        if "audio_buffer" not in st.session_state:
            st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

        status_indicator = st.empty()
        while True:
            if webrtc_ctx.audio_receiver:
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                except queue.Empty:
                    status_indicator.info("No frame arrived.")
                    continue

                for i, audio_frame in enumerate(audio_frames):
                    sound = pydub.AudioSegment(
                        data=audio_frame.to_ndarray().tobytes(),
                        sample_width=audio_frame.format.bytes,
                        frame_rate=audio_frame.sample_rate,
                        channels=len(audio_frame.layout.channels),
                    )
                    # st.markdown(f'{len(audio_frame.layout.channels)}, {audio_frame.format.bytes}, {audio_frame.sample_rate}')
                    # 2, 2, 48000
                    st.session_state["audio_buffer"] += sound
            else:
                break

        audio_buffer = st.session_state["audio_buffer"]

        if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
            audio_buffer.export(wavpath, format="wav")
            st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

def sanitize_audio(wavpath):
    sound = pydub.AudioSegment.from_wav(wavpath)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    sound.export(wavpath, format="wav")

#create recorder component
def record_page():
    if "wavpath" not in st.session_state:
        cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        tmp_wavpath = TMP_DIR / f'{cur_time}.wav'
        st.session_state["wavpath"] = str(tmp_wavpath)

    wavpath = st.session_state["wavpath"]

    save_frames_from_audio_receiver(wavpath)

    if Path(wavpath).exists():
        sanitize_audio(wavpath)
        global file_to_process
        file_to_process = wavpath

st.sidebar.markdown("# Sound to Analyze")
st.sidebar.write('Record a sound in the browser')
record_page()
st.sidebar.markdown("***")
st.sidebar.markdown("...or upload a wav file...")
uploaded_file = st.sidebar.file_uploader("", type='wav')
st.sidebar.markdown("***")

if uploaded_file is not None:
    file_to_process = uploaded_file

if file_to_process:
    st.sidebar.audio(file_to_process)
    sample_rate, wav_data = wavfile.read(file_to_process, 'rb')
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    duration = len(wav_data)/sample_rate
    # st.write(f'File name: {file_to_process}')
    # st.write(f'Sample rate: {sample_rate} Hz')
    # st.write(f'Total duration: {duration:.2f}s')
    # st.write(f'Size of the input: {len(wav_data)}')
    st.markdown("## :tada: Upload successful!")
    process = st.button("Click here to analyze audio!")
else: st.write("<<< Please select some audio to process.")

if process == True:
    with st.spinner("processing"):
        waveform = wav_data / tf.int16.max
        scores, embeddings, spectrogram = model(waveform)
        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()
        infered_class = class_names[scores_np.mean(axis=0).argmax()]
        st.markdown(f'The main sound detected is: **{infered_class}**.')
        st.write("Here's some diagnostics and a timeline of all sounds detected.")
        st.markdown("**[REFRESH](/)** the page to try again.")

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
