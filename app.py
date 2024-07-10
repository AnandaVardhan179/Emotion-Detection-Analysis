from flask import Flask, request, render_template, jsonify
from moviepy.editor import VideoFileClip
import os
import joblib
import numpy as np
import librosa

app = Flask(__name__)

# Load your trained model
model = joblib.load('model.pkl')

def extract_audio_from_video(video_path, audio_output_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        video_path = os.path.join('uploads', file.filename)
        audio_output_path = os.path.join('uploads', 'output_audio.wav')
        file.save(video_path)
        extract_audio_from_video(video_path, audio_output_path)

        # Extract features from the audio
        y, sr = librosa.load(audio_output_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        mfccs = np.expand_dims(mfccs, axis=0)

        # Scale the features
        scaler = joblib.load('scaler.pkl')
        mfccs_scaled = scaler.transform(mfccs)

        # Predict the emotion
        prediction = model.predict(mfccs_scaled)
        return jsonify({'emotion': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
