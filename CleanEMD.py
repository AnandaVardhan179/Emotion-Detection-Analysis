import numpy as np
import pandas as pd
import os
import cv2
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

def load_fer2013_data(data_path):
    images = []
    labels = []
    emotions = sorted(os.listdir(data_path)) 
    for label, emotion in enumerate(emotions):
        emotion_folder = os.path.join(data_path, emotion)
        for subdir, _, files in os.walk(emotion_folder):
            for img_file in files:
                img_path = os.path.join(subdir, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                image = cv2.resize(image, (64, 64))
                images.append(image)
                labels.append(label)
    images = np.array(images)
    images = np.expand_dims(images, -1)
    labels = np.array(labels)
    return images, labels

fer2013_path = 'fer2013'
images, labels = load_fer2013_data(fer2013_path)
images = images / 255.0 
print("FER2013 Data Loaded and Preprocessed")

def load_ravdess_data(data_path):
    audio_features = []
    audio_labels = []
    actors = sorted(os.listdir(data_path))
    for label, actor in enumerate(actors):
        actor_folder = os.path.join(data_path, actor)
        if not os.access(actor_folder, os.R_OK):
            print(f"Permission denied: {actor_folder}")
            continue
        for audio_file in os.listdir(actor_folder):
            audio_path = os.path.join(actor_folder, audio_file)
            try:
                y, sr = librosa.load(audio_path, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                mfccs = np.mean(mfccs.T, axis=0)
                audio_features.append(mfccs)
                audio_labels.append(label)
            except PermissionError:
                print(f"Permission denied: {audio_path}")
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
    return np.array(audio_features), np.array(audio_labels)

ravdess_path = r'ravdess\audio_speech_actors_01-24' 
audio_features, audio_labels = load_ravdess_data(ravdess_path)
print("RAVDESS Data Loaded and Preprocessed")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


X_train, X_test, y_train, y_test = train_test_split(audio_features, audio_labels, test_size=0.2, random_state=42)


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)


X_test_scaled = scaler.transform(X_test)


print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import joblib


svm_clf.fit(X_train_scaled, y_train)


joblib.dump(svm_clf, 'model.pkl')


y_pred = svm_clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


print(classification_report(y_test, y_pred))
