from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sqlite3
import os

app = Flask(__name__)

model = load_model('face_emotionModel.h5')

# Emotion labels in FER2013 order
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Create database if missing
conn = sqlite3.connect('database.db')
conn.execute('CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, emotion TEXT)')
conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        filepath = os.path.join('static', file.filename)
        os.makedirs('static', exist_ok=True)
        file.save(filepath)

        # Load image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48,48))
        img = img.reshape(1,48,48,1) / 255.0

        # Predict emotion
        emotion_index = np.argmax(model.predict(img))
        emotion = emotion_labels[emotion_index]

        # Log to database
        conn = sqlite3.connect('database.db')
        conn.execute("INSERT INTO uploads (filename, emotion) VALUES (?, ?)", (file.filename, emotion))
        conn.commit()
        conn.close()

        return f"Predicted Emotion: {emotion}"
    return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)
