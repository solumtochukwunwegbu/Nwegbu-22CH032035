import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Load FER2013 CSV ---
print("Loading FER2013 dataset...")
data = pd.read_csv('data/fer2013.csv')

# Extract pixels and emotions
pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
X = np.stack(pixels, axis=0)
y = data['emotion'].values

# Normalize pixel values
X = X / 255.0
X = X.reshape(-1, 48, 48, 1)

# One-hot encode emotion labels
y = to_categorical(y, num_classes=7)

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Build CNN Model ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# --- Train the model ---
model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test))

# --- Save the model ---
model.save('face_emotionModel.h5')
print("✅ Model saved as face_emotionModel.h5")
