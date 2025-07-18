import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2  # Add this import for resizing

# Load dataset
data = pd.read_csv('fer2013.csv')

# Extract features and labels
pixels = data['pixels'].tolist()
width, height = 48, 48

# Convert pixel strings to arrays and emotion labels
faces = []
emotions = []  # Initialize emotions list here
for index, pixel_sequence in enumerate(pixels):
    face = [int(pixel) for pixel in pixel_sequence.split()]
    
    # Check if the pixel sequence has the correct length before reshaping
    if len(face) == width * height:  
        # Convert to NumPy array and resize the face
        face = np.asarray(face).reshape((48, 48))  # Ensure the reshaping
        face = cv2.resize(face, (width, height))  # Resize to 48x48 just in case
        face = face.astype('float32') / 255.0  # Normalize pixel values
        
        faces.append(face)
        # Append the emotion label for this face
        emotions.append(data['emotion'][index])  
    else:
        print(f"Skipping invalid pixel sequence with length {len(face)}")  # Print a message for skipped sequences

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)  # Shape becomes (num_samples, 48, 48, 1)

# One-hot encode emotion labels
emotions = pd.get_dummies(emotions).values  # One-hot encode after cleaning

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 emotions in FER-2013

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_test, y_test),
    shuffle=True
)

model.save('emotion_model.h5')
