import cv2
import numpy as np
import spotipy
import random
from spotipy.oauth2 import SpotifyOAuth
from keras.models import load_model
from twilio.rest import Client
import tkinter as tk

# ========== CONFIGURATIONS ==========

# Emotion labels (must match the model's output order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load pre-trained emotion recognition model
model = load_model("emotion_model.h5")

# Spotify credentials
SPOTIFY_CLIENT_ID = 'spotify client ID'
SPOTIFY_CLIENT_SECRET = 'spotify client secret'
SPOTIFY_REDIRECT_URI = 'https://open.spotify.com/'

# Twilio credentials
TWILIO_SID = 'Twilio SID'
TWILIO_AUTH_TOKEN = 'Twilio Auth Token'
TWILIO_FROM_NUMBER = 'Your Twilio Number'
TO_NUMBER = 'Your number'  # Number to alert

# ========== SPOTIFY SETUP ==========
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope="user-read-playback-state,user-modify-playback-state"
))


# ========== FUNCTIONS ==========

def play_song(song_name):
    results = sp.search(q=song_name, limit=1, type='track')
    if results['tracks']['items']:
        track_uri = results['tracks']['items'][0]['uri']
        devices = sp.devices()
        if devices['devices']:
            device_id = devices['devices'][0]['id']
            sp.start_playback(device_id=device_id, uris=[track_uri])
            print(f"Playing on Spotify: {song_name}")
        else:
            print("No active Spotify device found.")
    else:
        print("Song not found.")


def show_joke(file_path="jokes.txt"):
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            jokes = [line.strip() for line in f if line.strip()]
        
        if not jokes:
            print("No jokes found in the file.")
            return
        
        joke_text = random.choice(jokes)  # Select a random joke

        # Show it in a Tkinter window
        root = tk.Tk()
        root.title("Here's a Joke!")
        root.geometry("400x200")
        label = tk.Label(root, text=joke_text, font=("Arial", 14), wraplength=380)
        label.pack(pady=40)
        root.after(10000, root.destroy)  # Close after 5 seconds
        root.mainloop()

    except FileNotFoundError:
        print("Joke file not found. Make sure jokes.txt exists.")


def send_sms(body_text):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=body_text,
        from_=TWILIO_FROM_NUMBER,
        to=TO_NUMBER
    )
    print(f"SMS sent: {message.sid}")

# ========== EMOTION DETECTION ==========

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_normalized = roi_resized.astype("float") / 255.0
        roi_expanded = np.expand_dims(roi_normalized, axis=0)
        roi_expanded = np.expand_dims(roi_expanded, axis=-1)

        prediction = model.predict(roi_expanded)
        emotion_index = int(np.argmax(prediction))
        emotion = emotion_labels[emotion_index]

        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # === Actions Based on Emotion ===
        if emotion in ["Happy", "Neutral"]:
            show_joke()
            play_song("https://open.spotify.com/playlist/37i9dQZF1EVJSvZp5AOML2?si=KOfcWW9sROmPwuBXQQh4yg")
            
        elif emotion in ["Sad", "Fear", "Angry"]:
            send_sms("Hey, your friend seems a bit anxious. Maybe check on them?")
            play_song("https://open.spotify.com/playlist/5nsgGWC8hbeHECB4zBtTRO?si=8buVj3KySAidq2tLjBEsgw")

        break  # Act on first detected face only

    cv2.imshow("Emotion Detector", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
