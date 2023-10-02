import cv2
import mediapipe as mp
import os
import numpy as np
from my_functions import *
from tensorflow.keras.models import load_model

# Set the path to the data directory
PATH = os.path.join('data')

# Load the trained model
model = load_model('my_model')

# Create an instance of the GingerIt grammar correction tool
# parser = GingerIt()

# Load the GIF file using OpenCV
gif_path = 'bat.gif'
cap = cv2.VideoCapture(gif_path)

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    frame_counter = 0
    sentence = []

    while True:
        ret, image = cap.read()
        if not ret:
            break  # No more frames in the GIF

        results = image_process(image, holistic)
        # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
        keypoints = keypoint_extraction(results)

        # Check if the maximum prediction value is above 0.9
        prediction = model.predict(keypoints[np.newaxis, :, :])
        if np.amax(prediction) > 0.9:
            recognized_action = actions[np.argmax(prediction)]
            sentence.append(recognized_action)

        # Reset if the "Spacebar" is pressed
        if keyboard.is_pressed(' '):
            sentence = []

        # ... (rest of the code for grammar check and displaying results)

        # Show the image on the display
        cv2.imshow('GIF Frame', image)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
            break

    cap.release()
    cv2.destroyAllWindows()
