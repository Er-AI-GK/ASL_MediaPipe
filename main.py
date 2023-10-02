import keyboard
import numpy as np
import os
import mediapipe as mp
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from my_functions import *
from tensorflow.keras.models import load_model

# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Load the trained model
model = load_model('my_model')

# Initialize the lists
sentence, keypoints, last_prediction = [], [], []

# Load the GIF image
root = Tk()
root.withdraw()  # Hide the main window
gif_filename = askopenfilename(title="Select a GIF file")
root.destroy()
#gif_filename2 = 'fgh.gif'
cap = cv2.VideoCapture(gif_filename)

if not cap.isOpened():
    print("Cannot open GIF file.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    frame_number = 0
    while frame_number < frame_count:
        _, image = cap.read()
        if image is None:
            break

        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results = image_process(image, holistic)
        # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
        keypoints.append(keypoint_extraction(results))

        # Check if 10 frames have been accumulated
        if len(keypoints) == 10:
            # Convert keypoints list to a numpy array
            keypoints = np.array(keypoints)
            # Make a prediction on the keypoints using the loaded model
            prediction = model.predict(keypoints[np.newaxis, :, :])
            # Clear the keypoints list for the next set of frames
            keypoints = []

            # Check if the maximum prediction value is above 0.9
            if np.amax(prediction) > 0.9:
                # Check if the predicted sign is different from the previously predicted sign
                if last_prediction != actions[np.argmax(prediction)]:
                    # Append the predicted sign to the sentence list
                    sentence.append(actions[np.argmax(prediction)])
                    # Record a new prediction to use it on the next cycle
                    last_prediction = actions[np.argmax(prediction)]

        # Limit the sentence length to 7 elements to make sure it fits on the GIF
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Reset if the "Spacebar" is pressed
        if keyboard.is_pressed(' '):
            sentence, last_prediction = []

        # Check if the list is not empty
        if sentence:
            # Capitalize the first word of the sentence
            sentence[0] = sentence[0].capitalize()

        # Create the predicted text as a string
        predicted_text = ' '.join(sentence)

        # Display the predicted text on each frame
        cv2.putText(image, predicted_text, (20, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('GIF', image)
        cv2.waitKey(1)

        frame_number += 1

    # Release the GIF
    cap.release()
    cv2.destroyAllWindows()
