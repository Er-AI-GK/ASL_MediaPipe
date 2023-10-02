import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Prompt the user to enter the actions (signs) they want to record and store in the dataset
actions_str = input("Enter the actions (signs) to be recorded, separated by commas: ")
actions = np.array([action.strip() for action in actions_str.split(',')])
sequences = 30
frames = 10
# Set the path where the dataset will be stored
PATH = os.path.join('data')

# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

# Use a file dialog to manually select the GIF file
root = Tk()
root.withdraw()  # Hide the main window
gif_path = askopenfilename(title="Select a GIF file")
root.destroy()

if not gif_path:
    print("No GIF file selected. Exiting.")
else:
    # Load the selected GIF file
    gif = cv2.VideoCapture(gif_path)

    # Loop through each action, sequence, and frame to record data
    for action, sequence, frame in product(actions, range(sequences), range(frames)):
        # Read the next frame from the GIF
        ret, image = gif.read()

        # If the GIF has ended, start from the beginning
        if not ret:
            gif.release()
            gif = cv2.VideoCapture(gif_path)
            ret, image = gif.read()

        # Process the image and extract hand landmarks using the MediaPipe Holistic pipeline
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            results = image_process(image, holistic)
            # Draw the hand landmarks on the image
            draw_landmarks(image, results)

            # Display text on the image indicating the action and sequence number being recorded
            cv2.putText(image, 'Recording data for "{}". Sequence number {}.'.format(action, sequence),
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('GIF Frames', image)
            cv2.waitKey(1)

        # Extract the landmarks from both hands and save them in arrays
        keypoints = keypoint_extraction(results)
        frame_path = os.path.join(PATH, action, str(sequence), str(frame))
        np.save(frame_path, keypoints)

    # Release the GIF and close any remaining windows
    gif.release()
    cv2.destroyAllWindows()
