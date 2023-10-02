#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:49:59 2023

@author: akatsuki
"""

# Import necessary libraries
import streamlit as st
import os
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import metrics
from itertools import product
import string
import keyboard
#from gingerit.gingerit import GingerIt

# Import functions from my_functions.py
from my_functions import draw_landmarks, image_process, keypoint_extraction

# Set the path to the data directory
DATA_PATH = os.path.join('data')
MODEL_PATH = 'my_model'

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(DATA_PATH))

# Create a Streamlit web app
st.title("Sign Language Recognition App")

# Create a function for data collection
def data_collection_module():
    st.header("Data Collection")

    # Define the actions (signs) that will be recorded and stored in the dataset
    actions = np.array(['Hello', 'How', 'More', 'Idea', 'Good'])

    # Define the number of sequences and frames to be recorded for each action
    sequences = 30
    frames = 10

    # Create directories for each action, sequence, and frame in the dataset
    for action, sequence in product(actions, range(sequences)):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

    # Access the camera and check if the camera is opened successfully
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access camera.")
        return

    # Create a MediaPipe Holistic object for hand tracking and landmark extraction
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        # Loop through each action, sequence, and frame to record data
        for action, sequence, frame in product(actions, range(sequences), range(frames)):
            if frame == 0:
                st.write(f"Recording data for the '{action}' sign. Sequence number {sequence}.")
                st.write("Press 'Space' when you are ready to start recording.")
                if st.button("Start Recording"):
                    while True:
                        if st.button("Stop Recording"):
                            break
                        _, image = cap.read()
                        results = image_process(image, holistic)
                        draw_landmarks(image, results)
                        
                        st.image(image, caption=f"Recording: {action}, Sequence: {sequence}, Frame: {frame}", use_column_width=True)
                        
                        keypoints = keypoint_extraction(results)
                        frame_path = os.path.join(DATA_PATH, action, str(sequence), str(frame))
                        np.save(frame_path, keypoints)
                        
                        if not cap.isOpened() or not st.button("Start Recording"):
                            break
            else:
                _, image = cap.read()
                results = image_process(image, holistic)
                draw_landmarks(image, results)
                st.image(image, caption=f"Recording: {action}, Sequence: {sequence}, Frame: {frame}", use_column_width=True)

    # Release the camera
    cap.release()

# Create a function for model training
def train_model():
    st.header("Model Training")

    # Define the number of sequences and frames
    sequences = 30
    frames = 10

    # Create a label map to map each action label to a numeric value
    label_map = {label: num for num, label in enumerate(actions)}

    # Initialize empty lists to store landmarks and labels
    landmarks, labels = [], []

    # Iterate over actions and sequences to load landmarks and corresponding labels
    for action, sequence in product(actions, range(sequences)):
        temp = []
        for frame in range(frames):
            npy = np.load(os.path.join(DATA_PATH, action, str(sequence), str(frame) + '.npy'))
            temp.append(npy)
        landmarks.append(temp)
        labels.append(label_map[action])

    # Convert landmarks and labels to numpy arrays
    X, Y = np.array(landmarks), to_categorical(labels).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

    # Define the model architecture
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(frames, 126)))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(32, return_sequences=False, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, Y_train, epochs=100)

    # Save the trained model
    model.save('mymodel')

    st.success("Model training complete and saved.")

# Create a function for real-time sign language recognition
def recognize_sign():
    st.header("Real-time Sign Language Recognition")

    # Load the trained model
    model = load_model('mymodel')

    # Create an instance of the GingerIt grammar correction tool
    parser = GingerIt()

    # Initialize the lists
    sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

    # Access the camera and check if the camera is opened successfully
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access camera.")
        return

    # Create a holistic object for sign prediction
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while True:
            _, image = cap.read()
            results = image_process(image, holistic)
            draw_landmarks(image, results)
            keypoints.append(keypoint_extraction(results))

            if len(keypoints) == 10:
                keypoints = np.array(keypoints)
                prediction = model.predict(keypoints[np.newaxis, :, :])
                keypoints = []

                if np.amax(prediction) > 0.9:
                    if last_prediction != actions[np.argmax(prediction)]:
                        sentence.append(actions[np.argmax(prediction)])
                        last_prediction = actions[np.argmax(prediction)]

            if len(sentence) > 7:
                sentence = sentence[-7:]

            if keyboard.is_pressed(' '):
                sentence, keypoints, last_prediction, grammar, grammar_result = []

            if sentence:
                sentence[0] = sentence[0].capitalize()

            if len(sentence) >= 2:
                if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                    if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (
                            sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                        sentence[-1] = sentence[-2] + sentence[-1]
                        sentence.pop(len(sentence) - 2)
                        sentence[-1] = sentence[-1].capitalize()

            if keyboard.is_pressed('enter'):
                text = ' '.join(sentence)
                grammar = parser.parse(text)
                grammar_result = grammar['result']

            if grammar_result:
                textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_X_coord = (image.shape[1] - textsize[0]) // 2
                cv2.putText(image, grammar_result, (text_X_coord, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_X_coord = (image.shape[1] - textsize[0]) // 2
                cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            st.image(image, caption="Real-time Sign Language Recognition", use_column_width=True)

            if not cap.isOpened():
                break

    cap.release()

# Create navigation sidebar
st.sidebar.header("Navigation")
selected_page = st.sidebar.radio("Go to", ["Data Collection", "Train Model", "Recognize Sign"])

# Display selected module
if selected_page == "Data Collection":
    data_collection_module() 
elif selected_page == "Train Model":
    train_model()
elif selected_page == "Recognize Sign":
    recognize_sign()
