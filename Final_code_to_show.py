import cv2
import numpy as np
import math
import time
import tkinter as tk
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import pyttsx3

# Initialize text to speech engine
engine = pyttsx3.init()

# Variable to hold formed word
formed_word = ""

# List to hold predicted letters
predicted_letters = []

# Function to update the predicted letter in the text box
def update_predicted_letter(letter):
    text_box1.delete(1.0, tk.END)  # Clear previous content
    text_box1.insert(tk.END, letter)  # Update with new letter

# Function to update the formed word in the text box
def update_formed_word(word):
    text_box2.delete(1.0, tk.END)  # Clear previous content
    text_box2.insert(tk.END, word)  # Update with new word

# Function to handle Speak button click
def speak_word():
    engine.say(text_box2.get(1.0, tk.END))
    engine.runAndWait()

# Function to process video feed and predict hand gesture
def process_video_feed():
    global formed_word  # Declare formed_word as a global variable
    global predicted_letters  # Declare predicted_letters as a global variable
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)

            imgWhite[:, wGap: wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw= False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[ hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw= False)

        letter = labels[index]
        update_predicted_letter(letter)
        
        # Only add the predicted letter to the formed word if it's not already present
        if letter not in predicted_letters:
            formed_word += letter
            predicted_letters.append(letter)
            update_formed_word(formed_word)

    cv2.imshow("Image", imgOutput)
    root.after(10, process_video_feed)  # Repeat the function every 10 milliseconds

# Initialize Tkinter GUI
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("800x600")

# Create frame for displaying camera feed
frame = tk.Frame(root, bg='')
frame.pack(padx=10, pady=10)

# Create label for camera feed
label = tk.Label(frame)
label.pack()
##########################################
# # Create text box for displaying predicted letter
# text_box1 = tk.Text(root, height=1, width=10, font=("Helvetica", 24))
# text_box1.pack(pady=10

# # Create text box for displaying formed word
# text_box2 = tk.Text(root, height=1, width=20, font=("Helvetica", 24))
# text_box2.pack(pady=10)
###########################################

label1 = tk.Label(root, text="WORD:", font=("Helvetica", 24))
label1.pack(padx=10, pady=10)

# Create the first text box
text_box1 = tk.Text(root, height=1, width=10, font=("Helvetica", 24))
text_box1.pack(pady=10)

# Create label for the second text box
label2 = tk.Label(root, text="SENTENCES:", font=("Helvetica", 24))
label2.pack(padx=10, pady=10)

# Create the second text box
text_box2 = tk.Text(root, height=1, width=20, font=("Helvetica", 24))
text_box2.pack(pady=10)

# Create Speak button
speak_button = tk.Button(root, text="Speak", command=speak_word)
speak_button.pack(padx=10,pady=10)

# Initialize camera and hand detection
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Start processing video feed
process_video_feed()

# Start Tkinter event loop
root.mainloop()

# Release camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()


