import cv2
import numpy as np
import math
import time
import tkinter as tk
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Function to update the predicted letter in the text box
def update_predicted_letter(letter):
    text_box.delete(1.0, tk.END)  # Clear previous content
    text_box.insert(tk.END, letter)  # Update with new letter

# Function to process video feed and predict hand gesture
def process_video_feed():
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

        update_predicted_letter(labels[index])

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

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

# Create text box for displaying predicted letter
text_box = tk.Text(root, height=1, width=10, font=("Helvetica", 24))
text_box.pack(pady=10)

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
