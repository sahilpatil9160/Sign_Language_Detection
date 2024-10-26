# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import  Classifier
# import numpy as np
# import math
# import time

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifter = Classifier("Model/keras_model.h5", "Model/labels.txt")

# offset = 20
# imgSize = 300

# folder = "Data/C"
# counter = 0

# lables = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# while True:
#     success, img = cap.read()
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']

#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

#         imgCropShape = imgCrop.shape


#         aspectRatio = h / w

#         if aspectRatio > 1:
#             k = imgSize/h
#             wCal = math.ceil(k*w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize-wCal)/2)

#             imgWhite[:, wGap: wCal+wGap] = imgResize
#             prediction, index = classifter.getPrediction(imgWhite, draw= False)
#             print(prediction, index)

#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)

#             imgWhite[ hGap: hCal + hGap, :] = imgResize
#             prediction, index = classifter.getPrediction(imgWhite, draw= False)


#         cv2.putText(imgOutput, lables[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
#         cv2.rectangle(imgOutput, (x-offset, y-offset),
#                       (x+w+offset, y+h+offset), (255, 0, 255), 4)



#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)

#     cv2.imshow("Image", imgOutput)
#     key = cv2.waitKey(1)



#####################################################################################################################333

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Initialize hand detector with maxHands=1
detector = HandDetector(maxHands=1)

# Initialize classifier with pre-trained model and labels
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants and variables
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
labels = [chr(i) for i in range(65, 91)]  # A to Z

while True:
    # Capture frames from webcam
    success, frame = cap.read()

    if not success:
        break

    # Find hands in the frame
    frame = cv2.flip(frame, 1)
    hands, _ = detector.findHands(frame)

    if hands:
        # Get hand landmarks
        hand = hands[0]
        lmList = hand["lmList"]

        # Get bounding box around hand
        bbox = detector.boundingbox(frame, hand, draw=False)

        # Crop hand region
        x, y, w, h = bbox
        imgCrop = frame[y-offset:y+h+offset, x-offset:x+w+offset]

        # Resize cropped image
        imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))

        # Predict gesture
        label, confidence = classifier.getPrediction(imgCrop)

        # Get index of predicted label
        index = labels.index(label)

        # Draw label and bounding box on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(frame, f'{label} {confidence*100:.2f}%', (bbox[0], bbox[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Display cropped hand image and resized image
        cv2.imshow("Cropped Hand", imgCrop)
        imgWhite = np.ones((300, 300), dtype=np.uint8) * 255
        imgWhite[:imgCrop.shape[0], :imgCrop.shape[1]] = imgCrop
        cv2.imshow("Resized Image", imgWhite)

    # Display output image
    cv2.imshow("Hand Gesture Recognition", frame)

    # Wait for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()


