import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import layers, models
from math import radians  # For tilt angle conversion
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess data with augmentation
def load_and_preprocess_data(dataset_path, image_size, angle_range=5):
    data = []
    labels = []

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_size, image_size))  # Resize image

            # Random tilt augmentation
            image = random_tilt(image, radians(angle_range))  # Convert angle to radians

            # Random flip augmentation
            flip_code = np.random.choice([0, 1])
            image = cv2.flip(image, flip_code)

            image = image.astype('float32') / 255.0  # Normalize pixel values
            data.append(image)
            labels.append(folder)
  
    data = np.array(data)
    labels = np.array(labels)
  
    return data, labels

# Function for random tilt with angle conversion
def random_tilt(image, angle_range):
    tilt_angle = np.random.uniform(low=-angle_range, high=angle_range)  # Degrees

    # Convert angle to radians for cv2.getRotationMatrix2D
    tilt_angle_rad = radians(tilt_angle)

    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), tilt_angle_rad, 1)

    return cv2.warpAffine(image, rotation_matrix, (cols, rows))

# Define CNN model
def create_model(image_size, num_classes, l1_lambda=0.000):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3), kernel_regularizer=keras.regularizers.L1(l1_lambda)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.L1(l1_lambda)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.L1(l1_lambda)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load and preprocess data
train_data_path = "C:/Users/SIDDHESH/Desktop/hand sign detection/Model/keras_model.h5"  # Replace with your path
image_size = 64
angle_range = 5  # Adjust tilt angle range (in degrees)
train_data, train_labels = load_and_preprocess_data(train_data_path, image_size, angle_range)

# One-hot encode labels
label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)

# Split data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=21)

# Define and train model
num_classes = len(label_binarizer.classes_)  # Get number of output classes
model = create_model(image_size, num_classes)
history = model.fit(train_images, train_labels, epochs=7, batch_size=16, validation_data=(val_images, val_labels))

# Evaluate the model
val_loss, val_acc = model.evaluate(val_images, val_labels)
print('Validation accuracy:', val_acc)

# Predict the labels for validation images
val_predictions = np.argmax(model.predict(val_images), axis=1)
val_true_labels = np.argmax(val_labels, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(val_true_labels, val_predictions)

# Save labels to a file in the current directory
label_file_path = "C:/Users/SIDDHESH/Desktop/hand sign detection/Model/labels.txt"  # Define the path for the label file
os.makedirs(os.path.dirname(label_file_path), exist_ok=True)  # Create director/y if not exists
with open(label_file_path, "w") as file:
    for label in label_binarizer.classes_:
        file.write(label + "\n")
        
# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
            xticklabels=label_binarizer.classes_, yticklabels=label_binarizer.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Save the model
model.save("keras_model.h5")