

from google.colab import files
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Step 1: Upload video
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# Step 2: Extract frames every 90 seconds with face detection, crop faces, and convert to grayscale
output_folder = '/content/frames/'
os.makedirs(output_folder, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = 90 * fps
frame_count = 0
saved_face_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]  # Crop the face from the grayscale frame
            face_path = os.path.join(output_folder, f'face_{saved_face_count}.jpg')
            cv2.imwrite(face_path, face)
            saved_face_count += 1

    frame_count += 1

cap.release()
print(f"Saved {saved_face_count} grayscale face images.")

# Step 3: Display saved face images
saved_faces = [f for f in os.listdir(output_folder) if f.startswith('face_')]

for face_file in saved_faces[:5]:  # Display first 5 face images
    img_path = os.path.join(output_folder, face_file)
    img = mpimg.imread(img_path)
    plt.imshow(img, cmap='gray')  # Display images in grayscale
    plt.title(face_file)
    plt.axis('off')
    plt.show()

from google.colab import files
import zipfile
import os

# Upload the FER-2013 dataset zip file
uploaded = files.upload()

# Unzip the dataset
dataset_zip = list(uploaded.keys())[0]
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall('/content/fer2013')

print("Dataset extracted!")

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Define paths
train_folder = '/content/fer2013/train/'
test_folder = '/content/fer2013/test/'

# Categories
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load images from folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in emotion_labels:
        emotion_folder = os.path.join(folder, label)
        for filename in os.listdir(emotion_folder):
            if filename.endswith('.jpg'):
                img_path = os.path.join(emotion_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))  # Resize to 48x48
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load training and test data
X_train, y_train = load_images_from_folder(train_folder)
X_test, y_test = load_images_from_folder(test_folder)

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN input
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build the model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Save the model
model.save('/content/emotion_detection_model.h5')

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

files.download('/content/emotion_detection_model.h5')

from google.colab import drive
drive.mount('/content/drive')
model_path = '/content/drive/My Drive/emotion_detection_model.h5'

# Save the model to Google Drive
model.save(model_path)
import os

# Define the path to the specific folder in Google Drive
# Replace with the path to your specific folder
drive_folder_path = '/content/drive/My Drive/17FRUVkDI8h6StPa6R4Qf0cW2oms4NX1N'
model_path = os.path.join(drive_folder_path, 'emotion_detection_model.h5')

# Save the model to the specified folder
model.save(model_path)

print(f"Model saved to {model_path}")

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    img = cv2.resize(img, (48, 48))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 48, 48, 1)  # Reshape for model prediction
    return img

# Define paths to the images
image_paths = [
    '/content/frames/face_0.jpg',
    '/content/frames/face_1.jpg',
    '/content/frames/face_2.jpg'
]

# Preprocess the images
preprocessed_images = [preprocess_image(path) for path in image_paths]

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Predict emotions
predictions = [model.predict(img) for img in preprocessed_images]
predicted_labels = [emotion_labels[np.argmax(pred)] for pred in predictions]

print("Predicted Emotions:")
for path, label in zip(image_paths, predicted_labels):
    print(f"Image: {path} -> Emotion: {label}")

import matplotlib.pyplot as plt

def display_image_with_prediction(image_path, label):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted Emotion: {label}')
    plt.axis('off')
    plt.show()

# Display images with predicted emotions
for path, label in zip(image_paths, predicted_labels):
    display_image_with_prediction(path, label)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Example usage with a batch of images
# datagen.flow(X_train, y_train, batch_size=32)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

from tensorflow.keras.callbacks import ReduceLROnPlateau

# Reduce learning rate on plateau
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                 patience=2,
                                 verbose=1,
                                 factor=0.5,
                                 min_lr=0.00001)

# Train the model with the learning rate scheduler
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=64,
                    validation_data=(X_val, y_val),
                    callbacks=[lr_reduction],
                    verbose=1)

def convert_to_rgb(images):
    # Convert grayscale images to RGB
    return np.stack([np.stack([img.squeeze()]*3, axis=-1) for img in images])

# Convert the images
X_train_rgb = convert_to_rgb(X_train)
X_val_rgb = convert_to_rgb(X_val)
X_test_rgb = convert_to_rgb(X_test)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create data generator for training data
train_generator = datagen.flow(X_train_rgb, y_train, batch_size=64)

# Optionally, create a generator for validation data if needed
val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(X_val_rgb, y_val, batch_size=64)

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Build Model using VGG16 as base
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[lr_reduction, early_stopping],
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_rgb, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Save the model
model.save('/content/drive/My Drive/17FRUVkDI8h6StPa6R4Qf0cW2oms4NX1N/emotion_detection_model2.h5')

# Predict on the test set
y_pred = model.predict(X_test_rgb)
y_pred_labels = np.argmax(y_pred, axis=1)

# Generate confusion matrix and classification report
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_labels)
cr = classification_report(np.argmax(y_test, axis=1), y_pred_labels)

print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)

