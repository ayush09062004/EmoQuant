from google.colab import files
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Step 4: Load and preprocess the FER-2013 dataset
uploaded = files.upload()
dataset_zip = list(uploaded.keys())[0]
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall('/content/fer2013')

print("Dataset extracted!")

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

train_folder = '/content/fer2013/train/'
test_folder = '/content/fer2013/test/'
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

X_train, y_train = load_images_from_folder(train_folder)
X_test, y_test = load_images_from_folder(test_folder)

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN input
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# Convert grayscale to RGB
def convert_to_rgb(images):
    return np.stack([np.stack([img.squeeze()]*3, axis=-1) for img in images])

X_train_rgb = convert_to_rgb(X_train)
X_test_rgb = convert_to_rgb(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train_rgb, y_train, test_size=0.2, random_state=42)

# Step 5: Build and train the model using ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)

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
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[lr_reduction, early_stopping],
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_rgb, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Save the model
model_path = '/content/drive/My Drive/emotion_detection_model_resnet.h5'
model.save(model_path)
print(f"Model saved to {model_path}")

# Predict on the test set
y_pred = model.predict(X_test_rgb)
y_pred_labels = np.argmax(y_pred, axis=1)

# Generate confusion matrix and classification report
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_labels)
cr = classification_report(np.argmax(y_test, axis=1), y_pred_labels)

print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)



