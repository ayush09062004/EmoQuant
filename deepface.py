#we took 7 test images
# Install DeepFace and its dependencies
!pip install deepface
!pip install opencv-python-headless


import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from google.colab import files

# Upload images
uploaded = files.upload()

# List of uploaded file names
uploaded_files = list(uploaded.keys())

# Ensure exactly 7 images are uploaded (for testing)
if len(uploaded_files) != 7:
    raise ValueError("Please upload exactly 7 images.")
print(f"Uploaded files: {uploaded_files}")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image {image_path}")

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, img

def predict_emotion(image, faces):
    emotions = []
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        try:
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            dominant_emotion = result['dominant_emotion']
            emotions.append(dominant_emotion)
        except Exception as e:
            emotions.append(f"Error: {str(e)}")
    return emotions

# Display images and predict emotions
fig, axs = plt.subplots(1, 7, figsize=(20, 5))

for i, file_name in enumerate(uploaded_files):
    # Detect faces in the image
    faces, img = detect_faces(file_name)

    # Predict emotions for detected faces
    emotions = predict_emotion(img, faces)

    # Display image and detected emotions
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axs[i].imshow(img_rgb)
    axs[i].set_title(', '.join(emotions) if emotions else 'No face detected')
    axs[i].axis('off')

plt.show()

