# ===============================================
# Facial Recognition Pipeline using OpenCV
# Google Colab Compatible
# ===============================================

import cv2
import matplotlib.pyplot as plt
from google.colab import files

# ---------------------------------------
# Step 1: Image Acquisition
# ---------------------------------------
print("Upload a face image")
uploaded = files.upload()

image_path = list(uploaded.keys())[0]

# Read the uploaded image
image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# Convert BGR to RGB for display
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(5,5))
plt.imshow(rgb_image)
plt.title("Step 1 : Acquired Image")
plt.axis("off")
plt.show()

# ---------------------------------------
# Step 2: Low-Level Computer Vision
# Image Preprocessing
# ---------------------------------------

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reduce noise
gray = cv2.GaussianBlur(gray,(5,5),0)

# Improve contrast
gray = cv2.equalizeHist(gray)

plt.figure(figsize=(5,5))
plt.imshow(gray,cmap="gray")
plt.title("Step 2 : Preprocessed Image")
plt.axis("off")
plt.show()

# ---------------------------------------
# Step 3: Mid-Level Computer Vision
# Face Detection
# ---------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30)
)

output = rgb_image.copy()

for (x,y,w,h) in faces:
    cv2.rectangle(output,
                  (x,y),
                  (x+w,y+h),
                  (0,255,0),
                  3)

plt.figure(figsize=(6,6))
plt.imshow(output)
plt.title("Step 3 : Face Detection")
plt.axis("off")
plt.show()

# ---------------------------------------
# Step 4: High-Level Computer Vision
# Decision Making
# ---------------------------------------

print("========== FINAL DECISION ==========")

if len(faces) > 0:
    print("Face Detected")
    print("Identity verification can now be performed.")
else:
    print("No Face Detected")

print("Number of Faces Detected:", len(faces))
print("====================================")
