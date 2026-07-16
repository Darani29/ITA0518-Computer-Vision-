# =====================================================
# Facial Recognition Pipeline using OpenCV (Google Colab)
# =====================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# -----------------------------------------------------
# Step 1: Image Acquisition
# -----------------------------------------------------
print("Upload a face image")
uploaded = files.upload()

image_path = list(uploaded.keys())[0]

img = cv2.imread(image_path)

if img is None:
    print("Image could not be loaded.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(5,5))
plt.imshow(img_rgb)
plt.title("Image Acquisition")
plt.axis("off")
plt.show()

# -----------------------------------------------------
# Step 2: Low-Level Computer Vision
# Image Preprocessing
# -----------------------------------------------------

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Remove noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Improve contrast
equalized = cv2.equalizeHist(blur)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(blur, cmap='gray')
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(equalized, cmap='gray')
plt.title("Histogram Equalization")
plt.axis("off")

plt.show()

# -----------------------------------------------------
# Step 3: Mid-Level Computer Vision
# Face Detection
# -----------------------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

faces = face_detector.detectMultiScale(
    equalized,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30)
)

output = img_rgb.copy()

for (x,y,w,h) in faces:
    cv2.rectangle(output,
                  (x,y),
                  (x+w,y+h),
                  (0,255,0),
                  3)

plt.figure(figsize=(6,6))
plt.imshow(output)
plt.title("Detected Face")
plt.axis("off")
plt.show()

# -----------------------------------------------------
# Step 4: High-Level Computer Vision
# Decision Making
# -----------------------------------------------------

print("----------- FINAL DECISION -----------")

if len(faces) > 0:
    print("Face Detected")
    print("Identity verification can now be performed.")
else:
    print("No Face Detected")

print("--------------------------------------")
print("Number of Faces Detected :", len(faces))
