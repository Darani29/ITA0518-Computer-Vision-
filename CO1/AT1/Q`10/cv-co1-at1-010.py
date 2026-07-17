# ===========================================
# Sampling and Image Formation Demonstration
# Google Colab Compatible
# ===========================================

import cv2
import matplotlib.pyplot as plt
from google.colab import files

# ------------------------------------------
# Step 1: Upload Image
# ------------------------------------------
print("Upload an image")
uploaded = files.upload()

image_path = list(uploaded.keys())[0]

image = cv2.imread(image_path)

if image is None:
    print("Error loading image!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ------------------------------------------
# Step 2: Original Image
# ------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# ------------------------------------------
# Step 3: Simulate Low Sampling
# Reduce Resolution
# ------------------------------------------
low_resolution = cv2.resize(
    gray,
    (gray.shape[1]//4, gray.shape[0]//4),
    interpolation=cv2.INTER_AREA
)

# Upsample for display
upsampled = cv2.resize(
    low_resolution,
    (gray.shape[1], gray.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# ------------------------------------------
# Step 4: Edge Detection
# ------------------------------------------
original_edges = cv2.Canny(gray,100,200)

low_edges = cv2.Canny(upsampled,100,200)

# ------------------------------------------
# Step 5: Display Results
# ------------------------------------------

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(gray,cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(upsampled,cmap='gray')
plt.title("Low Resolution Image")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(original_edges,cmap='gray')
plt.title("Edges in Original Image")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(low_edges,cmap='gray')
plt.title("Edges in Low Resolution Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# ------------------------------------------
# Step 6: Conclusion
# ------------------------------------------
print("========== RESULT ==========")

print("Original Resolution :", gray.shape)
print("Low Resolution :", low_resolution.shape)

print("\nObservation:")
print("- High sampling preserves image details.")
print("- Low sampling removes fine object details.")
print("- Object boundaries become blurred.")
print("- Edge detection becomes less accurate.")
print("- Image formation quality directly affects object visibility.")
