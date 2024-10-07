import numpy as np
from skimage.filters import gaussian
from skimage.feature import blob_log
import matplotlib.pyplot as plt

image = plt.imread('butterfly.jpg')

if image.ndim == 3:
    image = image[:, :, 0]  # i will take the first channel


# I will apply the Gaussian blur to reduce noise
image_blurred = gaussian(image, sigma=1)

# Detecting blobs
blobs = blob_log(image_blurred, max_sigma=30, num_sigma=10, threshold=0.1)


blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

# Plot the blobs on the original image
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(image, cmap='gray')
for blob in blobs:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax.add_patch(c)

plt.show()