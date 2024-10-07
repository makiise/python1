import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature

# Load an image
image = io.imread('butterfly.jpg', as_gray=True)

# Detect blobs using the Laplacian of Gaussian (LoG) method
blobs_log = feature.blob_log(image, max_sigma=30, num_sigma=10, threshold=0.1)

# Compute blob radii in the 3rd column
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

# Plot the blobs
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
    ax.add_patch(c)

plt.title('Blobs Detected')
plt.axis('off')
plt.show()
