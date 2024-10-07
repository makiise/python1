import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# I chose the image of the David the greatest Bowie
image = plt.imread('bacteri')

# convert the image to the grayscale
image_gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

# Sobel matrix for central difference
sobel_central = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

# Sobel matrix for forward difference
sobel_forward = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])

# computing gradients using central difference formulas
gy_central, gx_central = np.gradient(image_gray)

# Computing gradients using forward difference formulas
gy_forward = signal.convolve2d(image_gray, sobel_forward.T, 'same')
gx_forward = signal.convolve2d(image_gray, sobel_forward, 'same')

# Now, we need to Compute edge magnitudes
edge_mag_central = np.sqrt(gx_central**2 + gy_central**2)
edge_mag_forward = np.sqrt(gx_forward**2 + gy_forward**2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image_gray, cmap='gray')
axes[0].set_title('Grayscale Image')
axes[1].imshow(edge_mag_central, cmap='gray')
axes[1].set_title('Central Difference')
axes[2].imshow(edge_mag_forward, cmap='gray')
axes[2].set_title('Forward Difference')
plt.show()
