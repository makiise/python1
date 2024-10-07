import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

image = plt.imread('bacteria.PNG')
image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# Laplacian
laplacian = ndimage.gaussian_laplace(image_gray, sigma=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_gray, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(laplacian, cmap='gray')
axes[1].set_title('Laplacian Edge Detection')
plt.show()