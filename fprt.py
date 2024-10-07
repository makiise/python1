import numpy as np
import matplotlib.pyplot as plt


def edge_detection(image, a=0.9, b=0.01):
    # Convert image to grayscale
    gray_image = np.mean(image, axis=2)

    # Compute derivatives
    dxx = np.gradient(np.gradient(gray_image, axis=1), axis=1)
    dy = np.gradient(gray_image, axis=0)

    # Compute combined derivative using linear combination
    combined_derivative = a * dxx + b * dy

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(dxx ** 2 + dy ** 2)

    return combined_derivative, grad_magnitude


# Load input image
input_image = plt.imread('venus.jpg')

# Perform edge detection using linear combination of derivatives
edges, grad_magnitude = edge_detection(input_image, a=0.9, b=0.01)

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detected (a=0.9, b=0.01)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(grad_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.show()