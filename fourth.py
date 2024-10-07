import numpy as np
import matplotlib.pyplot as plt


def edge_detection(image, a=0.7, b=0.3):
    # Convert image to grayscale
    gray_image = np.mean(image, axis=2)

    # Compute derivatives
    dx = np.gradient(gray_image, axis=1)
    dy = np.gradient(gray_image, axis=0)

    # Compute combined derivative using linear combination
    combined_derivative = a * dx + b * dy

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(dx ** 2 + dy ** 2)

    return combined_derivative, grad_magnitude


input_image = plt.imread('venus.jpg')

edges, grad_magnitude = edge_detection(input_image, a=0.7, b=0.3)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detected (a=0.7, b=0.3)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(grad_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.show()