import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def higher_order_edge_detection(image, order):
    # Convert image to grayscale
    gray_image = np.mean(image, axis=2)

    # Compute higher-order derivatives
    if order == 4:
        derivative_x = np.gradient(np.gradient(np.gradient(gray_image, axis=0), axis=0), axis=0)
        derivative_y = np.gradient(np.gradient(np.gradient(gray_image, axis=1), axis=1), axis=1)
    elif order == 10:
        # Define separable kernel for the 10th order derivative
        kernel_1d = np.array([1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1])
        kernel_2d = np.outer(kernel_1d, kernel_1d)

        # Compute the 10th order derivative using 2D convolution
        derivative_x = convolve2d(gray_image, kernel_2d, mode='same', boundary='wrap')
        derivative_y = convolve2d(gray_image, kernel_2d.T, mode='same', boundary='wrap')
    else:
        raise ValueError("Invalid order. Order should be 4 or 10.")

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(derivative_x ** 2 + derivative_y ** 2)

    # Normalize gradient magnitude to [0, 255]
    grad_magnitude = (grad_magnitude * 255 / np.max(grad_magnitude)).astype(np.uint8)

    return grad_magnitude


# Load input image
input_image = plt.imread('venus.jpg')

# Perform edge detection using different orders
plt.figure(figsize=(10, 5))

# Compute and display order 4 derivative
edges_order_4 = higher_order_edge_detection(input_image, 4)
plt.subplot(1, 2, 1)
plt.imshow(edges_order_4, cmap='gray')
plt.title('Order 3 Edge Detection')
plt.axis('off')

# Compute and display order 10 derivative
edges_order_10 = higher_order_edge_detection(input_image, 10)
plt.subplot(1, 2, 2)
plt.imshow(edges_order_10, cmap='gray')
plt.title('Order 11 Edge Detection')
plt.axis('off')

plt.show()
