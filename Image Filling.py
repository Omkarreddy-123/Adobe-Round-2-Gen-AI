#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "C:\\Users\\ramar\\OneDrive\\Pictures\\Screenshots\\apple.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    print("Error: Image not loaded. Check the file path.")
else:
    # Apply edge detection with adjusted parameters
    edges = cv2.Canny(image, 30, 100)

    # Use morphological operations to close gaps in the edges
    kernel = np.ones((5, 5), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to fill the shapes
    mask = np.zeros_like(image)

    # Fill all detected contours on the mask, filtering out small contours
    min_contour_area = 100  # Adjust the area threshold as needed
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Create a copy of the original image to modify
    final_image = image.copy()

    # Fill the shapes with black only within the detected contours
    final_image[mask == 255] = 0

    # Visualize the contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)

    # Display the original, edge-detected, contour visualization, and final images
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 4, 2)
    plt.title('Edge Detection')
    plt.imshow(edges, cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title('Contours')
    plt.imshow(contour_image, cmap='gray')

    plt.subplot(1, 4, 4)
    plt.title('Final Image with Filled Shapes')
    plt.imshow(final_image, cmap='gray')

    plt.show()


# In[ ]:





# In[ ]:




