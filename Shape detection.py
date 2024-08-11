#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# Load the image
image_path = "C:\\Users\\ramar\\OneDrive\\Pictures\\Screenshots\\apple.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def detect_shapes(contours):
    shapes = {"Line": 0, "Circle/Ellipse": 0, "Rectangle": 0, 
              "Rounded Rectangle": 0, "Triangle": 0, "Pentagon": 0, 
              "Hexagon": 0, "Polygon": 0, "Unidentified": 0}
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        shape = "Unidentified"
        
        # Detect straight lines
        if len(approx) == 2:
            shape = "Line"
        
        # Detect circles and ellipses
        elif len(approx) > 10:
            shape = "Circle/Ellipse"
        
        # Detect rectangles
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            shape = "Rectangle" if aspect_ratio >= 0.95 and aspect_ratio <= 1.05 else "Rounded Rectangle"
        
        # Detect regular polygons (triangles, pentagons, etc.)
        elif len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 5:
            shape = "Pentagon"
        elif len(approx) == 6:
            shape = "Hexagon"
        elif len(approx) > 6:
            shape = "Polygon"
        
        shapes[shape] += 1

    return shapes

# Detect and label shapes
shapes = detect_shapes(contours)

# Print the shape names and counts
for shape, count in shapes.items():
    print(f"{shape}: {count}")


# In[ ]:




