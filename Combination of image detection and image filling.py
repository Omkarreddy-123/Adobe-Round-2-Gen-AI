#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from svgpathtools import svg2paths2
import numpy as np
import cv2
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
from io import BytesIO
import base64
from scipy.optimize import leastsq
import math
import matplotlib.pyplot as plt

# Function to determine the shape of the object
def detect_shape(contour):
    shape = "unidentified"
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
    num_vertices = len(approx)
    
    if num_vertices == 3:
        shape = "triangle"
    elif num_vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif 5 <= num_vertices <= 6:  # Handle pentagon and hexagon
        shape = "polygon"
    elif num_vertices > 6:  # Handle polygons with more than 6 sides
        shape = "polygon" 
    else:  # Check for circle and ellipse based on circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        shape = "circle" if circularity > 0.7 else "ellipse" 
    
    return shape

# Function to convert a path to a contour
def path_to_contour(path):
    contour = []
    for seg in path:
        for point in [seg.start, seg.end]:
            contour.append([int(point.real), int(point.imag)])
    return np.array(contour)

# Function to process SVG and detect shapes
def process_svg(file):
    file_data = file.read()
    
    paths, attributes, svg_attributes = svg2paths2(BytesIO(file_data))
    
    tree = ET.parse(BytesIO(file_data))
    root = tree.getroot()
    
    shape_counts = {
        "triangle": 0,
        "square": 0,
        "rectangle": 0,
        "polygon": 0,
        "circle": 0,
        "ellipse": 0,
        "unidentified": 0
    }
    
    for path in paths:
        contour = path_to_contour(path)
        
        if contour.size == 0:
            continue
        
        # Use a larger image to avoid cropping issues
        img = np.zeros((1000, 1000), dtype=np.uint8)
        cv2.fillPoly(img, [contour], 255)
        
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            shape = detect_shape(contour)
            if shape in shape_counts:
                shape_counts[shape] += 1
            else:
                shape_counts["unidentified"] += 1
    
    return shape_counts

# Function to convert SVG to base64
def svg_to_base64(svg_data):
    encoded = base64.b64encode(svg_data).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"

# Function to fit a circle using least squares
def fit_circle(XY):
    def objective(params):
        x0, y0, r = params
        return np.sqrt((XY[:, 0] - x0) ** 2 + (XY[:, 1] - y0) ** 2) - r

    # Initial guess
    x_mean = np.mean(XY[:, 0])
    y_mean = np.mean(XY[:, 1])
    r_guess = np.mean(np.sqrt((XY[:, 0] - x_mean) ** 2 + (XY[:, 1] - y_mean) ** 2))
    initial_params = [x_mean, y_mean, r_guess]
    
    result = leastsq(objective, initial_params)
    return result[0]

# Function to detect symmetry
def detect_symmetry(image):
    """
    Detects whether a shape is symmetric or asymmetric and draws the symmetry line if symmetric.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        bool: True if the shape is symmetric, False otherwise.
        tuple: Coordinates of the symmetry line if symmetric, otherwise None.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the shape from the background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours of the shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Ignore small contours
        if area < 1000:
            continue

        # Calculate the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the aspect ratio of the contour
        aspect_ratio = float(w) / h

        # Ignore contours that are too long and thin
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            continue

        # Extract the region of interest (ROI) from the thresholded image
        roi = thresh[y:y+h, x:x+w]

        # Flip the ROI horizontally and vertically
        flipped_roi_x = np.fliplr(roi)
        flipped_roi_y = np.flipud(roi)

        # Compare the ROI with its flipped versions
        similarity_x = np.mean(roi == flipped_roi_x)
        similarity_y = np.mean(roi == flipped_roi_y)

        # If the similarity is above a certain threshold, the shape is symmetric
        if similarity_x > 0.9 or similarity_y > 0.9:
            # Calculate the symmetry line
            if similarity_x > 0.9:
                line_x = x + w // 2
                return True, (line_x, y, line_x, y + h)
            else:
                line_y = y + h // 2
                return True, (x, line_y, x + w, line_y)

    # If no symmetric shape is found, return False
    return False, None

# Function to process symmetry detection
def symmetry_detection_page():
    st.header("Symmetry Detection")

    # Upload the image
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Detect symmetry
        symmetric, line_coords = detect_symmetry(image)

        # Draw the symmetry line if the shape is symmetric
        if symmetric:
            cv2.line(image, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), (0, 255, 0), 2)
            st.write("The shape is symmetric.")
        else:
            st.write("The shape is asymmetric.")

        # Display the image
        st.image(image, channels="BGR", caption="Processed Image")

# Function to process occlusion completion
def occlusion_completion_page():
    st.header("Occlusion Completion")

    # Upload the image
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        # Check if the image is loaded correctly
        if image is None:
            st.error("Error: Image not loaded. Check the file path.")
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

            # Display the results
            st.image(final_image, caption="Completed Occlusion Image", channels="GRAY")


# Function to handle regularization page
def regularization_page():
    st.header("Regularization")

    # Upload the SVG file
    uploaded_file = st.file_uploader("Choose an SVG file", type="svg")

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        
        # Convert SVG to base64
        svg_data = uploaded_file.read()
        base64_svg = svg_to_base64(svg_data)
        
        # Display the SVG file as an image
        st.image(base64_svg, use_column_width=True)
        
        # Reset the file pointer after displaying
        uploaded_file.seek(0)
        
        # Process the SVG file to detect shapes
        shape_counts = process_svg(uploaded_file)
        
        if shape_counts:
            st.write(f"Shape Counts: {shape_counts}")
        else:
            st.write("No identifiable shapes found.")

def main():
    st.title("SVG Shape Detection")

    # Page selector at the top
    pages = ["Regularization", "Occlusion Completion", "Symmetry Detection"]
    page = st.selectbox("Choose a Page", pages, index=0)

    # Page content based on the selected menu item
    if page == "Occlusion Completion":
        occlusion_completion_page()
    elif page == "Regularization":
        regularization_page()
    elif page == "Symmetry Detection":
        symmetry_detection_page()

if __name__ == "__main__":
    main()
