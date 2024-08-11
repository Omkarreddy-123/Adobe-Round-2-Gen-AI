#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy matplotlib svgwrite cairosvg


# In[2]:


pip install pycairo


# In[3]:


pip install cairosvg


# In[ ]:





# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import os
from scipy.optimize import curve_fit

# Function to read CSV files
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Function to visualize shapes
def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 10))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            if not np.isnan(XY).any():
                ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

# Function to convert matplotlib color codes to valid SVG color codes
def mpl_to_svg_color(mpl_color):
    color_map = {
        'b': 'blue',
        'g': 'green',
        'r': 'red',
        'c': 'cyan',
        'm': 'magenta',
        'y': 'yellow',
        'k': 'black'
    }
    return color_map.get(mpl_color, mpl_color)

# Function to convert polylines to SVG
def polylines2svg(paths_XYs, svg_path):
    # Determine the bounds of the drawing
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            min_x = min(min_x, np.min(XY[:, 0]))
            max_x = max(max_x, np.max(XY[:, 0]))
            min_y = min(min_y, np.min(XY[:, 1]))
            max_y = max(max_y, np.max(XY[:, 1]))

    # Add padding
    padding = 0.2
    width = (max_x - min_x) * (0.6 + padding)
    height = (max_y - min_y) * (0.2 + padding)
    
    # Set fixed dimensions for the SVG output
    fixed_width = 800  # Desired width of the SVG
    fixed_height = 600  # Desired height of the SVG
    
    # Calculate scaling factors
    scale_x = fixed_width / width
    scale_y = fixed_height / height
    scale = min(scale_x, scale_y)  # Uniform scaling to fit the SVG within fixed dimensions

    # Adjust the width and height according to the scale factor
    width *= scale
    height *= scale
    
    # Center the SVG content
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    viewBox_min_x = center_x - (width / 2)
    viewBox_min_y = center_y - (height / 2)
    
    # Create the SVG drawing with fixed dimensions and a centered viewBox
    viewBox = f"{viewBox_min_x} {viewBox_min_y} {width} {height}"
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges', size=(fixed_width, fixed_height), viewBox=viewBox)
    group = dwg.g()

    # Define colors
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, path in enumerate(paths_XYs):
        c = mpl_to_svg_color(colours[i % len(colours)])
        for XY in path:
            if not np.isnan(XY).any():
                # Translate coordinates to fit within the viewBox
                translated_XY = XY - [viewBox_min_x, viewBox_min_y]
                path_data = [f"M {translated_XY[0, 0]},{translated_XY[0, 1]}"]
                path_data += [f"L {x},{y}" for x, y in translated_XY[1:]]
                if np.allclose(translated_XY[0], translated_XY[-1]):
                    path_data.append("Z")
                group.add(dwg.path(d=" ".join(path_data), fill='none', stroke=c, stroke_width=3))
    
    # Add the group to the drawing and save
    dwg.add(group)
    dwg.save()




# Function to fit a line
def fit_line(XY):
    x, y = XY[:, 0], XY[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

# Function to fit a circle
def fit_circle(XY):
    x, y = XY[:, 0], XY[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    def calc_R(xc, yc):
        return np.sqrt((x - xc)*2 + (y - yc)*2)
    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    center_estimate = x_m, y_m
    center, _ = curve_fit(f, center_estimate)
    xc, yc = center
    R = calc_R(xc, yc).mean()
    return xc, yc, R

# Function to check if the points form a line
def is_line(XY, tol=1e-2):
    if len(XY) < 2:
        return False

    x, y = XY[:, 0], XY[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    y_fit = m * x + c
    residuals = np.abs(y - y_fit)
    return np.all(residuals < tol)

# Function to check if the points form a circle
def is_circle(XY, tol=1e-2):
    try:
        xc, yc, R = fit_circle(XY)
        residuals = np.abs(np.sqrt((XY[:, 0] - xc)*2 + (XY[:, 1] - yc)*2) - R)
        return np.all(residuals < tol)
    except:
        return False

# Function to fit a rectangle (heuristic approach)
def fit_rectangle(XY):
    # Assume rectangle vertices are given in order
    return XY

# Function to check symmetry
def check_symmetry(XY):
    from sklearn.decomposition import PCA
    XY_centered = XY - np.mean(XY, axis=0)
    pca = PCA(n_components=2)
    pca.fit(XY_centered)
    principal_axes = pca.components_

    def is_symmetric(axis):
        axis_normal = np.array([-axis[1], axis[0]])
        projections = np.dot(XY_centered, axis_normal)
        symmetric_projections = np.dot(XY_centered, -axis_normal)
        return np.allclose(projections, symmetric_projections, atol=1e-2)

    symmetries = [is_symmetric(principal_axes[0]), is_symmetric(principal_axes[1])]
    return symmetries

# Function to regularize paths by fitting shapes
def regularize_paths(paths_XYs):
    regularized_paths = []
    for XYs in paths_XYs:
        for XY in XYs:
            if len(XY) > 1:
                if is_line(XY):
                    m, c = fit_line(XY)
                    x_min, x_max = np.min(XY[:, 0]), np.max(XY[:, 0])
                    fitted_XY = np.array([[x_val, m * x_val + c] for x_val in np.linspace(x_min, x_max, num=len(XY))])
                    regularized_paths.append([fitted_XY])
                elif is_circle(XY):
                    xc, yc, R = fit_circle(XY)
                    theta = np.linspace(0, 2 * np.pi, 100)
                    fitted_XY = np.array([[xc + R * np.cos(t), yc + R * np.sin(t)] for t in theta])
                    regularized_paths.append([fitted_XY])
                elif len(XY) == 4:  # Placeholder for rectangles
                    fitted_XY = fit_rectangle(XY)
                    regularized_paths.append([fitted_XY])
                else:
                    symmetries = check_symmetry(XY)
                    if any(symmetries):
                        print(f"Symmetry detected: {symmetries}")
                    regularized_paths.append([XY])
            else:
                regularized_paths.append([XY])
    return regularized_paths

# Main function to process input data and produce expected results
def main():
    csv_path = "C:\\Users\\ramar\\Downloads\\frag2_sol.csv"
    paths_XYs = read_csv(csv_path)
    
    regularized_paths = regularize_paths(paths_XYs)

    plot(regularized_paths)
    
    output_dir = "C:\\Users\\ramar\\OneDrive\\Pictures\\adobe"
    svg_path = os.path.join(output_dir, 'output.svg')
    polylines2svg(regularized_paths, svg_path)
    print(f"SVG file has been saved to {svg_path}.")

if __name__ == "__main__":
    main()


# In[ ]:


import ctypes.util
print(ctypes.util.find_library('cairo'))


# In[ ]:




