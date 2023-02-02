import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
import math
from scipy.spatial import ConvexHull
from skimage.draw import polygon

#this function resizes the image
def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)    

#this function join the coordinate points extracted from the edge of the image to form a polygon
def draw_graph(point):
    cent=(sum([p[0] for p in point])/len(point),sum([p[1] for p in point])/len(point))
    # sort by polar angle
    point.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
    # plot points
    plt.scatter([p[0] for p in point],[p[1] for p in point], label='data', marker='o')
    # plot polyline
    plt.gca().add_patch(Patches.Polygon(point,closed=False,fill=False))
    #plt.grid()
    plt.show()


def find_corners(coords, n_corners_range=(3, 4), threshold=20):
    hull = ConvexHull(coords)
    corners = hull.points[hull.vertices]
    centroid = np.mean(corners, axis=0)
    distances = np.linalg.norm(corners - centroid, axis=1)
    sorted_indices = np.argsort(distances)[::-1]
    unique_corners = []
    for corner in corners[sorted_indices]:
        if not any([np.linalg.norm(corner - uc) < threshold for uc in unique_corners]):
            unique_corners.append(corner)
        if len(unique_corners) >= n_corners_range[0]:
            break
    return np.array(unique_corners[:n_corners_range[1]])

def split_polygon(coords, corners):
    divided_polygons = []
    for i in range(len(corners)):
        start = corners[i]
        end = corners[(i + 1) % len(corners)]
        line_segment = np.array([start, end])
        divided_polygon = []
        for coord in coords:
            if is_point_on_line(coord, line_segment[0], line_segment[1]):
                divided_polygon.append(coord)
        divided_polygons.append(np.array(divided_polygon))
    return divided_polygons

def is_point_on_line(point, line_start, line_end, epsilon=10):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    if x < min(x1, x2) - epsilon or x > max(x1, x2) + epsilon:
        return False
    if y < min(y1, y2) - epsilon or y > max(y1, y2) + epsilon:
        return False
    if x1 == x2:
        return abs(x - x1) < epsilon
    if y1 == y2:
        return abs(y - y1) < epsilon
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1
    return abs(y - slope * x - y_intercept) < epsilon

images = os.path.join(r"C:\Users\OPE\Documents\Visual Studio 2008\computer vision\thepics.jpg")
img = cv.imread(images)
img_resize = rescaleFrame(img, scale = .15)

edge_pixel = []
piece = []
background = []
points = []

xl = []
yl = []
rerr = []

for i in range (img_resize.shape[0]):
    for j in range(img_resize.shape[1]):
        numbers = img_resize[i][j]
        if any(numbers < 197):
            background.append(numbers)
            #numbers[0], numbers[1], numbers[2] = [0, 0, 0]
        if all(numbers > 197):
            piece.append(numbers)
            coordinates = [i, j] 
            #numbers[0], numbers[1], numbers[2] = [255, 255, 255]  
            
            if any(img_resize[i][j - 1] < 197) or any(img_resize[i][j + 1] < 197) or any(img_resize[i - 1][j] < 197) or any(img_resize[i + 1][j] < 197):
                edge_pixel.append(img_resize[i][j])  
                pts = np.array([coordinates])
                points.append(coordinates)

for coord in points:
    xl.append(coord[0])
    yl.append(coord[1])

coords = np.array(points)
coord = find_corners(coords)
print(split_polygon(coords, coord))


draw_graph(points)
cv.imshow('download', img_resize)
cv.waitKey(0) 
cv.destroyAllWindows()


