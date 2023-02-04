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


def find_corners(coords, n_corners_range=(4, 4), threshold=20):
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

def compare_sides(side1, side2, angle_threshold=10, length_ratio_threshold=0.1, curvature_threshold=0.1):
    # Calculate the angle between the two sides
    angle = np.arccos(np.dot(side1, side2) / (np.linalg.norm(side1) * np.linalg.norm(side2)))
    
    # Calculate the length ratio between the two sides
    length_ratio = np.linalg.norm(side1) / np.linalg.norm(side2)
    
    # Calculate the curvature of the two sides
    curvature1 = np.abs(np.polyfit(side1[:,0], side1[:,1], 2)[0])
    curvature2 = np.abs(np.polyfit(side2[:,0], side2[:,1], 2)[0])
    curvature_ratio = curvature1 / curvature2
    
    # Check if the sides match by comparing the angle, length ratio, and curvature
    if (np.abs(angle) < angle_threshold) and (np.abs(length_ratio - 1) < length_ratio_threshold) and (np.abs(curvature_ratio - 1) < curvature_threshold):
        return True
    else:
        return False

def get_side_properties(side):
    # Calculate length ratio, angle, and curvature for a side
    length_ratio = np.linalg.norm(side)
    curvature = np.abs(np.polyfit(side[:,0], side[:,1], 2)[0])
    return length_ratio, curvature

def compare_sides(side1, side2):
    # Compare two sides based on length ratio, angle, and curvature
    length_ratio1, curvature1 = get_side_properties(side1)
    length_ratio2, curvature2 = get_side_properties(side2)
    #angle = np.arccos(np.dot(side1, side2) / (np.linalg.norm(side1) * np.linalg.norm(side2)))
    length_ratio = length_ratio1 / length_ratio2
    curvature_ratio = curvature1 / curvature2

    # Define a threshold for each property to determine a match
    length_ratio_threshold = 0.1
    angle_threshold = 10
    curvature_threshold = 0.1

    if (np.abs(curvature_ratio - 1) < curvature_threshold) and abs(length_ratio1 - length_ratio2) < length_ratio_threshold:
        return True
    else:
        return False

def find_matches(side_list):
    # Find matches in a list of sides
    matches = []
    while len(side_list) > 0:
        current_side = side_list.pop()
        current_matches = []
        for i, side in enumerate(side_list):
            print(current_side)
            #print(side)
            if compare_sides(current_side, side):
                current_matches.append(side)
                side_list.pop(i)
                i -= 1
        if len(current_matches) > 0:
            matches.append((current_side, current_matches))
    print(matches)        
    return matches
    


os.chdir("C:/Users/OPE/Documents/Visual Studio 2008/computer vision/torn_papers")
images = "C:/Users/OPE/Documents/Visual Studio 2008/computer vision/torn_papers"
k = len(os.listdir(images))
i = 0
all_sides = []
while i < k:
    for image in os.listdir(images):
        if image.endswith(".jpg") or image.endswith(".png"):
            img = cv.imread(image)
            img_resize = rescaleFrame(img, scale = .15)

            edge_pixel = []
            piece = []
            background = []
            points = []

            xl = []
            yl = []
            rerr = []
            pixel_value = 100
            for i in range (img_resize.shape[0]):
                for j in range(img_resize.shape[1]):
                    numbers = img_resize[i][j]
                    if any(numbers < pixel_value):
                        background.append(numbers)
                        #numbers[0], numbers[1], numbers[2] = [0, 0, 0]
                    if all(numbers > pixel_value):
                        piece.append(numbers)
                        coordinates = [i, j] 
                        #numbers[0], numbers[1], numbers[2] = [255, 255, 255]  
                        
                        if any(img_resize[i][j - 1] < pixel_value) or any(img_resize[i][j + 1] < pixel_value) or any(img_resize[i - 1][j] < pixel_value) or any(img_resize[i + 1][j] < pixel_value):
                            edge_pixel.append(img_resize[i][j])  
                            pts = np.array([coordinates])
                            points.append(coordinates)

            for coord in points:
                xl.append(coord[0])
                yl.append(coord[1])

            coords = np.array(points)
            coord = find_corners(coords)
            sides = split_polygon(coords, coord)
            all_sides.append(sides)
            print(sides)
            draw_graph(points)
            #cv.imshow('download', img_resize)
            i += 1
            cv.waitKey(0) 
            cv.destroyAllWindows()

p = 0
while p < len(all_sides) and (p + 1) < len(all_sides):
    j = 0
    while j < len(all_sides[p]):
        for k in range(0, len(all_sides[p + 1])):
            if compare_sides(all_sides[p][j], all_sides[p + 1][k]):
                print(True)
                print(f"{p}side{j} matches with {p + 1}side{k} ")
            else:
                print(False)        
        j += 1       
    p += 1
