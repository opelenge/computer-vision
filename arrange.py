import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
import math
from scipy.spatial import ConvexHull
from scipy.spatial import distance

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
    plt.gca().add_patch(Patches.Polygon(point,closed=True,fill=False))
    # plot polyline
    plt.gca().add_patch(Patches.Polygon(point,closed=True,fill=False))
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
    unique_corners = np.array(unique_corners[:n_corners_range[1]])
    # Compute angle of each corner wrt centroid
    angles = np.arctan2(unique_corners[:, 1] - centroid[1], unique_corners[:, 0] - centroid[0])
    # Sort the corners based on their angle
    sorted_indices = np.argsort(angles)
    return unique_corners[sorted_indices]


def get_side_properties(side):
    # Calculate length ratio, angle, and curvature for a side
    length_ratio = np.linalg.norm(side)
    curvature = np.abs(np.polyfit(side[:,0], side[:,1], 2)[0])
    return length_ratio, curvature

def compare_sides(dist1, dist2):

    if abs(dist1 - dist2) <= 1:
        return True
    else:
        return False

def sort_counterclockwise(points, start):
    # Get the centroid of the polygon
    n = len(points)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid_x = sum(x) / n
    centroid_y = sum(y) / n
    # Sort the points based on their polar angle with respect to the centroid
    sorted_points = sorted(points, key=lambda p: math.atan2(p[1] - centroid_y, p[0] - centroid_x))
    # Check if the starting point is present in the sorted list of points
    start_index = None
    for i, p in enumerate(sorted_points):
        if p[0] == start[0] and p[1] == start[1]:
            start_index = i
            break
    # If the starting point is not present in the sorted list of points, return the sorted list
    if start_index is None:
        return sorted_points
    # Reorder the points so that the starting point is the first element
    sorted_points = sorted_points[start_index:] + sorted_points[:start_index]
    return sorted_points

def split_polygon(corners, all_points):
    result = []
    for i in range(len(corners)):
        start = corners[i]
        end = corners[(i + 1) % len(corners)]
        start_index = -1
        end_index = -1
        for j, point in enumerate(all_points):
            if (start == point).all():
                start_index = j
                break
        for j, point in enumerate(all_points[start_index:]):
            if (end == point).all():
                end_index = j + start_index
                break
        if start_index <= end_index:
            result.append(np.array(all_points[start_index: end_index + 1]))
        else:
            result.append(np.array(all_points[start_index:] + all_points[:end_index + 1]))
    result[-1] = np.concatenate((result[-1], corners[0].reshape(1, -1)))               
    return result

def label(sides, points):
    for p in range(0, len(sides)):
        fig, ax = plt.subplots()
        plt.scatter([s[0] for s in points[p]],[s[1] for s in points[p]], label='data', marker='o')
        for r in range(0, len(sides[p])):
            ax.annotate(str(p) + str(r), (sides[p][r][0][0], sides[p][r][1][1]))
        #plt.savefig(f"{p} verts_sorted.jpg")        
    plt.show()       

def draw_line(p1, p2, num_points=100):
    x1, y1 = p1
    x2, y2 = p2
    line = []
    for i in range(num_points + 1):
        t = i / num_points
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        line.append((x, y))
    return line      
   

def calculate_distance(points):
    first_point = points[0]
    last_point = points[-1]
    distances = []
    x1, y1 = first_point
    x2, y2 = last_point
    slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    intercept = y1 - slope * x1
    for point in points:
        x, y = point
        # Calculate the equation of the perpendicular line
        perp_slope = -1 / slope if slope != 0 else float('inf')
        perp_intercept = y - perp_slope * x
        # Calculate the intersection point of the two lines
        inter_x = (perp_intercept - intercept) / (slope - perp_slope) if slope - perp_slope != 0 else x
        inter_y = slope * inter_x + intercept
        # Calculate the distance between the current point and the intersection point
        distance = ((x - inter_x)**2 + (y - inter_y)**2)**0.5
        distances.append(distance)
    x_points = [x for x, y in points]
    y_points = [y for x, y in points]
    plt.scatter(x_points, y_points)
    # Draw the line segment
    line = draw_line(first_point, last_point)
    x_line = [x for x, y in line]
    y_line = [y for x, y in line]
    plt.plot(x_line, y_line)
    for point, distance in zip(points, distances):
        x, y = point
        # Calculate the equation of the perpendicular line
        perp_slope = -1 / slope if slope != 0 else float('inf')
        perp_intercept = y - perp_slope * x
        # Calculate the intersection point of the two lines
        inter_x = (perp_intercept - intercept) / (slope - perp_slope) if slope - perp_slope != 0 else x
        inter_y = slope * inter_x + intercept
        # Draw the perpendicular line and the distance
        plt.plot([x, inter_x], [y, inter_y], 'k--')
        plt.plot(inter_x, inter_y, 'ro')
        plt.annotate(f"{distance:.2f}", (inter_x, inter_y), textcoords="offset points", xytext=(0,10), ha='center')  
    #plt.show()   
    t = 0    
    for i in distances: 
        t += i
    #print(t)    
    #print(distances) 
    #print(len(distances))  
    return t  

os.chdir("C:/Users/OPE/Documents/Visual Studio 2008/computer vision/torn_papers")
images = "C:/Users/OPE/Documents/Visual Studio 2008/computer vision/torn_papers"
k = len(os.listdir(images))
i = 0
all_sides = []
ptu = []
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
            pixel_value = 5
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
            ptu.append(points)
            corners = find_corners(coords)
            vert_sort = sort_counterclockwise(coords, corners[0])
            sides = split_polygon(corners, vert_sort)
            all_sides.append(sides)
            #for t in sides:
                #calculate_distance(t)

            #print(vert_sort)
            #print(corners)
            #print(sides[3])
            #print(len(sides[1]))
            #draw_graph(points)
            #cv.imshow('download', img_resize)
            i += 1
            cv.waitKey(0) 
            cv.destroyAllWindows()
distance = []
p = 0
while p < len(all_sides):
    distances = {}
    j = 0   
    while j < len(all_sides[p]):
        length = calculate_distance(all_sides[p][j])  
        distances[f"{p}side{j}"] = length
        j += 1          
    p += 1
    distance.append(distances)
#print(distance)

f = 0
while f < len(distance):
    for k, v in distance[f].items():
        x = range(0, len(distance))
        list = []
        for i in x:
            for r, j in distance[i].items():
                if compare_sides(v, j):
                    if r != k and r[0] != k[0]:
                        list.append(r)
                else:
                    continue
        if len(list) != 0:            
            print(f"{list} are potential matches to {k}")  
    f += 1          

label(all_sides, ptu)  