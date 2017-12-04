"""

CS 585 Fall '17 Project

Authors: Connor Gilheany
Add your names
"""


import numpy as np
import cv2
import os
import random

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


class Rectangle:
    #The top left point
    firstX = 0
    firstY = 0
    subimage = None #the array of pixels in the rectangle

    def __init__(self, firstX, firstY, subimage):
        self.firstX = firstX
        self.firstY = firstY
        self.subimage = subimage


    def isLicensePlate(self):
        """
        Detect which rectangles are license plates based on their proportions (TODO figure out what other criteria to check)
        """
        height = len(self.subimage)
        width = len(self.subimage[0])
        return abs(width * 1.0 / height - 2) < 0.2


def main():
    images = loadImages()

    for image in images:
        labels, areas = labelBinaryImageComponents(image)
        image = colorComponents(labels)
        #rectangles = findRectangles(image)
        #plates = detectLicensePlates(rectangles)
        #readLicensePlates(plates)
        
        cv2.imshow('image', image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def binary(image):
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] < 230:
                image[i][j] = 0
            else:
                image[i][j] = 255
    return image

def loadImages():
    return list(map(lambda name: cv2.imread("images/"+name), filter(lambda name: name[0] != ".", os.listdir("images/"))))


def findRectangles(image):
    """
    Finds all the rectangles (possible license plates) in the image.
    Returns an array of Rectangle objects (defined at top of this file)
    """
    pass

def detectLicensePlates(rectangles):
    """
    Takes an array of Rectangle objects (defined at top of this file)
    Returns an array of Rectangles which might be license plates
    """
    return list(filter(lambda x: x.isLicensePlate(), rectangles))

def readLicensePlates(plates):
    """
    Takes an array of Rectangle objects (defined at the top of this file) which are probably license plates
    Template match to find the letters on the plate
    """

"""
Parameter components: a 2x2 matrix of component labels
Returns colored: A 2x2x3 color image
"""
def colorComponents(components):
    colored = np.zeros((components.shape[0], components.shape[1], 3), np.uint8)
    #coloredComponents = map(getColorForComponent, components)
    for x in range(len(components)):
        for y in range(len(components[x])):
            colored[x][y] = getColorForComponent(components[x][y])
    return colored

"""
Parameter image: A 2x2 grayscale image
Returns:
    labels: a 2x2 matrix of component IDs
    areas: a dictionary {componentID: total area}
"""
def labelBinaryImageComponents(image):
    # image = image * -1
    labels = np.zeros((image.shape[0], image.shape[1], 1))
    currentLabel = 1
    stack = []

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if labels[x][y] == 0: #and image[x,y] == 0:#Make it image[x,y] == 0 to label black components instead of white
                currentLabel += 1
                labels[x,y] = currentLabel
                stack += [(x,y)]
                while len(stack) > 0: #flood fill the component
                    point = stack.pop()
                    neighbors = get8Neighbors(point, image.shape)
                    sameColorNeighbors = list(filter(lambda pt: areColorsSimilar(image[pt[0], pt[1]], image[point[0], point[1]]), neighbors))
                    unmarkedNeighbors = list(filter(lambda pt: labels[pt[0], pt[1]] == 0, sameColorNeighbors))
                    for pt in unmarkedNeighbors:
                        labels[pt[0], pt[1]] = labels[point[0], point[1]]
                        stack.append(pt)
    #labels, areas = removeSmallComponents(labels, 40)
    areas = []
    return labels, areas

def areColorsSimilar(color1, color2):
    color1_rgb = sRGBColor(color1[0], color1[1], color1[2]);
    color2_rgb = sRGBColor(color2[0], color2[1], color2[2]);

    # Convert from RGB to Lab Color Space
    color1_lab = convert_color(color1_rgb, LabColor);
    color2_lab = convert_color(color2_rgb, LabColor);

    # Find the color difference
    delta_e = delta_e_cie2000(color1_lab, color2_lab);

    #print("The difference between the 2 color = " + str(delta_e))
    THRESHOLD = 15
    return delta_e < THRESHOLD 
    #TODO: implement similar color algorithm, then check and see how the connected component labeling algorithm does on a colored image

def dist3(p1, p2):
    return ((int(p2[0]) - int(p1[0])) ** 2 + (int(p2[1]) - int(p1[1])) ** 2 + (int(p2[2]) - int(p1[2])) ** 2) ** 0.5

colors = {0: [0,0,0]}
def getColorForComponent(index):
    if index not in colors:
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        colors[index] = [b,g,r]
    return colors[index]

def get8Neighbors(point, dimensions):
    x = point[0]
    y = point[1]
    maxX = dimensions[0]
    maxY = dimensions[1]
    neighbors = [(x-1, y-1), (x, y-1), (x+1, y-1), 
                 (x-1, y),             (x+1, y),
                 (x-1, y+1), (x, y+1), (x+1, y+1) ]
    return list(filter(lambda point: point[0] >= 0 and point[0] < maxX and point[1] >= 0 and point[1] < maxY, neighbors))

def get4Neighbors(point, dimensions):
    x = point[0]
    y = point[1]
    maxX = dimensions[1]
    maxY = dimensions[0]
    neighbors = [   (x-1, y),       (x, y-1),  #order changed so we can  loop clockwise starting from the west
                             (x+1, y),
                          (x, y+1),]
    return list(filter(lambda point: point[0] >= 0 and point[0] < maxX and point[1] >= 0 and point[1] < maxY, neighbors))

"""
Parameters:
    labels: a 2x2 matrix of component IDs
    minPixels: the smallest area a component should have
Returns:
    labels: a 2x2 matrix of component IDs where component area > minPixels
    areas: a dictionary {componentID: total area}
"""
def removeSmallComponents(labels, minPixels):
    areas = {}
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            label = labels[i,j]
            if label in areas:
                areas[label] = areas[label] + 1
            else:
                areas[label] = 1
    labelsToRemove = []
    for x in areas:
        if areas[x] < minPixels:
            labelsToRemove += [x]
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i,j] in labelsToRemove:
                labels[i,j] = 0
    return labels, areas


def open(image):
    return dilate(erode(image))

def close(image):
    return erode(dilate(image))

def dilate(image):
    newImage = np.zeros(image.shape)
    for x in range(image.shape[0]-1):
        for y in range(image.shape[1]-1):
            if image[x][y] > 0:
                newImage[x][y] = 1
                newImage[x+1][y] = 1
                newImage[x][y+1] = 1
                newImage[x+1][y+1] = 1
    return newImage

def erode(image):
    newImage = np.zeros(image.shape)
    for x in range(image.shape[0]-1):
        for y in range(image.shape[1]-1):
            if image[x][y] > 0 and image[x+1][y] > 0 and image[x][y+1] > 0 and image[x+1][y+1] > 0:

                newImage[x][y] = 1
                newImage[x+1][y] = 0
                newImage[x][y+1] = 0
                newImage[x+1][y+1] = 0
    return newImage

"""
rect = Rectangle(0,0,cv2.imread("images/plate.jpg"))
print(rect.isLicensePlate())
rect = Rectangle(0,0,cv2.imread("images/car.jpg"))
print(rect.isLicensePlate())"""
main()























