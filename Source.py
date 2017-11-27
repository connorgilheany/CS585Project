"""

CS 585 Fall '17 Project

Authors: Connor Gilheany
Add your names
"""


import numpy as np
import cv2
import random
import os
from matplotlib import pyplot as plt


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
       # image = close(dilate(binary(image)))
        #image = cv2.Canny(image, 100, 175, apertureSize=3) #only finds edges
        #image = close(image)

        #cv2.imshow('binary', image)
        
        rectangles = findRectangles(image)
        
        #labels, areas = labelBinaryImageComponents(image)
        #image = colorComponents(labels)
        #plates = detectLicensePlates(rectangles)
        #readLicensePlates(plates)

        #cv2.imshow('image', image)
        #cv2.imshow('edges', rectangles)
        #cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    labels = np.zeros(image.shape)
    currentLabel = 1
    stack = []

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if labels[x,y] == 0 and image[x,y] != 0:#Make it image[x,y] == 0 to label black components instead of white
                currentLabel += 1
                labels[x,y] = currentLabel
                stack += [(x,y)]
                while len(stack) > 0: #flood fill the component
                    point = stack.pop()
                    neighbors = get8Neighbors(point, image.shape)
                    sameColorNeighbors = list(filter(lambda pt: image[pt[0], pt[1]] == image[point[0], point[1]], neighbors))
                    unmarkedNeighbors = list(filter(lambda pt: labels[pt[0], pt[1]] == 0, sameColorNeighbors))
                    for pt in unmarkedNeighbors:
                        labels[pt[0], pt[1]] = labels[point[0], point[1]]
                        stack.append(pt)
    #labels, areas = removeSmallComponents(labels, 40)
    areas = []
    return labels, areas

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

def binary(image):
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] < 75:
                image[i][j] = 0
            else:
                image[i][j] = 255
    return image

def loadImages():
    names = list(filter(lambda fileName: fileName[0] != ".", os.listdir("images")))
    images = []
    for x in names:
        path = "images/"+x
        images += [cv2.imread(path)] #uncomment for grayscale read
    return images



def findRectangles(colorImage):
    grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)

    """
    Finds all the rectangles (possible license plates) in the image.
    Returns an array of Rectangle objects (defined at top of this file)
    """
    edges = cv2.Canny(grayImage, 100, 175, apertureSize=3) #only finds edges
    lines = cv2.HoughLinesP(edges,1,np.pi/180, threshold=60, minLineLength=20, maxLineGap=10)

    a,b,c = lines.shape
    for i in range(a):
        cv2.line(colorImage, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("og", edges)
    cv2.imshow("test",colorImage)


    """print(len(lines))
    for x in range(0, len(lines)):
        for rho,theta in lines[x]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(colorImage,(x1,y1),(x2,y2),(0,0,255),1)
    cv2.imshow("rects", colorImage)"""
    cv2.waitKey(0)
    return edges

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
    pass

def blur(image):
    return cv2.blur(image, (3,3))

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


"""rect = Rectangle(0,0,cv2.imread("images/plate.jpg"))
print(rect.isLicensePlate())
rect = Rectangle(0,0,cv2.imread("images/car.jpg"))
print(rect.isLicensePlate())"""

main()
























