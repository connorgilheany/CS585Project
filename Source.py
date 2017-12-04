"""

CS 585 Fall '17 Project

Authors: 
Connor Gilheany
Add your names
"""


import numpy as np
import cv2
import os
import random



class Rectangle:
    #The top left point
    x = 0
    y = 0
    subimage = None #the array of pixels in the rectangle

    def __init__(self, x, y, subimage):
        self.x = x
        self.y = y
        self.subimage = subimage


    def isLicensePlate(self):
        """
        Detect which rectangles are license plates based on their proportions (TODO figure out what other criteria to check)
        """
        height = len(self.subimage)
        width = len(self.subimage[0])
        if width * height < 1000:
            return False
        return abs(width * 1.0 / height - 2) < 0.3

#class Character(Rectangle):


def main():
    images = loadImages()
    for image in images:

        rectangles = findRectangles(image)
        plates = detectLicensePlates(rectangles)
        readLicensePlates(plates)
    cv2.destroyAllWindows()

def binary(image):
    newImage = np.zeros(image.shape)
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] < 230:
                newImage[i][j] = 0
            else:
                newImage[i][j] = 255
    return newImage

def loadImages():
    #might have taken the functional programming a little too far here
    return list(map(lambda name: cv2.imread("images/"+name, 0), filter(lambda name: name[0] != ".", os.listdir("images/"))))


def findRectangles(image):
    """
    Finds all the rectangles (possible license plates) in the image.
    Returns an array of Rectangle objects (defined at top of this file)
    """
    rectangles = []
    blurred = cv2.medianBlur(image, 5)
    kernel = np.ones((3,3), np.uint8)
    kernel2 = np.ones((2,2), np.uint8)
    thresholded = cv2.dilate(cv2.erode(cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2), kernel), kernel2)
    image2, contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours: 
        contourProps = cv2.boundingRect(contour)
        x = contourProps[0]
        y = contourProps[1]
        width = contourProps[2]
        height = contourProps[3]
        y2 = y + height
        x2 = x + width
        subimage = image[y:y2, x:x2]
        rectangles += [Rectangle(x, y, subimage)]
    return rectangles


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
    for plate in plates: 
        thresholded = dilate(cv2.adaptiveThreshold(plate.subimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43, 2))
        #binaryPlate = binary(plate.subimage)
        labels, _ = labelBinaryImageComponents(thresholded)
        """
        TODO:
        Find a rectangle for each object, resize it to the character template size, then template match
        
        """
        cv2.imshow('image', colorComponents(labels))
        cv2.waitKey(0)
        #take all those objects

def labelBinaryImageComponents(image):
    # image = image * -1
    labels = np.zeros(image.shape)
    currentLabel = 1
    stack = []

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if labels[x,y] == 0 and image[x,y] == 0:
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
    labels, areas = removeSmallComponents(labels, 40)
    return labels, areas

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

def get8Neighbors(point, dimensions):
    x = point[0]
    y = point[1]
    maxX = dimensions[0]
    maxY = dimensions[1]
    neighbors = [(x-1, y-1), (x, y-1), (x+1, y-1), 
                 (x-1, y),             (x+1, y),
                 (x-1, y+1), (x, y+1), (x+1, y+1) ]
    return list(filter(lambda point: point[0] >= 0 and point[0] < maxX and point[1] >= 0 and point[1] < maxY, neighbors))

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

colors = {0: [0,0,0]}
def getColorForComponent(index):
    if index not in colors:
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        colors[index] = [b,g,r]
    return colors[index]

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























