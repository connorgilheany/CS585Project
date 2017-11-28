"""

CS 585 Fall '17 Project

Authors: Connor Gilheany
Add your names
"""


import numpy as np
import cv2
import os
import random



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
        rectangles = findRectangles(image)
        plates = detectLicensePlates(rectangles)
        readLicensePlates(plates)
        
        #cv2.imshow('image', image)
        #cv2.waitKey(0)
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
    #might have taken the functional programming a little too far here
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























