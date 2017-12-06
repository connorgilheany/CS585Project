"""

CS 585 Fall '17 Project

Authors: 
Connor Gilheany
Alana King
Kayce McCue
Caroline Montague
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
        if self.area() < 1000:
            return False
        return abs(self.width() * 1.0 / self.height() - 2) < 0.3

    def isCharacter(self):
        if self.area() < 100:
            return False
        return self.height() > self.width() #abs(self.height() * 1.0 / self.width() - 2) < 1

    def height(self):
        return len(self.subimage)

    def width(self):
        return len(self.subimage[0])

    def area(self):
        return self.height() * self.width()

class CharacterTemplate(Rectangle):
    character = ""
    template = None

    def __init__(self, character, template):
        self.character = character
        self.template = template


def main():
    images = loadImages()
    loadTemplates()
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
        thresholded = cv2.adaptiveThreshold(plate.subimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
        #binaryPlate = binary(plate.subimage)
        cv2.imshow("image",thresholded)
        cv2.waitKey(0)
        image2, contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for contour in contours:
            contourProps = cv2.boundingRect(contour)
            x = contourProps[0]
            y = contourProps[1]
            width = contourProps[2]
            height = contourProps[3]
            y2 = y + height
            x2 = x + width
            subimage = plate.subimage[y:y2, x:x2]
            character = Rectangle(x, y, subimage)
            if character.area() / plate.area() * 100 > 2.5 and character.isCharacter():
                rectangles += [character]


        for potentialCharacter in rectangles:
            binaryImage = cv2.adaptiveThreshold(potentialCharacter.subimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            #binaryImage = cv2.adaptiveThreshold(potentialCharacter.subimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43, 2)
            char, correlation = matchImageToCharacter(binaryImage)
            print("This image best matches "+char+" (correlation="+str(correlation)+")")
            cv2.imshow("character", binaryImage)
            cv2.waitKey(0)
            #cv2.imshow("character", rectangle.subimage)
            #cv2.waitKey(0)



        """
        TODO:
        Find a rectangle for each object, resize it to the character template size, then template match
        """



        """for x in range(1, numberOfObjects):
            rect = getRectangleForObject(plate.subimage, labels, x)
            if rect is None:
                continue
            print("x="+str(rect.x)+" y="+str(rect.y)+" subimage shape="+str(rect.subimage.shape))"""


        #cv2.imshow('image', colorComponents(labels))
        #cv2.waitKey(0)
        #take all those objects

def getRectangleForObject(image, labels, object):
    minX = 100000000000
    minY = 100000000000
    maxX = 0
    maxY = 0
    for y in range(len(labels)):
        for x in range(len(labels[y])):
            if labels[y][x] == object:
                print("Found x="+str(x)+", y="+str(y)+" object="+str(object))
                if y < minY:
                    minY = y
                if y > maxY:
                    maxY = y
                if x < minX:
                    minX = x
                if x > maxX:
                    maxX = x
    if minX == 100000000000 or minY == 100000000000 or maxX == 0 or maxY == 0:
        return None
    return Rectangle(minX, minY, labels[minY:maxY][minX:maxX])

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


#template dictionary
template_dict = {} 
def loadTemplates():
    global template_dict
    template_img = list(map(lambda name: cv2.imread("templates/"+name, 0), filter(lambda name: name[0] != ".", os.listdir("templates/"))))
    s = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(len(template_img)):
        letter = template_img[i]
        template_dict[s[i]] = letter
    return template_dict

def matchImageToCharacter(image):
    global template_dict
    s = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    best_correlation = -1
    best_correlation_char = ""
    for char in s:
        template = template_dict[char]
        resized_image = cv2.resize(image, (template.shape[1], template.shape[0]))
        correlation = computeImageSimilarity(resized_image, template)
        if correlation > best_correlation:
            best_correlation = correlation
            best_correlation_char = char
    return [best_correlation_char, best_correlation]

def computeImageSimilarity(image1, image2):
    return cv2.matchTemplate(image1, image2, cv2.TM_CCORR_NORMED)



"""
rect = Rectangle(0,0,cv2.imread("images/plate.jpg"))
print(rect.isLicensePlate())
rect = Rectangle(0,0,cv2.imread("images/car.jpg"))
print(rect.isLicensePlate())"""
main()
"""
loadTemplates()
charToMatch = "0"
image = template_dict[charToMatch]
char, correlation = matchImageToCharacter(image)
print("The character "+charToMatch+" best matches with "+char+" (correlation="+str(correlation)+")")
"""





















