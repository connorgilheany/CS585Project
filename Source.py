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
from collections import defaultdict


class Rectangle:
    #The top left point
    x = 0
    y = 0
    subimage = None #the array of pixels in the rectangle
    correlation = 0

    def __init__(self, x, y, subimage, correlation = 0):
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
        ratio = self.height() * 1.0 / self.width()
        return ratio > 1 and ratio < 7

    def height(self):
        return len(self.subimage)

    def width(self):
        return len(self.subimage[0])

    def area(self):
        return self.height() * self.width() * 1.0

class Character:
    character = ""
    correlation = 0
    location = None

    def __init__(self, character, correlation, location):
        self.character = character
        self.correlation = correlation
        self.location = location


def main():
    images = loadImages()
    loadTemplates()
    for image in images:

        rectangles = findRectangles(image)
        plates = detectLicensePlates(rectangles)
        orderedCharacters = readLicensePlates(plates)
        print(orderedCharacters)
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
    thresholded = cv2.dilate(cv2.erode(cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3), kernel), kernel2)
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
        cv2.imshow("contour", thresholded)
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
            if character.area() / plate.area() * 100 > 4 and character.isCharacter():
                rectangles += [character]

        
        characters = []
        for potentialCharacter in rectangles:
            #potentialCharacter.subimage = cv2.blur(potentialCharacter.subimage, (3,3))
            #cv2.imshow("blurred", potentialCharacter.subimage)
            binaryImage = cv2.adaptiveThreshold(potentialCharacter.subimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
            #kernel = np.ones((2,2), np.uint8)
            #binaryImage = (erode(dilate(binaryImage)))

            #binaryImage = cv2.adaptiveThreshold(potentialCharacter.subimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43, 2)
            char, correlation, location = matchImageToCharacter(binaryImage)
            print("This image best matches "+char+" (correlation="+str(correlation)+")")
            cv2.imshow(char +" "+str(correlation), binaryImage)
            cv2.waitKey(0)
            
            if correlation > .005:
                character = Character(char, correlation, potentialCharacter)
                characters += character

        orderedCharacters = sorted(characters, key=lambda character: character.location.x)
      #  for i in orderedCharacters - 1:
       #     if abs(i.location.x = [i + 1].location.x) < 10


        return orderedCharacters

            #cv2.imshow("character", rectangle.subimage)
            #cv2.waitKey(0)

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

def dilate(image, kernelSize=2):
    kernel = np.ones((kernelSize,kernelSize), np.uint8)
    return cv2.dilate(image, kernel)

def erode(image, kernelSize=2):
    kernel = np.ones((kernelSize,kernelSize), np.uint8)
    return cv2.erode(image, kernel)


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
    global methods_dict
    s = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    best_correlation = -1
    best_correlation_char = ""
    best_location = None
    for char in s:
        template = template_dict[char]
        resized_image = cv2.resize(image, (int(template.shape[1]), int(template.shape[0]))) #make the image the size of the original template
        bordered_image = np.full((int(resized_image.shape[0] * 1.2), int(resized_image.shape[1] * 1.2)), 255)
        minY = int(resized_image.shape[0] * 0.1)
        minX = int(resized_image.shape[1] * 0.1)
        for y in range(len(resized_image)):
            for x in range(len(resized_image[y])):
                bordered_image[y + minY][x + minX] = resized_image[y][x]
        bordered_image = bordered_image.astype(np.uint8)
        #cv2.imshow("borded", bordered_image)
        #cv2.waitKey(0)
        for scale_factor in [1, 0.95, 0.9, 0.85]:
            template = cv2.resize(template, (int(template.shape[1] * scale_factor), int(template.shape[0]*scale_factor))) #resize the template to the scale factor
            for rotation in [5, 0, -5]: #accounts for letters which are slightly turned
                rotated_template = rotate(template, rotation)#ndimage.rotate(template, rotation)
                correlation, location = computeImageSimilarity(bordered_image, rotated_template)
                print(char +" " +str(correlation))
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_correlation_char = char
                    best_location = location

    return [best_correlation_char, best_correlation, best_location]

def computeImageSimilarity(image1, image2, method=cv2.TM_CCOEFF_NORMED):
    match = cv2.matchTemplate(image1, image2, method)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(match)
    return maxVal, maxLoc

def rotate(image, degrees):
    rows,cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1)
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst

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





















