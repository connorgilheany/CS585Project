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
        return self.subimage.shape[0]

    def width(self):
        return self.subimage.shape[1]

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

class LicensePlate:
    rectangle = None
    characters = []

    def __init__(self, rectangle, characters):
        self.rectangle = rectangle
        self.characters = characters

    def toString(self):
        orderedCharacters = sorted(self.characters, key=lambda character: character.location.x)
        return ''.join(list(map(lambda character: character.character, orderedCharacters)))

    def filterSmallCharacters(self):
        if len(self.characters) > 1:
            highest = max(self.characters, key=lambda char: char.location.area())
            secondHighest = max(list(filter(lambda x: x.character != highest.character, self.characters)), key=lambda char: char.location.area())
            if highest.location.area() * 0.75 > secondHighest.location.area(): #if theres one character that encompasses multiple, we ignore it
                self.characters = list(filter(lambda char: char.character != highest.character, self.characters))
            self.characters = list(filter(lambda char: char.location.height() > highest.location.height() * 0.7, self.characters))

class CharacterTemplate:
    template = None
    char = ""

    def __init__(self, char, template):
        self.char = char
        self.template = template


def main():
    images = loadImages()
    loadTemplates()
    for image in images:
        rectangles = findRectangles(image)
        plates = detectLicensePlates(rectangles)
        plates = readLicensePlates(plates)
        #characters = filterDuplicateCharacters(orderedCharacters)
        drawCharacters(image, plates)
        cv2.imshow("image", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def filterDuplicateCharacters(characters):
    charsToRemove = []
    for char in characters:
        for char2 in characters:
            if abs(char.location.x - char2.location.x) < 15:
                if char.correlation > char2.correlation:
                    charsToRemove += [char2]
                else:
                    charsToRemove += [char]
    return [x for x in characters if x not in charsToRemove]

def drawCharacters(image, plates):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for plate_index in range(len(plates)):
        plate = plates[plate_index]
        if len(plate.characters) < 3:
            print("skipping short license plate with characters: "+plate.toString())
            continue

        shouldShow = True
        for plate2_index in range(plate_index, len(plates)):
            if plate_index == plate2_index:
                continue #avoid out of bounds exception caused by starting plate2_index loop at plate_index+1
            plate2 = plates[plate2_index]
            if abs(plate.rectangle.x - plate2.rectangle.x) < 30:
                if len(plate2.characters) >= len(plate.characters):
                    shouldShow = False
        if not shouldShow:
            print("skipping duplicate license plate with characters: "+plate.toString())
            continue
        #print(char.location.area())
        print("Found plate: "+plate.toString())
        cv2.putText(image, plate.toString(), (plate.rectangle.x, plate.rectangle.y), font, 1, (0, 0, 0), 4)


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
        rect = Rectangle(x, y, subimage)
        if rect.area() > 500:
            cv2.imshow("contour", subimage)
            cv2.waitKey(0)
        if rect.area() / (image.shape[0] * image.shape[1]) * 100 > 3:
            rectangles += [rect]
    return rectangles


def detectLicensePlates(rectangles):
    """
    Takes an array of Rectangle objects (defined at top of this file)
    Returns an array of Rectangles which might be license plates
    """
    return list(map(lambda z: LicensePlate(z, []), filter(lambda x: x.isLicensePlate(), rectangles)))

def readLicensePlates(plates):
    """
    Takes an array of Rectangle objects (defined at the top of this file) which are probably license plates
    Template match to find the letters on the plate
    """
    characters = []
    for plate in plates: 
        thresholded = cv2.bitwise_not(cv2.adaptiveThreshold(plate.rectangle.subimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 5))
        #binaryPlate = binary(plate.subimage)
        image2, contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #rectangles = []
        for contour in contours:
            contourProps = cv2.boundingRect(contour)
            x = contourProps[0]
            y = contourProps[1]
            width = contourProps[2]
            height = contourProps[3]
            y2 = y + height
            x2 = x + width
            subimage = plate.rectangle.subimage[y:y2, x:x2]
            character = Rectangle(plate.rectangle.x + x, plate.rectangle.y + y, subimage)
            charPercentageOfPlate = character.area() / plate.rectangle.area() * 100
            #if charPercentageOfPlate > 1:
                #cv2.imshow("area: "+str(charPercentageOfPlate), subimage)
                #cv2.waitKey(0)
            if charPercentageOfPlate > 1.5 and character.isCharacter():
                #this might be a character
                binaryImage = cv2.adaptiveThreshold(character.subimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 6)
                char, correlation, location = matchImageToCharacter(binaryImage)
                print("This image best matches "+char+" (correlation="+str(correlation)+")")
                #cv2.imshow(char +" "+str(correlation), binaryImage)
                #cv2.waitKey(0)
                if correlation > .3:
                    characterObj = Character(char, correlation, character)
                    plate.characters += [characterObj]
        print(plate.toString())
        plate.filterSmallCharacters()
    return plates

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
templates = [] 
def loadTemplates():
    global templates
    templates = list(map(lambda name: CharacterTemplate(name[0], cv2.imread("templates/"+name, 0)), filter(lambda name: name[0] != ".", os.listdir("templates/"))))
    

def matchImageToCharacter(image):
    global templates
    best_correlation = -1
    best_correlation_templateObj = None
    best_location = None
    best_scale = 0

    for templateObj in templates:
        template = templateObj.template
        char = templateObj.char
        resized_image = cv2.resize(image, (int(template.shape[1]), int(template.shape[0]))) #make the image the size of the original template
        whitespace_scale = 1.3
        bordered_image = np.full((int(resized_image.shape[0] * whitespace_scale), int(resized_image.shape[1] * whitespace_scale)), 255)
        minY = int(resized_image.shape[0] * (whitespace_scale-1)/2)
        minX = int(resized_image.shape[1] * (whitespace_scale-1)/2)
        for y in range(len(resized_image)):
            for x in range(len(resized_image[y])):
                bordered_image[y + minY][x + minX] = resized_image[y][x]
        bordered_image = bordered_image.astype(np.uint8)
        #cv2.imshow("borded", bordered_image)
        #cv2.waitKey(0)
        for scale_factor in [1.15, 1.075, 1.025, 0.975, 0.9]:# 0.85]:
            template = cv2.resize(template, (int(template.shape[1] * scale_factor), int(template.shape[0]*scale_factor))) #resize the template to the scale factor
            for rotation in [0]:#[5, 0, -5]: #accounts for letters which are slightly turned
                #rotated_template = rotate(template, rotation)#ndimage.rotate(template, rotation)
                correlation, location = computeImageSimilarity(bordered_image, template)
                #cv2.imshow("bordered", bordered_image)
                #cv2.imshow("rotated template", rotated_template)
               # print(char +" " +str(correlation))
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_correlation_templateObj = templateObj
                    best_location = location
                    best_scale = scale_factor
    #DEBUG CODE BELOW - shows how the template match is happening for each character

    template = best_correlation_templateObj.template
    resized_image = cv2.resize(image, (int(template.shape[1]), int(template.shape[0]))) #make the image the size of the original template
    whitespace_scale = 1.3
    bordered_image = np.full((int(resized_image.shape[0] * whitespace_scale), int(resized_image.shape[1] * whitespace_scale)), 255)
    minY = int(resized_image.shape[0] * (whitespace_scale-1)/2)
    minX = int(resized_image.shape[1] * (whitespace_scale-1)/2)
    for y in range(len(resized_image)):
        for x in range(len(resized_image[y])):
            bordered_image[y + minY][x + minX] = resized_image[y][x]
    bordered_image = bordered_image.astype(np.uint8)
    template = cv2.resize(template, (int(template.shape[1] * best_scale), int(template.shape[0]*best_scale))) #resize the template to the scale factor
    cv2.destroyAllWindows()
    print(best_location)
    cv2.imshow("best match image", bordered_image)
    for y in range(len(template)):
        for x in range(len(template[y])):
            if template[y][x] != 255:
                bordered_image[y + best_location[1]][x+best_location[0]] = template[y][x]
    cv2.imshow("Overlay "+str(best_correlation), bordered_image)
    cv2.imshow("bestpl8 scl:"+str(best_scale), template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [best_correlation_templateObj.char, best_correlation, best_location]

def computeImageSimilarity(image1, image2, method=cv2.TM_CCOEFF_NORMED):

    match = cv2.matchTemplate(image1, image2, method)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(match)
    return maxVal, maxLoc

def rotate(image, degrees):
    rows,cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1)
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst


main()




















