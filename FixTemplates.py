
import cv2
import os
import numpy as np


#template_img = list(map(lambda name: cv2.imread("templates/"+name, 0), filter(lambda name: name[0] != ".", os.listdir("templates/"))))


six = cv2.imread("templates/6.png")
print(six.shape)
"""six2 = cv2.imread("templates/62.png", 0)
six2 = cv2.adaptiveThreshold(six2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 2)
six2 = cv2.resize(six2, (six.shape[1], six.shape[0]))
cv2.imwrite("templates/62.png", six2)"""


"""
for templateFileString in os.listdir("templates/"):
    if templateFileString[0] == ".":
        continue
    template = cv2.imread("templates/"+templateFileString,0)
    print(templateFileString)
    bordered_image = np.full((int(template.shape[0]*1.05), int(template.shape[1] * 1.15)), 255)
    minY = int(template.shape[0] * 0.025)
    minX = int(template.shape[1] * 0.075)
    for y in range(len(template)):
        for x in range(len(template[y])):
            bordered_image[y + minY][x + minX] = template[y][x]
    bordered_image = bordered_image.astype(np.uint8)"""
    
"""
    thresholded = cv2.adaptiveThreshold(template, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43, 2)
    image2, contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[1]
    contourProps = cv2.boundingRect(contour)
    x = contourProps[0]
    y = contourProps[1]
    width = contourProps[2]
    height = contourProps[3]
    y2 = y + height
    x2 = x + width
    subimage = template[y:y2, x:x2]
    subimage = cv2.resize(subimage, (subimage.shape[1] * 4, subimage.shape[0] * 4))
    cv2.imwrite("templates/"+templateFileString, subimage)"""
    #cv2.imwrite("templates/"+templateFileString, bordered_image)