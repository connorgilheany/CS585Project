
import cv2
import os


#template_img = list(map(lambda name: cv2.imread("templates/"+name, 0), filter(lambda name: name[0] != ".", os.listdir("templates/"))))
for templateFileString in os.listdir("templates/"):
    if templateFileString[0] == ".":
        continue
    print(templateFileString)
    template = cv2.imread("templates/"+templateFileString, 0)
    thresholded = cv2.adaptiveThreshold(template, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43, 2)
    image2, contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        print("Skipping "+templateFileString)
        #continue

    contour = contours[1]
    contourProps = cv2.boundingRect(contour)
    x = contourProps[0]
    y = contourProps[1]
    width = contourProps[2]
    height = contourProps[3]
    y2 = y + height
    x2 = x + width
    subimage = template[y:y2, x:x2]
    cv2.imwrite("templates/"+templateFileString, subimage)