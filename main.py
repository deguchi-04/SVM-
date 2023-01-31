import cv2
import numpy as np


f = open("demofile.txt", "a")
imageNumber = 12850
while(imageNumber < 12975):
    imageName = "left" + str(imageNumber).zfill(4)
    print("Nome da imagem" + imageName)
    imgOrigin = cv2.imread("aveleda_2020_07_23_zed_images/" + imageName + ".jpg")

    imgResized = cv2.resize(imgOrigin, (640,480))

    cv2.rectangle(imgResized, (227,10), (414, 150), (0,0,255), 2)
    cv2.rectangle(imgResized, (227,160), (414, 300), (0,0,255), 2)
    cv2.rectangle(imgResized, (227,310), (414, 450), (0,0,255), 2)

    #cv2.imshow("Image Origin", imgOrigin)
    cv2.imshow("Image Resized", imgResized)
    cv2.waitKey(1000)
    print("Leaf density? 0 33 66 100")
    f.write(imageName + " ")
    readValue = input("First rectangle ") 
    f.write(readValue + " ")
    readValue = input("Second rectangle ") 
    f.write(readValue + " ")
    readValue = input("Third rectangle ") 
    f.write(readValue + " \n")

    #cv2.waitKey(1000)

    cv2.destroyAllWindows()

    imageNumber += 5
