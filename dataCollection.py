import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap=cv2.VideoCapture(0)
detector= HandDetector(maxHands=1) # так как захват только для одной руки
offset=20 #для того чтобы не вплотную к последней точке было
imgSize = 300
folder="Data/letter26"
counter=0
while True:
    success,img=cap.read()
    hands,img = detector.findHands(img)
    if hands:
        hand=hands[0]# распознаем только 1 руку
        x,y,w,h=hand['bbox']
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]# стартовая высота:конечная высота,стартовая ширина:конечная ширина
        imgCropShape=imgCrop.shape

        aspectRatio=h/w #находим соотношение сторон
        if aspectRatio>1:
            temp=imgSize/h
            wCalcutation=math.ceil(temp*w)
            imgResize=cv2.resize(imgCrop,(wCalcutation,imgSize))
            imgResizeShape=imgResize.shape
            if imgResize.shape[0] > 0 and imgResize.shape[1] > 0:
                wGap=math.ceil((imgSize-wCalcutation)/2)
                imgWhite[0:imgResizeShape[0], wGap:wCalcutation+wGap] = imgResize
        else:
            temp = imgSize / w
            hCalcutation = math.ceil(temp * h)
            if hCalcutation > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCalcutation))
                imgResizeShape = imgResize.shape
            if imgResize.shape[0] > 0 and imgResize.shape[1] > 0:
                hGap = math.ceil((imgSize - hCalcutation) / 2)
                imgWhite[hGap:hCalcutation+hGap,0:imgResizeShape[1]] = imgResize
        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
    cv2.imshow("Image",img) #работа с вебкой
    key=cv2.waitKey(1)
    if key==ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)