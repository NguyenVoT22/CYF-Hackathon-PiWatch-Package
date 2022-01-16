import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # Width
cap.set(4,480) # Height
 
while(True):
    ret, image = cap.read()
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(
        grey,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h), (0, 8, 20), 3) 
                 # format -> (img, strtpnt, endpnt, clr, thcknss)
                 # COLOR_BGR2GRAY <- BGR RGB -> COLOR_RGB2GRAY

        roi_gray = grey[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w] 
    
    cv2.imshow('colored', image)
    cv2.imshow('greyscale', grey)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # will closed when you press `ESC`
        break

cap.release()
cv2.destroyAllWindows()