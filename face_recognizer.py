import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # width
cam.set(4, 480) # height

face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n Please enter a user ID. Start w/ 0 for first user, then run again w/ 1 & 2 ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h), (0, 8, 20), 3) 
                 # format -> (img, strtpnt, endpnt, clr, thcknss)
                 # COLOR_BGR2GRAY <- BGR RGB -> COLOR_RGB2GRAY   
        count += 1

        # Images will be saved in the datasets folder.
        cv2.imwrite("faces_dataset/AuthorizedUsr." + str(face_id) + '.' + str(count) + ".jpg", grey[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 90: # Will be taking 90 face samples and stop video
         break

# Do a bit of cleanup
print("\n Exiting program..freeing up space...")
cam.release()
cv2.destroyAllWindows()