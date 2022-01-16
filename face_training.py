import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'faces_dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):
    
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # converts it to greyscale
        img_numpy = np.array(PIL_img,'uint8')
        
        #grab the user id for each image
        id = int(os.path.split(imagePath)[-1].split(".")[1]) 
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("\n We are currently training the faces ... Just a few seconds. Please wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Print the numer of faces trained and end program
print("\n We have sucessfully trained {0} face(s). Now exiting the program...".format(len(np.unique(ids))))