import cv2
import numpy as np
import os 
import smtplib
from email.message import EmailMessage
# from sense_hat import SenseHat
from sense_emu import SenseHat
import datetime


# color pallette from coolors
#  https://coolors.co/000814-001d3d-003566-ffc300-ffd60a-e03616

rb = (0, 8, 20) # rich black
ob = (0, 29, 61) # oxford blue
pb = (0, 53, 102) # prussian blue
e = (0, 0, 0) #empty
my = (255, 195, 0) # mikadao yellow
gwg = (255, 214, 10) # gold web golden
v = (224, 54, 22) # vermilion

sense = SenseHat()

sapphire_hrt = [
e, e, e, e, e, e, e, e,
e, pb, pb, e, pb, pb, e, e,
pb, pb, pb, pb, pb, pb, pb, e,
pb, pb, pb, pb, pb, pb, pb, e,
pb, pb, pb, pb, pb, pb, pb, e,
e, pb, pb, pb, pb, pb, e, e,
e, e, pb, pb, pb, e, e, e,
e, e, e, pb, e, e, e, e
]

citrine_hrt = [
e, e, e, e, e, e, e, e,
e, my, my, e, my, my, e, e,
my, my, my, my, my, my, my, e,
my, my, my, my, my, my, my, e,
my, my, my, my, my, my, my, e,
e, my, my, my, my, my, e, e,
e, e, my, my, my, e, e, e,
e, e, e, my, e, e, e, e
]

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_TRIPLEX

#initiate id counter
id = 0

# Update these names. Make sure to add as many as IDs trained; assumes first ID is 0.
names = ['Owner', 'Owner Spouse', 'Owner Relative'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # Width
cam.set(4, 480) # Height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img =cam.read()
    grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        grey,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (0, 8, 20), 3)         
        id, confidence = recognizer.predict(grey[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            dateAll = datetime.datetime.now()
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            # Email saying yesh this is the owner or their fam's member
            #build the email with the EmailMessage() object.
            msg = EmailMessage()
                
            msg['Subject'] = "Owner and their trustees"
            msg['From'] = "otheremail.locked@gmail.com"
            msg['To'] = "otheremail.locked@gmail.com" # replace with your company email

            #set the body with .set_content()
            msg.set_content("Hello [owner's name]!\n You are receiving this email because we would like you to know this was either  or your family who passed by.")

            #use smptlib to send using gmail server, on port 465. 
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)

            server.login("otheremail.locked@gmail.com", "Hackathons123") #tmp pwd
                
            #sends email   
            server.send_message(msg)
                    
            print("sending positive msg")
                    
            server.quit()   

            with open('trusted.txt', 'a+') as f:
                f.write("To let you know there was someone at your place that you trust at ")
                f.write(str(dateAll) + "\n")
                # f.write("\n")

            sense.set_pixels(sapphire_hrt)

        else:
            dateAll = datetime.datetime.now()
            id = "unknown"
            conf = "  {0}%".format(round(100 - conf)) # conf = confidence
            # Email saying yesh this is me/any other intruder
            #build the email with the EmailMessage() object.
            msg = EmailMessage()
                
            msg['Subject'] = "Intruder?!"
            msg['From'] = "otheremail.locked@gmail.com"
            msg['To'] = "otheremail.locked@gmail.com" # your company email

            #set the body with .set_content()
            msg.set_content("Hello [owner's name]!\n I don't think that was your trusted people at the door ... ")

            #use smptlib to send using gmail server, on port 465. 
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)

            #replace with your own credentials
            server.login("otheremail.locked@gmail.com", "Hackathons123") 
                
            #send email   
            server.send_message(msg)
                    
            print("Sending email...")
                    
            server.quit()   

            with open('intruders.txt', 'a+') as f:
                f.write("To let you know there was someone at your place that could be suspicious at ")
                f.write(str(dateAll) + "\n")
                # f.write("\n")
            
            sense.set_pixels(citrine_hrt)
               
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n Clean Up has finished.")
cam.release()
cv2.destroyAllWindows()