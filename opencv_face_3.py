import numpy as np
import cv2
import math
#import time

my_list = []
#fecev= 0
#nosev=0
i=0
pnx =0
pny =0
npx=0
npy=0


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

cap = cv2.VideoCapture(0)

while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if i < 20 :

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                roi_color2 = img[pny:pny-100,pnx:pnx-100]
                roi_gray2 = gray[pny:pny-100,pnx:pnx-100]




                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    pnx=ex+23
                    pny=ey+23
                    cv2.circle(roi_color,(pnx,pny),20,(0,255,0),2)


                nose = nose_cascade.detectMultiScale(roi_gray)
                for (nx,ny,nw,nh) in nose :
                        cv2.circle(roi_color,(npx,npy),20,(0,0,255),2)
                        npx=nx+33
                        npy=ny+18


                print "npx = ",npx ,"npy = ", npy ,"pnx = ", pnx ,"pny = ", pny


                deltax= npx - pnx
                deltay= npy - pny

                valeur_nose_eye = (deltay**2 + deltax**2)**0.5


                print "valeur_nose_eye = ", valeur_nose_eye

                my_list.append(valeur_nose_eye)
                i = i + 1

        else :
            break


        valeur = np.array(my_list)


        
            
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break



valeur2 =  sum(my_list)/len(my_list)
valeur = np.array(my_list)

print "valeur = " , valeur2

cap.release()
cv2.destroyAllWindows()
