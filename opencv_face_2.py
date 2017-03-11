import numpy as np
import cv2
import time

my_list = []
valeur_eye = 0
i=0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

while 1:
    if i<100:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]



            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
               cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    	      # valeur_eye = float(ey+eh)/(ex+ew)



          #    valeur = float(y+h)/(x+w)
            #print "valeur face" , valeur
    	    #print "valeur eye" , valeur_eye
            #time.sleep(1)
            # i=i+1
            #my_list.append(valeur)
    else:
        break


    C = np.array(my_list)










    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(C)
print "face valeur : " , np.mean(C)
