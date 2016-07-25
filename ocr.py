import sys

import numpy as np
import cv2

im = cv2.imread('data.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

#################      Now finding Contours         ###################

_, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

samples =  []
responses = []
keys = [i for i in range(48,58)]
print keys
for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roi_1 = im[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(28,28))
            cv2.imshow('norm',roi_1)
            key = cv2.waitKey(0)
            print key
            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,784))
                samples.append(sample[0])


print "training complete"
np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
