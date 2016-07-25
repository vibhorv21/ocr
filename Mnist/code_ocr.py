import struct
from skimage import data, color, exposure
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
import cv2
from PIL import Image
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import sys
import numpy as np





Y=np.zeros((28,28),dtype=np.uint8)
temp=np.zeros((28*28),dtype=np.uint8)



timg=open('tr-img')
magic= struct.unpack('>I',timg.read(4))[0]
nimg = struct.unpack('>I',timg.read(4))[0]
nrow = struct.unpack('>I',timg.read(4))[0]
ncol = struct.unpack('>I',timg.read(4))[0]

print "No of Training Samples ",
print nimg



X=[]


for a in range(0,nimg):
	for e in range(0,28*28):
			x=int(struct.unpack('>B',timg.read(1))[0])
			temp[e]=x
	Y=np.reshape(temp, (28, 28))
	img = Image.fromarray(Y)
	fd = hog(Y, orientations=9, pixels_per_cell=(5,5), cells_per_block=(1, 1), visualise=False)

	X.append(fd)




tl=open('tr-label')
magic= struct.unpack('>I',tl.read(4))[0]
label = struct.unpack('>I',tl.read(4))[0]

L=[ 0 for b in range(label)]
for a in range(0,label):
		x=int(struct.unpack('>B',tl.read(1))[0])
		L[a]=x

clf = RandomForestClassifier(n_estimators=70, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
clf = clf.fit(X,L)



############################# testing part  #########################
im = cv2.imread('test.png')

out = np.zeros(im.shape,np.uint8)

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

_, contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            cv2.imshow('im1',roi)
            roismall =  hog(cv2.resize(roi,(28,28)), orientations=9, pixels_per_cell=(5,5), cells_per_block=(1, 1), visualise=False)
            results = clf.predict(roismall)
            string = str(int((results[0])))
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

cv2.imshow('im',im)
cv2.imshow('out',out)
cv2.waitKey(0)
