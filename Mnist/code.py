import struct
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import sys
import cv2
import numpy as np

k=int(sys.argv[1])

timg=open('tr-img')
magic= struct.unpack('>I',timg.read(4))[0]
nimg = struct.unpack('>I',timg.read(4))[0]
nrow = struct.unpack('>I',timg.read(4))[0]
ncol = struct.unpack('>I',timg.read(4))[0]

print "No of Training Samples ",
print nimg



X=[[ 0 for i in range(784)] for b in range(nimg)]

for a in range(0,nimg):
	for b in range(0,nrow*ncol):
		x=int(struct.unpack('>B',timg.read(1))[0])
		X[a][b]=x


tl=open('tr-label')
magic= struct.unpack('>I',tl.read(4))[0]
label = struct.unpack('>I',tl.read(4))[0]



L=[ 0 for b in range(label)]

for a in range(0,label):
		x=int(struct.unpack('>B',tl.read(1))[0])
		L[a]=x



tsimg=open('ts-img')
magic= struct.unpack('>I',tsimg.read(4))[0]
nsimg = struct.unpack('>I',tsimg.read(4))[0]
nsrow = struct.unpack('>I',tsimg.read(4))[0]
nscol = struct.unpack('>I',tsimg.read(4))[0]

print "No of Testing Samples ",
print nsimg

Z=[[ 0 for i in range(784)] for j in range(nsimg)]

for a in range(0,nsimg):
	for b in range(0,nrow*nscol):
		x=int(struct.unpack('>B',tsimg.read(1))[0])
		Z[a][b]=x

tl=open('ts-label')
magic= struct.unpack('>I',tl.read(4))[0]
slabel = struct.unpack('>I',tl.read(4))[0]



Ls=[ 0 for b in range(slabel)]

for a in range(0,slabel):
		x=int(struct.unpack('>B',tl.read(1))[0])
		Ls[a]=x

neigh = KNeighborsClassifier(n_neighbors=k).fit(X,L)
indices = neigh.predict(Z)
pred=[]
for x in range(0,slabel):
	#print indices[x]
	pred.append(indices[x])

eff=0
for x in range(0,slabel):
	if pred[x]==Ls[x]:
		eff=eff+1

efc=float(eff)/float(slabel)
print "Eff. =",
print efc*100
