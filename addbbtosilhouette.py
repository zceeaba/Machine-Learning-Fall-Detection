from FallDetect.readpickle import readpickle
import cv2
from matplotlib import pyplot as plt
from FallDetect.readsilhouette import readsilhouette
import pandas as pd
import math
#data=readpickle("testdataforbb.txt")
data=readpickle("finalsillbbdata.txt")
def imagewrite():
    for j in list(data):
        if len(data[j]) < 2:
            del data[j]
    count = 0
    for i in data:
        #if count==0:
        #print(i,data[i])
        filename='./imagesb/'+str(count)+'.png'
        with open(filename, 'wb') as f:
            f.write(data[i][0])
        count = count+1

    print(data)

"""
import numpy as np
for j in list(data):
    if len(data[j]) < 2:
        del data[j]

count=0
for i in data:
    filename = './imagesb/' +str(count)+ '.png'
    image1 = cv2.imread(filename)
    bb=data[i][1]
    bbcen=data[i][2]
    cv2.circle(image1, (bbcen[0], bbcen[1]), 3, (0, 0, 255), -1)
    cv2.rectangle(image1,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),3)
    cv2.imwrite(filename,image1)
    count+=1
"""
import numpy as np
for j in list(data):
    if len(data[j]) < 2:
        del data[j]
array = readsilhouette()
#print(array)
count=0
print(len(array))
length=len(array)
keys=list(data.keys())
def determine(value):
    if value in keys:
        return False
    else:
        return True
newarray = [x for x in array if not determine(x["time"])]
bb=[]
bbcen=[]
for i in data:
    bb.append(data[i][1])
    bbcen.append(data[i][2])
    count+=1
def distance(pointa,pointb):
    dis=math.sqrt(((pointa[0]-pointb[0])**2+(pointa[1]-pointb[1])**2))
    return dis

def angle(pointa,pointb):
    y=pointa[1]-pointb[1]
    x=pointa[0]-pointb[0]
    angle=math.atan2(y,x)
    degree=math.degrees(angle)
    return degree

newarray[0]["bb"]=bb[0]
newarray[0]["bbcen"]=bb[1]
newarray[0]["distance"]=0
newarray[0]["angle"]=0

for j in range(1,len(newarray)):
    newarray[j]["bb"]=bb[j]
    newarray[j]["distance"]=distance(bb[j],bb[j-1])
    newarray[j]["angle"]=angle(bb[j],bb[j-1])
    newarray[j]["bbcen"]=bbcen[j]

pdarray = pd.DataFrame(newarray)

print(pdarray)
#gts = pdarray.filter(items=['groundtruthstate', 'imagevalue', 'time'])

"""
count=0
for i in data:
    filename = './imagesb/' +str(count)+ '.png'
    image1 = cv2.imread(filename)
    bb=data[i][2]
    cv2.circle(image1, (bb[0], bb[1]), 1, (0, 0, 255), -1)
    cv2.imwrite(filename,image1)
    count+=1
"""


""""
filename = './images/' + timestr + "_0_" + str(count) + '.png'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(i[1])
"""