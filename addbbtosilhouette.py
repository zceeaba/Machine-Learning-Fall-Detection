from FallDetect.readpickle import readpickle
import cv2
from matplotlib import pyplot as plt

data=readpickle("testdataforbb.txt")
def imagewrite():
    for j in list(data):
        if len(data[j]) < 2:
            del data[j]
    count = 0
    for i in data:
        #if count==0:
        #print(i,data[i])
        filename='./images/'+str(count)+'.png'
        with open(filename, 'wb') as f:
            f.write(data[i][0])
        count = count+1

    print(data)

import numpy as np
for j in list(data):
    if len(data[j]) < 2:
        del data[j]
"""
for i in data:
    filename = './images/' +str(count)+ '.png'
    image1 = cv2.imread(filename)
    bb=data[i][1]
    cv2.rectangle(image1,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),3)
    cv2.imwrite(filename,image1)
    count+=1
"""
"""
count=0
for i in data:
    filename = './images/' +str(count)+ '.png'
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