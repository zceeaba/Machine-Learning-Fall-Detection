from FallDetect.readpickle import readpickle
import cv2
from matplotlib import pyplot as plt

data=readpickle("testdataforbb.txt")
def imagewrite():
    count = 0
    for i in data:
        if count==0:
            #print(i,data[i])
            filename='./images/'+'one'+'.png'
            with open(filename, 'wb') as f:
                f.write(data[i][0])
        count = count+1

count=0
bb=0
for j in data:
    if count==1:
        bb=data[j][1]
    count+=1

import numpy as np

filename = './images/' + 'one' + '.png'
image1 = cv2.imread(filename)
print(bb)
cv2.rectangle(image1,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),3)
#image1=image1+image2
plt.imshow(image1),plt.show()

""""
filename = './images/' + timestr + "_0_" + str(count) + '.png'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(i[1])
"""