import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = './images/20180322-171605_1_29.png'
filenameb='./images/20180322-171604_0_18.png'
image1 = cv2.imread(filename,0)
image2=cv2.imread(filenameb,0)
def calculatefeatures(img1,img2):
    n_white_pix = np.sum(img1 == 255)
    n_white_pixb=np.sum(img2==255)
    wherewhitea=np.argwhere(img1 == 255)
    wherewhiteb=np.argwhere(img2==255)
    peakx,peaky,minx,miny=0,0,0,0
    peakxb,peakyb,minxb,minyb=0,0,0,0
    count=0
    for coordinates in wherewhitea:
        count+=1
        if coordinates[1]>peakx:
            peakx=coordinates[1]
        if coordinates[0]>peaky:
            peaky=coordinates[0]
        if count==1:
            minx=peakx
            miny=peaky
        if minx>0 and miny>0:
            if coordinates[0]<miny:
                miny=coordinates[0]
            if coordinates[1]<minx:
                minx=coordinates[1]
    countb=0
    for coordinates in wherewhiteb:
        countb+=1
        if coordinates[1]>peakxb:
            peakxb=coordinates[1]
        if coordinates[0]>peakyb:
            peakyb=coordinates[0]
        if countb==1:
            minxb=peakxb
            minyb=peakyb
        if minxb>0 and minyb>0:
            if coordinates[0]<minyb:
                minyb=coordinates[0]
            if coordinates[1]<minxb:
                minxb=coordinates[1]

    print(minx,miny,peakx,peaky)
    print(minxb,minyb,peakxb,peakyb)

calculatefeatures(image1,image2)

plt.imshow(image1),plt.show()
plt.imshow(image2),plt.show()

"""
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
grayb=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3=None
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img3)
plt.imshow(img3),plt.show()
"""
