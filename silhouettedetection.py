from FallDetect.readsilhouette import readsilhouette
import pandas as pd
from PIL import Image
import time

def processsilhouette():
    array=readsilhouette()
    pdarray=pd.DataFrame(array)
    gts=pdarray.filter(items=['groundtruthstate','imagevalue','time'])
    print(pdarray)
    count=0
    for i in gts.values:
        if count<2000:
            timestr = i[2].strftime("%Y%m%d-%H%M%S")
            if i[0]==0:
                #print(i[1])
                filename ='./images/'+timestr+"_0_"+str(count)+'.png'  # I assume you have a way of picking unique filenames
                with open(filename, 'wb') as f:
                    f.write(i[1])
                #img = Image.open(filename)
                #img.show()
                count+=1
            elif i[0] == 1:
                # print(i[1])
                filename = './images/'+timestr+"_1_" + str(count) + '.png'  # I assume you have a way of picking unique filenames
                with open(filename, 'wb') as f:
                    f.write(i[1])
                # img = Image.open(filename)
                # img.show()
                count += 1
            elif i[0]==2:
                # print(i[1])
                filename = './images/' +timestr+"_2_"+ str(count) + '.png'  # I assume you have a way of picking unique filenames
                with open(filename, 'wb') as f:
                    f.write(i[1])
                # img = Image.open(filename)
                # img.show()
                count += 1

processsilhouette()