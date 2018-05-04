from FallDetect.decodesilhouette import decodesilhouette
from FallDetect.readpickle import readpickle
from FallDetect.bounding_box import returnrecord
from FallDetect.pickledata import pickledata
from FallDetect.aks_test_bounding import aks_test_bounding

def pickledcombinedata():
    #data=returnrecord()
    data=aks_test_bounding()
    pickledata(data,"aks_test_bounding")

def combinesilandbb():
    data=readpickle("combinedsilandbb.txt")

    count=0
    for key in list(data):
        #if type(data[key][1])==str:
        if len(data[key])>3:
            #if type(data[key][3]) == str:
            count+=1
            #print(data[key][3])
            del data[key]
    print(count)


    for key in list(data):
        if type(data[key][0])==list:
        #if len(data[key])>3:
            #if type(data[key][3]) == str:
            count+=1
            #print(data[key][3])
            del data[key]
    print(count)

    """"
    final_list_time_stamp, final_list_2D_bb, final_list_2D_cen, final_list_silhouette=aks_test_bounding()
    data={}
    for i in range(len(final_list_silhouette)):
        data[final_list_time_stamp[i]]=[]
        data[final_list_time_stamp[i]].append(final_list_silhouette[i])
        data[final_list_time_stamp[i]].append(final_list_2D_bb[i])
        data[final_list_time_stamp[i]].append(final_list_2D_cen[i])

    print(data)
    """

    for key,value in data.items():
        data[key][0]=decodesilhouette(data[key][0])
        #data[key][3] = decodesilhouette(data[key][3])
    keylist = data.keys()
    sorted(keylist)
    newdata={}
    for key in keylist:
        newdata[key]=data[key]
       #print(key, data[key])

    pickledata(newdata,"testdataforbb")

    print(newdata)

combinesilandbb()