from FallDetect.readpickle import readpickle
vlist = readpickle("videoclassifier.txt")
from matplotlib import pyplot as plt
import matplotlib
import datetime
angles = []
distances = []
mses = []
timesl=[]
ssims=[]
ground_truth_lay_time=[]
for i in vlist:
    if i["time"]>datetime.datetime(2018, 3, 22, 17, 27, 56, 0) and i["time"]<datetime.datetime(2018, 3, 22, 17,28,56 , 0):
        angles.append(i["angle"])
        distances.append(i["distance"])
        mses.append(i["mse"])
        timesl.append(i["time"])
        ssims.append(i["ssim"])
start_time_a = datetime.datetime(2018, 3, 22, 17, 27, 56)
start_time_b = datetime.datetime(2018, 3, 22, 17, 15, 56)
times_a = [[8, 10, 21, 23], [41, 45, 57, 61], [75, 80, 93, 97], [110, 114, 126, 129], [142, 146, 159, 164],
           [178, 183, 195, 198], [219, 222, 234, 237], [258, 262, 274, 278], [299, 303, 313, 317], [332, 335, 349, 353],
           [378, 382, 392, 396], [420, 424, 435, 440], [452, 456, 466, 470], [486, 491, 501, 507]]
count=0
for times in times_a:
    if count<2:
        ground_truth_lay_time.append(start_time_a + datetime.timedelta(0, times[1]))
    count+=1

for i in range(len(distances)):
    if distances[i] > 1000:
        vlist[i]["distances"] = 0
        distances[i]=0
maxd = max(distances)
print(distances)
print(maxd)
maxa = max(angles)
maxmse = max(mses)
anglesnew,distancesnew,msesnew=[],[],[]
for i in range(len(distances)):
    anglesnew.append(float(vlist[i]["angle"] / maxa))
    distancesnew.append(float(distances[i] / maxd))
    msesnew.append(float(mses[i] / maxmse))

t=1
for lay_time in ground_truth_lay_time:
    if t:
        plt.axvline(lay_time, c='black', ls='dotted', lw=2.0, label='Fallen')
    else:
        plt.axvline(lay_time, c='black', ls='dotted', lw=2.0)
    t = 0


x_line = plt.plot(timesl, anglesnew, label='angle')
y_line = plt.plot(timesl, distancesnew, label='distance')
z_line = plt.plot(timesl, msesnew, label='mse')
m_line = plt.plot(timesl, ssims, label='ssim')

plt.xlabel('Timestamp(s)')
plt.ylabel('feature Value')
"""
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(myFmt)
"""
myFmt = matplotlib.dates.DateFormatter('%H:%M:%S')
plt.legend()
plt.show()
