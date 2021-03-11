import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology,draw

roi = (250, 500, 180, 500)
recv_height = 290

img = cv.imread("img.png")

import time

def process(img):
    tstart = time.time()
    imgroi = img[roi[2]:roi[3],roi[0]:roi[1]]
    h, w = imgroi.shape[:2]

    # 1. get the liquid
    # dark: water and cups
    # black: cups are most not-black
    mask_black = np.var(imgroi,axis=2)
    mask_black /= np.max(mask_black)/255
    mask_black = np.uint8(mask_black)

    mask_dark = np.uint8(np.mean(imgroi, axis=2))

    # 1.2 bin
    ret, bin_black = cv.threshold(mask_black,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, bin_black = cv.threshold(mask_black,min(255,ret*0.5),255,cv.THRESH_BINARY)
    ret, bin_dark = cv.threshold(mask_dark,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    darkblack = cv.bitwise_not(bin_black, mask=bin_dark)

    # 1.3 delete horizon connection
    hkernel = cv.getStructuringElement(cv.MORPH_RECT,(1,5))
    dh = darkblack#cv.dilate(darkblack, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)))
    dh= cv.erode(dh,hkernel)
    dh = cv.dilate(dh, hkernel)

    # 1.4 select the most center part
    num_labels, labels, stats, centers =cv.connectedComponentsWithStats(dh)

    for i in range(w/2):
        if labels[h/2, w/2-i] != 0:
            labelid = labels[h/2, w/2-i]
            break
        elif labels[h/2,w/2+i] != 0:
            labelid = labels[h/2,w/2+i]
            break

    liquid = labels==labelid
    #2 fit the curve

    lx, ly, lw, lh, larea = stats[labelid]

    liquid_pixels = np.where(liquid)
    liquid_line = imgroi.copy()
    liquid_line2 = imgroi.copy()
    #2.1 get the start
    #skel, distance =morphology.medial_axis(liquid, return_distance=True)
    liquid255 = np.zeros(imgroi.shape[:2],np.uint8)
    liquid255[liquid] = 1
    
    distance = cv.distanceTransform(liquid255, cv.DIST_L1, 3)
    print "skel  cost",time.time()-tstart
    #skel2 = morphology.skeletonize(liquid)
    #distance = distance+skel2*2
    distance = 1-np.exp(-distance)
    from scipy.ndimage import maximum_filter
    mx = maximum_filter(distance, size=5)
    waitlist = np.where((mx==distance) & (liquid))
    theid = 0
    def dis2lt(x,y):
        return abs(x-lx)+abs(y-ly)
    
    dists = [dis2lt(waitlist[0][i],waitlist[1][i]) for i in range(1,len(waitlist))]
    theid = np.argmin(dists)
    spx = waitlist[1][theid]
    spy = waitlist[0][theid]
    print 'start ',spx, spy

    print "spxy  cost",time.time()-tstart
    #2.2 get the curve
    # y = k * x^2 (k--g/v0)
    # Viscosity 0.001
    # #x = x0 + (int)(v0/c1*(1- np.exp(-c1 * rt))) 
            #y = y0 + (int)( 10/c2*(rt-(1-np.exp(-c2*rt))/c2))
    def curve(x0 ,y0, k):
        ret = []# np.zeros(imgroi.shape[:2])
        x = x0
        y = y0
        while y < h and x < w : #ly + lh
            ret.append([x,y])
            y += 1
            x = x0 + np.around(np.sqrt((y-y0)/k))
            
        #print ret, y, ly+lh, x, lx+lw
        return np.array(ret,np.int32)

    def curvemap(x0, y0, k):
        ret=np.zeros(imgroi.shape[:2])
        x,y = x0, y0
        pts = []
        while y<h and x<w:
            pts.append([x,y])
            x += 1
            y = np.uint32(y0 + k* (x-x0)**2)
        cv.polylines(ret,[np.array(pts)],False,1)
            # ret[y,x]=1
            # y += 1
            # x = x0 + np.uint32(np.sqrt((y-y0)/k))
        return ret
    
    def eval_map(cm, dx, dy):
        m = np.zeros(imgroi.shape[:2])
        m[dy:,dx:] = cm[:h-dy,:w-dx]
        return np.sum(m*distance)

    #2.3 evaluate the curve 
    def evaluate(pts):
        ret = 0
        for x,y in pts:
            ret += distance[y,x]
        return ret

    
    TRYSIZE = 8
    speeds = list(np.linspace(0.1,1.5,15))
    bmatch = np.zeros((len(speeds)))
    bdxdy = np.zeros((len(speeds),2))
    for i in range(len(speeds)):
        spd = speeds[i]
        oricurve = curvemap(spx, spy, spd)
        match = np.zeros((TRYSIZE,TRYSIZE))
        for dx in range(0,TRYSIZE):
            for dy in range(0,TRYSIZE):
                match[dx,dy]=np.sum(oricurve[:h-dy,:w-dx]*distance[dy:,dx:])
                #match[dx,dy]=eval_map(oricurve,dx,dy)
                #curves.append(curve(spx+dx,spy+dy,spd))
        #match = [evaluate(c) for c in curves]

        bmatch[i]=np.max(match)
        bdxdyid = np.argmax(match)
        bdxdy[i] = [bdxdyid/8,bdxdyid%8]
        #print i, bmatch[-1], bdxdy[-1]

    spdid = np.argmax(bmatch)
    print bdxdy[spdid][0], bdxdy[spdid][1]
    spx += bdxdy[spdid][0]
    spy += bdxdy[spdid][1]
    spd = speeds[spdid]



    endx = np.sqrt((recv_height-spy)/spd) + spx


    
    print "time cost:", time.time()-tstart

    print spd, spx, spy,np.max(bmatch)

    best_curve = curve(spx,spy,spd)
    cv.polylines(liquid_line, [best_curve], False, (0,255,0))
    print 'x=', endx
    cv.circle(liquid_line, ((int)(endx), recv_height), 2, (255,0,0))
    newdis = distance.copy()
    cv.polylines(newdis, [best_curve], False, (2))
    plt.subplot(231)
    plt.imshow(cv.cvtColor(imgroi, cv.COLOR_RGB2BGR))
    plt.subplot(232)
    plt.imshow(liquid)
    plt.subplot(233)
    plt.imshow(labels)
    plt.subplot(234)
    plt.imshow(newdis)
    plt.subplot(235)
    plt.imshow(distance)
    plt.subplot(236)
    
    plt.imshow(cv.cvtColor(liquid_line, cv.COLOR_RGB2BGR))
    plt.show()
    # cv.imshow('liquid', liquid_line)
    # cv.waitKey(2000)

#process(img)


namefile = open("/home/robot/kinect_tool/data/rgbdata.txt")
for i in range(100):
    namefile.readline()
imgname = namefile.readline()
for i in range(30):
    imgfile = "/home/robot/kinect_tool/data/"+imgname.split(',')[1][:-1]
    img = cv.imread(imgfile)
    process(img)
    imgname = namefile.readline()
