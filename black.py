#!usr/bin/python
#coding:utf-8

import cv2 as cv
import numpy as np
from scipy.ndimage import maximum_filter

SEARCH_LENGTH_FACTOR = 0.5
DEBUG = True

roi = (250, 500, 180, 500)
recv_height = 290
img = cv.imread("img.png")

if DEBUG:
    import matplotlib.pyplot as plt
    import time

def process(img):

    tstart = time.time()

    # 0. ROI
    imgroi = img[roi[2]:roi[3],roi[0]:roi[1]]
    h, w = imgroi.shape[:2]

    # 1. get the liquid
    # dark: water and cups
    # black: cups are most not-black
    mask_dark = np.uint8(np.mean(imgroi, axis=2)) # not dark
    mask_black = np.var(imgroi,axis=2)
    mask_black /= np.max(mask_black)/255
    mask_black = np.uint8(mask_black)

    # 1.2 bin
    ret, bin_black = cv.threshold(mask_black,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, bin_black = cv.threshold(mask_black,min(255,ret*0.5),255,cv.THRESH_BINARY) # 
    ret, bin_dark = cv.threshold(mask_dark,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    darkblack = cv.bitwise_not(bin_black, mask=bin_dark)

    # 1.3 delete horizon connection
    hkernel = cv.getStructuringElement(cv.MORPH_RECT,(1,5))
    dh = darkblack#cv.dilate(darkblack, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)))
    dh= cv.erode(dh,hkernel)
    dh = cv.dilate(dh, hkernel)

    # 1.4 select the most center part
    # if can not find
    num_labels, labels, stats, centers =cv.connectedComponentsWithStats(dh)

    for i in range((int)(w/2*SEARCH_LENGTH_FACTOR)):
        if labels[(int)(h/2), (int)(w/2)-i] != 0:
            labelid = labels[(int)(h/2), (int)(w/2)-i]
            break
        elif labels[(int)(h/2),(int)(w/2)+i] != 0:
            labelid = labels[(int)(h/2),(int)(w/2)+i]
            break
    
    if labelid == None:
        return False

    liquid = labels==labelid

    if DEBUG:
        print("black cost:"+ str(time.time()-tstart))

    #2 fit the curve
    lx, ly, lw, lh, larea = stats[labelid]
    
    #2.1 get the keypoints    
    liquid255 = np.zeros(imgroi.shape[:2],np.uint8)
    liquid255[liquid] = 1    
    distance = cv.distanceTransform(liquid255, cv.DIST_L1, 3)
    #distance = 1-np.exp(-distance)
    mx = maximum_filter(distance, size=5)
    #2.1.1 the center points
    local_max = liquid & (mx == distance)

    _lmpt = np.where(np.transpose(local_max)) # in x order

    #2.1.2 decrease points' amount
    _lqkpts = []
    i = 0
    ll = len(_lmpt[0])
    while i<ll: # 乘上一个系数来查看黑色检测结果对拟合的影响
        j = 1
        sumj = _lmpt[1][i]
        while i<ll-1 and _lmpt[0][i+1] == _lmpt[0][i]:
            j += 1
            sumj += _lmpt[1][i+1]
            i += 1            
        _lqkpts.append([sumj/j, _lmpt[0][i]])
        i += 1
    
    lqkpts = np.transpose(_lqkpts)
    #2.1.3 fit
    pxparams = np.polyfit(lqkpts[1],lqkpts[0],3)
    px = np.poly1d(pxparams) # x-y

    print("fit cost:"+ str(time.time()-tstart))

    endpx = pxparams.copy()
    print(endpx)
    endpx[-1] -= recv_height
    endx = np.roots(endpx)
    endx = [ex.real for ex in endx if np.imag(ex) == 0]
    endx = (int)(np.round(endx[0]))
    # 有可能出问题，这时表示拟合完全失败
    
    print('endx cost:'+ str(time.time()-tstart))
    print('endx=', endx)

    # Plot
    liquid_line = imgroi.copy()    
    dwy = list(range(ly, ly+lh))
    pts = np.array([[i,np.int(px(i))] for i in dwy])
    cv.polylines(liquid_line,[pts],False,(0,255,0))
    cv.circle(liquid_line, ((int)(endx), recv_height), 2, (255,0,0))
    newdis = distance.copy()
    cv.polylines(newdis, [pts], False, (2))

    plt.subplot(231)
    plt.imshow(cv.cvtColor(imgroi, cv.COLOR_RGB2BGR))
    plt.subplot(232)
    plt.imshow(liquid)
    plt.subplot(233)
    keypoints = np.zeros(imgroi.shape[:2])
    for i in range(len(lqkpts[0])):
        keypoints[int(lqkpts[0][i]),int(lqkpts[1][i])] = 255
    plt.imshow(keypoints)
    plt.subplot(234)
    plt.imshow(newdis)
    plt.subplot(235)
    plt.imshow(cv.cvtColor(liquid_line, cv.COLOR_RGB2BGR))
    plt.subplot(236)
    
    plt.imshow(cv.cvtColor(liquid_line, cv.COLOR_RGB2BGR))
    plt.show()
    # cv.imshow('liquid', liquid_line)
    # cv.waitKey(2000)

    return pxparams, endx

process(img)

"""
namefile = open("/home/robot/kinect_tool/data/rgbdata.txt")
for i in range(100):
    namefile.readline()
imgname = namefile.readline()
for i in range(30):
    imgfile = "/home/robot/kinect_tool/data/"+imgname.split(',')[1][:-1]
    img = cv.imread(imgfile)
    process(img)
    imgname = namefile.readline()
"""