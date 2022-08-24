import os

import cv2,math
import numpy as np

import neuralNet


def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def straightenImage(img,tgImg):
    base = 15
    for i in range(5):
        leastCost = None
        for offset in [-1,0,1]:
            timg = rotateImage(img,offset*base/(2**i))
            cost = sum([math.sqrt(x) for x in np.sum(timg,axis=0)])
            if leastCost == None or cost < leastCost[1]:
                leastCost = [offset*base/(2**i),cost]
        img = rotateImage(img,leastCost[0])
        tgImg = rotateImage(tgImg,leastCost[0])
    return img,tgImg

def getCropWindow(img,axis,ratio):
    profile = np.sum(img / 255, axis=1-axis)
    prAvg = np.max(sorted(profile)[:int(len(profile)*(1-ratio))])
    window = [img.shape[axis] // 4, int(img.shape[axis] * 3 / 4)]
    for i in range(2, 10):
        for wi in [0, 1]:
            highestSum = None
            for offset in [-1, 0, 1]:
                twindow = window[:]
                twindow[wi] = window[wi] + offset * img.shape[axis] / (2 ** i)
                tsum = sum(
                    [np.sign(x - prAvg) * math.sqrt(abs(x - prAvg)) for x in profile[int(twindow[0]):int(twindow[1])]])
                if highestSum == None or tsum > highestSum[1]:
                    highestSum = [twindow, tsum]
            window = highestSum[0][:]
    return sorted([int(x) for x in window])

def cropDigitColumn(img,tgImg):
    x,y,w,h = cv2.boundingRect(cv2.findNonZero(img))
    img = img[y:y+h+20,x:x+w-20]
    tgImg = tgImg[y:y+h+20,x:x+w-20]
    padding = 20
    xWindow = getCropWindow(img,1,ratio=0.5)
    img = img[:,max(0,xWindow[0]-padding):min(img.shape[1]-1,xWindow[1]+padding)]
    tgImg = tgImg[:,max(0,xWindow[0]-padding):min(tgImg.shape[1]-1,xWindow[1]+padding)]
    '''yWindow = getCropWindow(img,0,ratio=0.5)
    img = img[max(0,yWindow[0]-padding):min(img.shape[0]-1,yWindow[1]+padding)]'''
    return img,tgImg
AVG_LINES = []
def cropEmptyRows(img,tgImg):
    profile = np.sum(img[:,int(img.shape[1]*2/5):int(img.shape[1]*4/5)]/255,axis=1)
    cuts = []
    gap = 0
    inCut = False
    for i in range(len(profile)-1,-1,-1):
        if profile[i] == 0 or i == 0 or (len(cuts) > 0 and cuts[-1][0]-i >= 30 and inCut):
            if inCut:
                inCut = False
                cuts[-1][1] = i
            gap += 1
        else:
            if gap > 50:
                inCut = True
                cuts.append([i, None, gap])

            gap = 0

    avgLine = np.mean([x[2]+(x[0]-x[1]) for x in cuts])
    global AVG_LINES
    AVG_LINES.append(avgLine)
    cuts = list(reversed(cuts))+[[img.shape[0]-1,img.shape[0]-5,0]]
    maxGap = [[0,cuts[0][1]],cuts[0][1]]
    for ci,cut in enumerate(cuts[:-1]):
        tgap = cuts[ci+1][1] - (cuts[ci][0]+cuts[ci][2])
        if maxGap == None or tgap > maxGap[1]:
            maxGap = [[cuts[ci][0]+cuts[ci][2],cuts[ci+1][1]],tgap]

    fimg = img[max(maxGap[0][0],np.argwhere(profile > 0)[0][0]):min(np.argwhere(profile > 0)[-1][0],maxGap[0][1])]
    tgImg = tgImg[max(maxGap[0][0],np.argwhere(profile > 0)[0][0]):min(np.argwhere(profile > 0)[-1][0],maxGap[0][1])]
    return fimg,tgImg






def toHeavierStrokes(img):
    img = cv2.dilate(img,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    ogDensity = len(np.argwhere(img))
    layers = [img[:]]
    while True:
        img = cv2.erode(img,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        if len(np.argwhere(img)) < ogDensity*0.1:
            break
        layers.append(img[:])

    finImg = np.zeros(img.shape,dtype=np.float32)
    weightSum = 0
    for li,layer in enumerate(layers):
        finImg+=layer*math.log(li+1,3)*3
        weightSum += math.log(li+1,3)*3
    return finImg/weightSum




def deriveDigits(imgname):
    imgpath = f'iebc_forms/imgcropsfin/{imgname}.jpg'
    neuralNet.run()
    #cv2.imwrite(f'iebc_forms/imgcleans/{imgname}.jpg',crimg)
    #print("done")

def splitDigitLines(img):
    profile = np.sum(img/255,axis=1)
    peak = max(profile)
    cutOff = peak*2/3
    cuts = []
    while True:
        if max(profile) > cutOff:
            cut = int(np.argwhere(profile == max(profile))[0])
            cuts.append(cut)
            profile[cut-15:cut+15] = 0
            continue
        break
    cuts.sort()
    imgs = []
    for ci in range(len(cuts)-1):
        imgs.append([img[cuts[ci]+5:cuts[ci+1]-5],ci])

    finimgs = sorted(imgs,reverse=True,key=lambda x:sum(np.sum(x[0],axis=1)))[:4]
    finimgs.sort(key=lambda x:x[1])
    return [x[0] for x in finimgs]





if __name__ == '__main__':
    for im in os.listdir('iebc_forms/imgcropsfin'):
        deriveDigits(im.split(".")[0])
    print("done")