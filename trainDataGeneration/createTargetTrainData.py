import random

import cv2,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from trainDataGeneration import generateDataset






def createNumberImg(nstr):
    isize = [60,300]
    FONT_SIZE = int(random.normalvariate(25,3))
    IEBC_FONT = ImageFont.truetype('../configData/IEBC_Font/IEBC_Regular.ttf', size=FONT_SIZE)
    black,white = int(random.expovariate(0.2)),int(255-random.expovariate(0.2))
    padding = random.randint(0,2)
    chrSizes = []

    im = np.full(isize, white, dtype=np.uint8)
    pIm = Image.fromarray(im)
    pImDraw = ImageDraw.Draw(pIm)

    for ci,char in enumerate(nstr):
        csize = list(pImDraw.textbbox((0,0),char,IEBC_FONT,spacing=padding))[2:]
        chrSizes.append(csize)
    tsize = (sum([sz[0] for sz in chrSizes]),max([sz[1] for sz in chrSizes]))
    org = [isize[1]//2-tsize[0]//2,isize[0]//2-tsize[1]//2]
    if len(nstr) > 10:
        org[0] = isize[1]-tsize[0]

    bounds = []

    for ci,char in enumerate(nstr):
        bound = [org[0]+sum([sz[0] for sz in chrSizes[:ci]])+chrSizes[ci][0]//2,
                 org[1]+chrSizes[ci][1]//2,chrSizes[ci][0]*1.5,chrSizes[ci][1]*1.5]
        bounds.append([int(char),[d/im.shape[(di+1)%2] for di,d in enumerate(bound)]])
        pImDraw.text(tuple([org[0]+sum([sz[0] for sz in chrSizes[:ci]]),org[1]]),char,black,IEBC_FONT)



    fBounds = [f"{cl} {' '.join([str(round(x, 6)) for x in bd])}\n" for cl, bd in bounds]

    return np.array(pIm),fBounds


def createTarget(*args):
    #unpacking arguments
    _,tpltIms,_ = args
    bg = tpltIms[0]

    if random.randint(0,5) == 5:
        num = ''.join(str(int(random.expovariate(0.4))) for _ in range(15))
    else:
        num = str(int(random.expovariate(0.01)))

    im,fBounds = createNumberImg(num)

    #place patterned background
    scale = np.clip(random.normalvariate(0.5,0.05),0.1,0.9)
    bg = cv2.resize(bg,(0,0),fx=scale,fy=scale)
    bgy,bgx = random.randint(250,bg.shape[0]-im.shape[0]-200),random.randint(250,bg.shape[1]-im.shape[1]-20)
    bgIm = np.clip(bg[bgy:bgy+im.shape[0],bgx:bgx+im.shape[1]].astype(np.int32)+
                   random.normalvariate(210,20),0,255).astype(np.uint8)
    im = np.min([bgIm,im],axis=0)

    #add background noise
    noise = np.random.random(im.shape)*0.4
    nIm = (im - im*noise).astype(np.uint8)

    #add random blur
    bk = random.randint(1,2)*2+1
    bIm = cv2.GaussianBlur(nIm,(bk,bk),0)

    return bIm,fBounds


if __name__ == '__main__':
    generateDataset(
        n=2500,
        generator=createTarget,
        folder="tgt",
        noiseMaps=[],
        templatePaths=["../configData/talliesBg.png"],
        extraData=[]
    )