import random

import cv2,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont

import utilityFunctions
from trainDataGeneration import generateDataset






def createNumberImg(nstr,nmaps):
    isize = [60,300]
    FONT_SIZE = int(random.normalvariate(25,3))
    IEBC_FONT = ImageFont.truetype('../configData/IEBC_Font/IEBC_Regular.ttf', size=FONT_SIZE)
    black,white = int(random.expovariate(0.2)),int(255-random.expovariate(0.2))
    padding = -2
    ek = int(random.expovariate(1.5))+1

    chrSizes = []

    im = np.full(isize, white, dtype=np.uint8)
    pIm = Image.fromarray(im)
    pImDraw = ImageDraw.Draw(pIm)

    for ci,char in enumerate(nstr):
        offx,offy,cx,cy = list(pImDraw.textbbox((0,0),char,IEBC_FONT))
        chrSizes.append([cx+ek,cy+ek,offx,offy])
    tsize = (sum([sz[0]-sz[2]+padding for sz in chrSizes]),max([sz[1]-sz[3] for sz in chrSizes]))
    org = [isize[1]//2-tsize[0]//2,isize[0]//2-tsize[1]//2]
    if len(nstr) > 10:
        org[0] = isize[1]-tsize[0]
    org[1] += int(random.normalvariate(0,2))

    bounds = []

    chrLocs = []
    for ci,char in enumerate(nstr):
        chr_x,chr_y = org[0]+sum([sz[0]-sz[2]+padding for sz in chrSizes[:ci]]),org[1]-chrSizes[ci][3],
        chr_w,chr_h = (chrSizes[ci][0]-chrSizes[ci][2]),(chrSizes[ci][1]-chrSizes[ci][3])

        bound = [chr_x+chrSizes[ci][0]//2,chr_y+chrSizes[ci][3]+chr_h//2,chr_w*1.5,chr_h*1.5]
        bounds.append([int(char),[d/im.shape[(di+1)%2] for di,d in enumerate(bound)]])
        pImDraw.text((chr_x,chr_y),char,black,IEBC_FONT)
        chrLocs.append([org[0]+sum([sz[0]-sz[2] for sz in chrSizes[:ci]]),org[1],
                        (chrSizes[ci][0] - chrSizes[ci][2]),
                        (chrSizes[ci][1] - chrSizes[ci][3])])

    fBounds = [f"{cl} {' '.join([str(round(x, 6)) for x in bd])}\n" for cl, bd in bounds]

    fIm = np.array(pIm)

    #bumpy characters
    fIm = cv2.erode(fIm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ek,ek)))

    #add noise to each character
    for x,y,w,h in chrLocs:
        chrNoiseRaw = utilityFunctions.pickNoiseMap(nmaps)
        nx,ny = random.randint(0,chrNoiseRaw.shape[1]-w),random.randint(0,chrNoiseRaw.shape[0]-h)
        chrNoise = chrNoiseRaw[ny:ny+h,nx:nx+w]
        chrNoise = (chrNoise - np.min(chrNoise)) * (220/(np.max(chrNoise)-np.min(chrNoise)))
        fIm[y:y+h,x:x+w] = np.max([chrNoise,fIm[y:y+h,x:x+w]],axis=0)
    return fIm,fBounds


def createTarget(*args):
    #unpacking arguments
    noiseMaps,tpltIms,_ = args
    bg = tpltIms[0]
    chrNMaps,bgNMaps = noiseMaps

    if random.randint(0,5) == 5:
        num = ''.join(str(int(random.expovariate(0.4))) for _ in range(15))
    else:
        num = str(int(random.expovariate(0.01)))

    im,fBounds = createNumberImg(num,chrNMaps)

    #place patterned background
    scale = np.clip(random.normalvariate(0.5,0.05),0.1,0.9)
    bg = cv2.resize(bg,(0,0),fx=scale,fy=scale)
    bgy,bgx = random.randint(250,bg.shape[0]-im.shape[0]-200),random.randint(250,bg.shape[1]-im.shape[1]-20)
    bgIm = np.clip(bg[bgy:bgy+im.shape[0],bgx:bgx+im.shape[1]].astype(np.int32)+
                   random.normalvariate(240,10),0,255).astype(np.uint8)
    im = np.min([bgIm,im],axis=0)

    #add background noise
    noise = np.random.random(im.shape)*0.2
    nIm = (im - im*noise).astype(np.int32)

    #add perlin noise
    pNoiseRaw = utilityFunctions.pickNoiseMap(bgNMaps)
    py = random.randint(0,pNoiseRaw.shape[0]-nIm.shape[0])
    pNoise = pNoiseRaw[py:py+nIm.shape[0]]*0.5
    pnIm = np.clip(nIm + pNoise*nIm,0,255).astype(np.uint8)


    #add random blur
    bk = random.randint(1,2)*2+1
    bIm = cv2.GaussianBlur(pnIm,(bk,bk),0)

    return bIm,fBounds


if __name__ == '__main__':
    generateDataset(
        n=1000,
        generator=createTarget,
        folder="tgt",
        noiseMaps=[[50, 50,50,[20],[1],4],[25,300,300,[20],[1]]],
        templatePaths=["../configData/talliesBg.png"],
        extraData=[]
    )