import cv2,time,random
import numpy as np
from perlin_noise import PerlinNoise


def padVal(val,pad,padstr = "0"):
    vstr = str(val)
    if len(vstr) >= pad:
        return vstr
    return padstr*(pad - len(vstr))+vstr

import math

def rotatePoint(origin, point, angle):
    angle = math.radians(angle)
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotateImage(image, angle,bg = 0):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image.astype(np.uint16)+1, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
  rIm = ((result == 0)*(bg+1) + (result-1)).astype(np.uint8)
  return rIm


def timeit(func):
    def wrapper(*args,**kwargs):
        st = time.time()
        ret = func(*args,**kwargs)
        print(f"{func.__name__} taken {time.time()-st} seconds")
        return ret
    return wrapper

def viewBounds(img,bounds,relative = True):
    rImg = np.copy(img)
    for bounds in bounds:
        if relative:
            bounds = bounds.split(" ")
            for i in range(1,5):
                    bounds[i] = int(float(bounds[i])*img.shape[((i-1)%2)])

        cls,x,y,w,h = bounds
        ax,ay = x-w//2,y-h//2
        cv2.rectangle(rImg,(ax,ay),(ax+w,ay+h),0,3)
    return rImg

def createNoiseMaps(n, gw, gh, layers = [20, 10, 5], weights = [0.3, 0.3, 0.4], div=10):
    NOISE_LAYERS = [PerlinNoise(octaves=l, seed=random.randrange(1, 10000)) for l in layers]
    genNoiseMap = lambda func, w=gw, h=gh, div=div: \
        cv2.resize(np.array([[func([x / (h / div), y / (h / div)])
                              for x in range(w // div)] for y in range(h // div)]), (w, h))

    noiseMaps = []
    for i in range(n):
        print("Creating Noise Maps ",i+1)
        noiseMap = sum([genNoiseMap(LAYER)*weights[li] for li,LAYER in enumerate(NOISE_LAYERS)])
        noiseMaps.append(noiseMap)
    return noiseMaps

def pickNoiseMap(noiseMaps):
    #pick map
    map = random.choice(noiseMaps)
    #maybe rotate
    map = np.rot90(map,k=random.choice([0,2]))
    #maybe flip
    flip = random.randint(-2,1)
    if flip == -2:
        return map
    map = cv2.flip(map,flip)
    return map
