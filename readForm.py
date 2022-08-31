import csv
import itertools
import math,cv2,os
import pickle
import random

from paddleocr import PaddleOCR
import numpy as np
import jellyfish
import matplotlib.pyplot as plt

import neuralNet

ocr = PaddleOCR(use_angle_cls=True, lang='en')
from PIL import Image

import pdf2image
POPPLER_PATH = "C:/Users/Admin/Documents/poppler-22.04.0/Library/bin"

CANDIDATES = "raila odinga ruto william samoei "

DEAD_FORMS = 0

# region UTILITY FUNCTIONS
#convert a pdf file to an image file
def pdfToImage(pdf, root ="iebc_forms", county ="pdfs"):
    images = pdf2image.convert_from_path(f'{root}/{county}/{pdf}.pdf',poppler_path=POPPLER_PATH)
    cvIm = np.array(images[0])
    # Convert RGB to BGR
    return cvIm[:, :, ::-1].copy()

#compare 2 strings equality relative to a minimum levenshtein_distance
def compareStrings(text, term):
    term = term.upper()
    text = text.upper()
    if jellyfish.levenshtein_distance(term,text) <= math.ceil(len(term)/8):
        return True
    return False

def padVal(val,pad,padstr = "0"):
    vstr = str(val)
    if len(vstr) >= pad:
        return vstr
    return padstr*(pad - len(vstr))+vstr
# endregion


# region VOTE TALLY CROP FUNCTIONS
#estimate the general location of the tallies based on an anchor phrase
def boundsToCrop(blBound,width,translation):
    return [
        blBound[0]+width*translation[0],
        blBound[1]+width*translation[1],
        width*translation[2],
        width*translation[3]
    ]#x,y,w,h

#crop out the general location of the tallies on the form
def cropTallyGamma(im, imId):
    #crop out the bottom 2/3 of the form to save ocr execution time

    #location of tallies relative to regular phrases on the form
    anchorPhrases = {
        "presidential election results at the polling station":[
            1.068,0.237,0.2,0.4
        ],
        "number of votes cast in favour of each candidate":[
            2.429,-0.128,0.375,0.75
        ],
        "no of valid votes obtained":[
            2.193,-0.33,0.785,1.59
        ]
    }

    #using paddleocr to read words and phrases on the form
    result = ocr.ocr(im[:im.shape[0] // 3], cls=True)

    cropEstimates = []
    for bounds,text in result:
        for k,v in anchorPhrases.items():
            #for every string found on the form, check if string is in the anchor phrases
            if compareStrings(text[0], k):
                #if anchorPhrase has been found, store its bottom left corner and its width
                bl = bounds[3]
                w = bounds[1][0] - bounds[0][0]

                #use the estimated relative location of the anchor phrases to estimate the location of the tallies
                estimate = [boundsToCrop(bl,w,v),text[1],len(k)]
                cropEstimates.append(estimate)
                break

    #weight each anchor phrase based on its length and accuracy and find the average tallies location
    weightedSum = [0,0,0,0]
    weights = 0
    if len(cropEstimates) == 0:
        return False
    for est in cropEstimates:
        for di,dim in enumerate(est[0]):
            weightedSum[di] += dim*est[1]*est[2]
        weights += est[1]*est[2]

    cx, cy, cw, ch = [int(round(v/weights)) for v in weightedSum]
    cropIm = im[cy:cy+ch,cx:cx+cw]
    allVals = sorted(np.reshape(cropIm,[-1]))
    lVal,hVal = allVals[int(len(allVals)*0.02)],allVals[int(len(allVals)*0.98)]
    lCap,hCap = 10,230
    offset,stretch = lCap - lVal,(hCap-lCap)/(hVal-lVal)
    finIm = np.clip((cropIm+offset)*stretch,lCap,hCap).astype(np.uint8)
    #crop out the voter tallies and save the image
    cv2.imwrite(f'iebc_forms/imgcrops/{imId}.jpg',finIm )
    return True
# endregion


# region UNUSED FUNCTIONS
def readChars(img,crop=False):
    img_path = f'iebc_forms/imgs/{img}.jpg'
    if crop:
        img_path = f'iebc_forms/imgcropsfin/{img}.jpg'

    result = ocr.ocr(img_path, cls=True)
    topBound = None
    bottomBound = None
    for bounds,text in result:
        if compareStrings(text[0], "number of votes cast in favor of each candidate") and topBound == None:
            topBound = [bounds[0],bounds[1]]
            '''if topBound == None or max([x[1] for x in topLine]) < max([x[1] for x in topBound]):
                topBound = topLine'''
        elif compareStrings(text[0], "total number of valid votes cast") and bottomBound == None:
            bottomBound = [bounds[3], bounds[2]]
            '''if bottomBound == None or min([x[1] for x in bottomLine]) > min([x[1] for x in bottomBound]):
                bottomBound = bottomLine'''

        #print(text)

    if crop:
        return [txt for _,txt in result]
    return [topBound,bottomBound]

def extendLine(bound,width):
    vec = np.subtract(bound[1],bound[0])+np.array([0.0001,0.0001])
    y = bound[1][1] + round(vec[1]/vec[0] * (width-bound[1][0]))
    return y

def extendBounds(bounds,width):
    topBounds,bottomBounds = bounds
    topLim = extendLine(topBounds,width)
    bottomLim = extendLine(bottomBounds,width)
    return topLim,bottomLim

def cropTally(bounds,img):
    img_path = f'iebc_forms/imgs/{img}.jpg'
    im = Image.open(img_path)
    edgeBounds = extendBounds(bounds,im.width)

    imcrop = im.crop([0,bounds[0][0][1],im.width,edgeBounds[1]])
    imcrop.save(f'iebc_forms/imgcrops/{img}.jpg')

def cropTallyAlpha(img):
    img_path = f'iebc_forms/imgs/{img}.jpg'
    im = Image.open(img_path)
    imcrop = im.crop([im.width*0.7,im.width*0.3,im.width*0.95,im.width*0.55])
    imcrop.save(f'iebc_forms/imgcrops/{img}.jpg')

def cropTallyBeta(img):
    global DEAD_FORMS
    crop = cv2.imread(img, 0)
    detector = cv2.QRCodeDetector()
    data,bbox,_ = detector.detectAndDecode(crop[0:int(crop.shape[0]*0.2)])
    if type(bbox) == type(None):
        cropTallyAlpha(img)
        cleanCrop(img)
        return True

    else:
        qrw,qrh = bbox[0][0][0] - bbox[0][2][0],bbox[0][1][1] - bbox[0][0][1]
        br = [bbox[0][1][1],bbox[0][1][0]]
        charstl = [br[0] + qrh * 2, br[1]]
        charsCrop = crop[int(charstl[0]):int(charstl[0] + qrh * 3), int(charstl[1]):int(charstl[1] + qrw * 1.3)]
        # cropBlur = cv2.GaussianBlur(charsCrop, (15, 15), 0)
        # cropAlpha = 255 - cv2.threshold(cropBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        try:
            cv2.imwrite(f'iebc_forms/imgcropsfin/{img}.jpg', charsCrop)
            return True
        except:
            return False

def getAxisCrop(img,axis,size,padding = 50):
    profile = np.sum(img / 255, axis=axis)
    maxCrop = None
    for xi in range(len(profile)-size):
        if maxCrop == None or sum(profile[xi:xi+size]) > maxCrop[1]:
            maxCrop = [[max(0,xi-padding),min(xi+size+padding,len(profile))],sum(profile[xi:xi+size])]
    return maxCrop

def removeBlackEdge(img):
    profile = np.sum(img/255,axis = 0)
    frontAvg = np.mean(profile[:len(profile)//2])*0.95
    for ri in range(len(profile)):
        if profile[len(profile)-ri-1] >= frontAvg or ri > len(profile)//3:
            return img[:,:len(profile)-ri-1]

def cleanCrop(img):
    img_path = f'iebc_forms/imgcrops/{img}.jpg'
    crop = cv2.imread(img_path,0)
    crop = removeBlackEdge(crop)
    cropBlur = cv2.GaussianBlur(crop,(5,5),0)
    cropAlpha = cv2.threshold(cropBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    x,y,w,h = cv2.boundingRect(cv2.findNonZero(cropAlpha))
    cropAlpha = 255 - cropAlpha[y:y+h,x:x+w]
    #finding fin crop
    xCrop = getAxisCrop(cropAlpha,0,300)
    yCrop = getAxisCrop(cropAlpha,1,500)
    cleanCrop = crop[yCrop[0][0]:yCrop[0][1],xCrop[0][0]:xCrop[0][1]]
    cv2.imwrite(f'iebc_forms/imgcropsfin/{img}.jpg',cleanCrop)

    #plt.imshow(cv2.medianBlur(crop, 21))

def locateQR(img):
    qimg = cv2.imread('configData/iebc.jpg', 0)
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(qimg, None)
    kp2, des2 = sift.detectAndCompute(img, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    print("keep going")
    if len(good) >= 3:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = qimg.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(qimg, kp1, img, kp2, good, None, **draw_params)

    plt.imshow(img3, 'gray'), plt.show()

def detectDropoff(profile):
    vars = []
    lEdge,rEdge = None,None
    for i in range(len(profile)//2-2):
        if i > len(profile)//6:
            if abs(profile[i+len(profile)//2]-profile[i+len(profile)//2+2]) > np.mean(vars)*5 and rEdge == None:
                rEdge = i+len(profile)//2
            if abs(profile[len(profile)//2-i]-profile[len(profile)//2-i-2]) > np.mean(vars)*5 and lEdge == None:
                lEdge = len(profile)//2 - i
        vars.append(abs(profile[i+len(profile)//2]-profile[i+len(profile)//2+2]))
        vars.append(abs(profile[len(profile)//2-i]-profile[len(profile)//2-i-2]))
    if lEdge == None:
        lEdge = 0
    if rEdge == None:
        rEdge = len(profile)-1
    return lEdge,rEdge

def cropForm(im):
    img_path = f'iebc_forms/imgs/{im}.jpg'
    img = cv2.imread(img_path, 0)
    bimg = cv2.GaussianBlur(img,(25,25),0)
    yprofile = np.sum(bimg / 255, axis=0)
    yBounds = detectDropoff(yprofile)
    xprofile = np.sum(bimg / 255, axis=1)
    xBounds = detectDropoff(xprofile)

    fimg = img[xBounds[0]:xBounds[1],yBounds[0]:yBounds[1]]
    return fimg

# endregion


# region VOTE TALLY COUNT FUNCTIONS
#get number of votes for each candidate
def getVoterTalliesCrop(path):
    root,county,pdf = path.split("/")
    filename = pdf.split(".")[0]
    im = pdfToImage(filename, root, county)

    # crop out the general location of the vote tallies on the form
    cropTallyGamma(im,filename.split("_")[3])
    #cropTallyAlpha(filename)
    #cleanCrop(filename)
    '''if charsFound:
        votes = digitRecognition.deriveDigits(filename)  # extract the vote tallies
        return votes
    return []'''

def weighLine(digits):
    digits = sorted(digits,key=lambda x:x[1][0])
    alignedWgt = np.var([(pos[3]-pos[1])/2 for k,pos,conf in digits])
    confWgt = 1/sum([conf**1.5 for k,pos,conf in digits])
    overlapWgt = np.mean([abs(digits[di][1][2]-digits[di+1][1][0]-25) for di,dg in enumerate(digits[:-1])])
    return [alignedWgt,confWgt,overlapWgt]

def compareWeights(wa,wb,priorities = [0.6,0.3,0.1]):
    diff = 0
    wi = 0
    for a,b in zip(wa,wb):
        diff += ((b-a)/a)*priorities[wi]
        wi+=1
    return diff

def getEachLineBeta(values):
    lines = []
    values = sorted(values,key=lambda x:(x[1][1]+x[1][3])/2)
    getMidY = lambda x:(x[1][1]+x[1][3])/2
    gaps = [getMidY(values[vi+1])-getMidY(values[vi])
            for vi in range(len(values)-1)]
    avgHeight = np.mean([val[1][3]-val[1][1] for val in values])
    prevJump = 0
    for gi,gap in enumerate(gaps):
        if gap > avgHeight*0.333:
            line = values[prevJump:gi+1]
            if len(line) >= 3:
                lines.append(line)
            prevJump = gi+1
        if gi == len(gaps)-1 and gi+2 - prevJump >= 3:
            lines.append(values[prevJump:gi+2])

    lines = [sorted(sorted(line,key=lambda x:x[2],reverse=True)[:3],key=lambda x:x[1][0]) for line in lines][-4:]
    tallies = [int(''.join([dg[0] for dg in line])) for line in lines]
    return lines,tallies







def getEachLine(values):
    lines = []
    allCombos = [[combo,weighLine(combo)] for combo in itertools.combinations(values,3)]
    while len(values) >= 3:
        bestLine = None
        for combo,weights in allCombos:
            if not all([dg in values for dg in combo]):
                continue
            if bestLine == None or compareWeights(bestLine[1],weights) < 0:
                bestLine = [combo,weights]
        lines.append(sorted(bestLine[0],key=lambda x:x[1][0]))
        for v in bestLine[0]:
            values.remove(v)
    lines = sorted(lines,key = lambda x:np.mean([d[1][1] for d in x]))
    tallies = [int(''.join([dg[0] for dg in line])) for line in lines]
    return [lines,tallies]

CANDIDATES = ["RAILA","RUTO","WAJACKOYAH","WAIHIGA"]
LEVELS = ['COUNTY','CONSTITUENCY','WARD','POLLING CENTER','STREAM','POLLING STATION']

def getLocationCodes():
    csvfile = csv.reader(open("configData/locationCodes.csv","r"))
    rows = [x for x in csvfile]
    keys,values = rows[0],rows[1:]
    kdata = dict()
    for v in values:
        pid = padVal(v[keys.index("pollingStationId")].strip(),15)
        kdata[pid] = {
            "COUNTY":v[keys.index("countyName")],
            "CONSTITUENCY":v[keys.index("constituency name")],
            "WARD":v[keys.index("wardname")],
            "POLLING CENTER":v[keys.index("pollingCenterName")],
            "STREAM":v[keys.index("stream")],
            "POLLING STATION": pid,
            "VOTERS":v[keys.index("voters")]
        }
    return kdata

def saveTallies(data,level,locData):
    global CANDIDATES
    global LEVELS
    file = open(f'iebc_forms/tallyData/{level.title().replace(" ","")}Tallies.csv','w',newline='')
    csvfile = csv.writer(file)
    entries = list(set([v[level] for v in locData.values()]))

    voteTallies = {k:{cd: 0 for cd in CANDIDATES} for k in entries}

    for k,v in data.items():
        for ti,tally in enumerate(v[1]):
            voteTallies[locData[k][level]][CANDIDATES[ti]]+=tally
    csvfile.writerow(LEVELS[:LEVELS.index(level)+1]+CANDIDATES)
    completedEntries = []
    for pdata in locData.values():
        entry = pdata[level]
        if entry in completedEntries:
            continue
        completedEntries.append(entry)
        csvfile.writerow([pdata[d] for d in LEVELS[:LEVELS.index(level)+1]]+
                         [voteTallies[entry][cd] for cd in CANDIDATES])
    file.close()


def compileTallyData(data):
    levels = ['POLLING STATION','WARD','CONSTITUENCY','COUNTY']
    locationData = getLocationCodes()
    for level in levels:
        saveTallies(data,level,locationData)



def processRawData(rawData):
    finData = dict()
    for k,v in rawData.items():
        if len(v) < 12:
            continue
        finData[k] = getEachLineBeta(v)
    return finData

def addToVotes(votes,newVotes):
    order = ["raila","ruto","waihiga","wajackoyah"]
    for vi,v in enumerate(newVotes[:len(order)]):
        votes[order[vi]] += v

def iterateForms():
    global CANDIDATES
    for county in os.listdir("ALL_FORMS"):
        print(f"Processing {county}")
        forms = os.listdir(f"ALL_FORMS/{county}")
        totalForms = len(forms)
        for fi,form in enumerate(forms):
            getVoterTalliesCrop(f"ALL_FORMS/{county}/{form}")
            print(f"processed form {fi+1} out of {totalForms}")
        break #'''
    rawData = neuralNet.runAlpha()
    votes = processRawData(rawData)

    #count votes
    voteTallies = {cd:0 for cd in CANDIDATES}
    for k,v in votes.items():
        for ti,tally in enumerate(v[1][-4:]):
            voteTallies[CANDIDATES[ti]]+=tally

    #save csv files
    compileTallyData(votes)


    pickle.dump(voteTallies,open("iebc_forms/tallyData/finTally.p","wb"))
    print("done")
# endregion

if __name__ == '__main__':
    iterateForms()
