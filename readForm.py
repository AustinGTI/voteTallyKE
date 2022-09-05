import csv,itertools,math,cv2,os,pickle
import time

from paddleocr import PaddleOCR
import numpy as np
import jellyfish
import matplotlib.pyplot as plt

from neuralNet import YOLOModel

from PIL import Image

import pdf2image

import utilityFunctions

POPPLER_PATH = "C:/Users/Admin/Documents/poppler-22.04.0/Library/bin"

#CANDIDATES = "raila odinga ruto william samoei "


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

def setImageContrast(im,lCap,hCap,maxTail = 0.02):
    allVals = sorted(np.reshape(cv2.resize(im,(0,0),fx=0.1,fy=0.1,interpolation=cv2.INTER_NEAREST), [-1]))
    lVal, hVal = allVals[int(len(allVals) * maxTail)], allVals[int(len(allVals) * (1-maxTail))]
    offset, stretch = lCap - lVal, (hCap - lCap) / (hVal - lVal)
    ctIm = np.clip((im + offset) * stretch, lCap, hCap).astype(np.uint8)
    return ctIm
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
def cropTallyGamma(im,imId,saveIm = True):
    #crop out the bottom 2/3 of the form to save ocr execution time

    #location of tallies relative to regular phrases on the form
    anchorPhrases = {
        "presidential election results at the polling station":[
            0.87,0.237,0.4,0.4
        ],

        "number of votes cast in favour of each candidate":[
            2.05,-0.128,0.75,0.75
        ],
        "no of valid votes obtained":[
            1.4,-0.33,1.59,1.59
        ]
    }
    #mIm = cv2.resize(im,(0,0),fx=downsize,fy=downsize)
    #using paddleocr to read words and phrases on the form
    fresult = OCR.ocr(im[:im.shape[0]//2], cls=False)



    #rotate image to perfectly horizontal text
    rots,confs = [],[]
    fresult = sorted(fresult,key=lambda x:len(x[1][0])*x[1][1]**2,reverse=True)


    cropEstimates = []
    for bounds,text in fresult:
        for k,v in anchorPhrases.items():
            #for every string found on the form, check if string is in the anchor phrases
            if compareStrings(text[0], k):
                #if anchorPhrase has been found, store its bottom left corner and its width
                #bl = bounds[3]
                #w = bounds[1][0] - bounds[0][0]

                #use the estimated relative location of the anchor phrases to estimate the location of the tallies
                #cropEstimates.append([bounds,len(k)])
                cropEstimates.append([bounds,k,v,text])
                break
        rot = math.degrees(math.atan((bounds[1][1]-bounds[0][1])/(bounds[1][0]-bounds[0][0])))
        rots.append(rot)
        confs.append(len(text[0])*text[1])

    if len(cropEstimates) == 0:
        print("No estimate text found in image")
        return False

    imRotation = sum([rot*conf for rot,conf in zip(rots,confs)])/sum(confs)
    cropEstimates = [[[utilityFunctions.rotatePoint(tuple(np.array(im.shape[1::-1]) / 2),pt,-imRotation)
                       for pt in bds],k,v,text]for bds,k,v,text in cropEstimates]
    for ci in range(len(cropEstimates)):
        bounds,k,v,text = cropEstimates[ci]
        bl = bounds[3]
        w = bounds[1][0] - bounds[0][0]
        estimate = [boundsToCrop(bl, w, v), text[1], len(k)]
        cropEstimates[ci] = estimate
    #estimate = [boundsToCrop(bl, w, v), text[1], len(k)]

    rIm = utilityFunctions.rotateImage(im,imRotation,bg = 200)

    #weight each anchor phrase based on its length and accuracy and find the average tallies location
    weightedSum = [0,0,0,0]
    weights = 0
    if len(cropEstimates) == 0:
        return False
    for est in cropEstimates:
        for di,dim in enumerate(est[0]):
            weightedSum[di] += dim*est[1]*est[2]
        weights += est[1]*est[2]

    #crop out the voter tallies
    cx, cy, cw, ch = [int(round(v/weights)) for v in weightedSum]

    cropIm = rIm[cy:cy+ch,cx:cx+cw]
    if not (cropIm.shape[0] > 0 and cropIm.shape[1] > 0):
        return False
    #increase the contrast of the image
    ctIm = setImageContrast(cropIm,0,255)
    #save the image
    if saveIm:
        cv2.imwrite(f'iebc_forms/imgcrops/{imId}.jpg',ctIm)
    return ctIm

def cropTallyEpsilon(im,imId,model,saveIm = True):
    im = im[:im.shape[0]//2]
    inference = model.inferImage(im,imId)
    targets = ["logo","qr","qrTl","qrTr","qrBr"]
    if not all([any([x[0] == tg and x[2] > 0.85 for x in inference]) for tg in targets]):
        print("Dead due to low confidence targets")
        return False

    #get best targets found
    locs = dict()
    for tg,bounds,conf in inference:
        if tg not in locs.keys():
            locs[tg] = [[(bounds[0]+bounds[2])//2,(bounds[1]+bounds[3])//2],
                        [bounds[2]-bounds[0],bounds[3]-bounds[1]],conf]
        else:
            if conf > locs[tg][2]:
                locs[tg] = [[(bounds[0]+bounds[2])//2,(bounds[1]+bounds[3])//2],
                        [bounds[2]-bounds[0],bounds[3]-bounds[1]],conf]



    #straighten form
    tVec = [locs['qrTr'][0][i]-locs['qrTl'][0][i] for i in range(2)]
    rVec = [locs['qrBr'][0][i]-locs['qrTr'][0][i] for i in range(2)]
    cVec = [locs['qr'][0][i]-locs['logo'][0][i] for i in range(2)]

    tAngle,cAngle = math.degrees(math.atan(tVec[1]/tVec[0])),math.degrees(math.atan(cVec[1]/cVec[0]))
    rAngle = math.degrees(math.atan(rVec[0]/rVec[1]))
    imRot = cAngle * 0.5 + tAngle * 0.25 + rAngle * 0.25

    if abs(imRot) > 2.5:
        rIm = utilityFunctions.rotateImage(im,imRot)

        #rotatePoints
        for k,v in locs.items():
            locs[k] = [utilityFunctions.rotatePoint([im.shape[i]//2 for i in range(2)],v[0],-imRot),v[1],v[2]]
    else:
        rIm = np.copy(im)

    #crop out votetallies
    relativeTallyPos = {
        "logo":[0.6984, 2.6159, 2.5773, 2.5773],
        "qr":[-0.4524, 1.9649, 1.9417, 1.9493],
        "qrBr":[-1.9670, 4.9171, 5.4945, 5.5248],
        "qrTl":[-0.5567, 6.1398, 5.4054, 5.3763],
        "qrTr":[-1.9617, 6.2928, 5.4644, 5.5248]
    }

    cropEstimates = []
    for k,v in relativeTallyPos.items():
        center,size,conf = locs[k]
        x,y,w,h = v
        estimate = [center[0]+size[0]*x,center[1]+size[1]*y,w*size[0],h*size[1]]
        cropEstimates.append([estimate,conf])

    sumEstimate = [0,0,0,0]
    for est,conf in cropEstimates:
        sumEstimate = [d+est[di]*conf for di,d in enumerate(sumEstimate)]

    cx,cy,cw,ch = [int(d/sum([x[1] for x in cropEstimates])) for d in sumEstimate]

    cropIm = rIm[cy:cy+ch,cx:cx+cw]
    if not (cropIm.shape[0] > 0 and cropIm.shape[1] > 0):
        return False

    # increase the contrast of the image
    ctIm = setImageContrast(cropIm, 0, 255)


    # save the image
    if saveIm:
        cv2.imwrite(f'iebc_forms/imgcrops/{imId}.jpg', ctIm)
    return ctIm








    return im

# endregion


# region UNUSED FUNCTIONS
def cropTallyTheta(im,imId):
    fIm = im[int(im.shape[0]*1/6):int(im.shape[0] * 1/2),int(im.shape[1]*2/3):]
    cv2.imwrite(f'iebc_forms/imgcrops/{imId}.jpg', fIm)
    return True

def readChars(img,crop=False):
    img_path = f'iebc_forms/imgs/{img}.jpg'
    if crop:
        img_path = f'iebc_forms/imgcropsfin/{img}.jpg'

    result = OCR.ocr(img_path, cls=True)
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
def getVoterTallies(root, county, pdf, dgModel,qrModel,method="epsilon"):
    filename = pdf.split(".")[0]
    im = pdfToImage(filename, root, county)

    # crop out the general location of the vote tallies on the form
    imIdx = filename.split("_")[3]
    if os.path.exists(f"iebc_forms/imgcrops/{imIdx}.jpg") and method == "epsilon":
        cropIm = cv2.imread(f"iebc_forms/imgcrops/{imIdx}.jpg")
    elif method == "epsilon":
        cropIm = cropTallyEpsilon(im,imIdx,qrModel)
    elif method == "gamma":
        cropIm = cropTallyGamma(im,imIdx)

    if not isinstance(cropIm,np.ndarray):
        return False

    inference = dgModel.inferImage(cropIm, imIdx, "imgscropsfin")
    votes = processRawData(inference)
    if not len(votes):
        print('Failed inference ',method)
        return False
    return [imIdx,votes]


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
        pid = utilityFunctions.padVal(v[keys.index("pollingStationId")].strip(),15)
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



def processRawData(inference):
    if len(inference) < 12:
        return []
    return getEachLineBeta(inference)

def addToVotes(votes,newVotes):
    order = ["raila","ruto","waihiga","wajackoyah"]
    for vi,v in enumerate(newVotes[:len(order)]):
        votes[order[vi]] += v

def iterateForms(formsPath = "I:\IEBC_DATA\FORM34A"):
    global CANDIDATES
    dgModel = YOLOModel("mnist")
    qrModel = YOLOModel("qr")

    #'''
    if os.path.exists("iebc_forms/tallyData/voteTallies.p"):
        voteTallies = pickle.load(open("iebc_forms/tallyData/voteTallies.p", "rb"))
    else:
        voteTallies = {cd: 0 for cd in CANDIDATES}

    if os.path.exists("iebc_forms/tallyData/allVotes.p"):
        allVotes = pickle.load(open("iebc_forms/tallyData/allVotes.p", "rb"))
    else:
        allVotes = dict()
    for county in os.listdir(formsPath):
        print(f"Processing {county}")
        forms = os.listdir(os.path.join(formsPath,county))
        totalForms = len(forms)
        for fi,form in enumerate(forms):
            if form.split("_")[3] in allVotes.keys():
                print(f"form {fi + 1} out of {totalForms} already exists")
                LOSS_RATE[0] += 1
                continue
            for cropMethod in ["epsilon","gamma"]:
                ret = getVoterTallies(formsPath, county, form, dgModel,qrModel,method=cropMethod)
                if isinstance(ret,list):
                    fId,fVotes = ret
                    break
            if ret == False:
                LOSS_RATE[1]+=1
                continue
            LOSS_RATE[0]+=1
            allVotes[fId] = fVotes

            for ti, tally in enumerate(fVotes[1][-4:]):
                voteTallies[CANDIDATES[ti]] += tally

            if fi == len(forms)-1 or fi%25 == 0:
                pickle.dump(voteTallies, open("iebc_forms/tallyData/voteTallies.p", "wb"))
                pickle.dump(allVotes, open("iebc_forms/tallyData/allVotes.p", "wb"))
            print(f"processed form {fi+1} out of {totalForms}")

        #'''

    #save data
    compileTallyData(allVotes)

    print(f"primary loss rate is {LOSS_RATE}")
    print(f"Final tallies are {voteTallies}")

    print("done")
# endregion

if __name__ == '__main__':
    LOSS_RATE = [0, 0]
    OCR = PaddleOCR(use_angle_cls=True, lang='en',show_log = False,use_gpu=True)
    iterateForms()
