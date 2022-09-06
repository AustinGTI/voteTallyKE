import cv2,os,math,random,csv,pickle
import numpy as np
from trainDataGeneration import generateDataset


#Generating training data specific to the general voter tally template to be trained on a YOLO v3 model

#The MNIST numbers dataset is used to simulate groups of 3 digit numbers across 4/5 rows seperated by lines which are
# then used to train the model including an 11th class representing cancelled digits (with a stroke running through)

#5,000 training data - 6 digits each

#the 11 possible classes
import utilityFunctions

CLASSES = list("0123456789d")

#utility funcs
#pad a numeric string with zeros for aesthetic file naming reasons
def padString(val,l,pad = "0"):
    vstr = str(val)
    if len(vstr) > l:
        return vstr
    return pad*(l - len(vstr))+vstr
#.........

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

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


#create a grid of digits that simulates voter tally figures
def createDigitGrid(allImgs):
    global CLASSES
    lines = 4 + (1 if random.random() > 0.9 else 0) #number of rows (4 candidates) (occasionally 5 for variation)
    perLine = 3             #digits per row/line
    pd = 15             #padding along the border of the grid
    probCancel = 0.05       #the probability of a number being cancelled
    offsetStd = 0.05        #digit position offset standard deviation
    pDigits = CLASSES[:-1]   #possible digits
    myDigits = [random.choice(pDigits) for i in range(lines*perLine)]   #random set of digits (rows * perLine)
    myChars = [cv2.imread(random.choice(allImgs[nm]),0) for nm in myDigits]          #random figure for each digit
    dgDims = myChars[0].shape        #base dimensions of the digit img
    finChars = []
    maxLi = 0
    dividers = []
    crossLines = []
    edgeLines = []
    for li in range(lines):
        maxCi = 0
        tmaxLi = maxLi
        lh = 0
        for ci in range(perLine):
            timg = myChars.pop(0)
            x,y,w,h = cv2.boundingRect(cv2.findNonZero(timg))
            lh = max(lh,h)
            cimg = timg[y:y+h,x:x+w]
            ipos = [maxLi * random.normalvariate(1,offsetStd/3) + pd,
                    maxCi * random.normalvariate(0.8,offsetStd) + pd]
            tmaxLi = max(tmaxLi,ipos[0]+cimg.shape[0])
            maxCi = max(maxCi,ipos[1]+cimg.shape[1])
            finChars.append([cimg,[max(0,int(round(d))) for d in ipos]])
        for ci in range(perLine):
            finChars[-ci-1][0][0] += int((lh - finChars[-ci-1][0].shape[0])/2)


        if random.random() < probCancel:
            crossPos = maxLi + (tmaxLi-maxLi)/2
            crossLines.append(int(crossPos + random.normalvariate(0,(tmaxLi-maxLi)*0.05)))
            for nm in range(li*perLine,li*perLine+perLine):
                myDigits[nm] = "d"
        else:
            edgeLines.append([int(maxLi + (tmaxLi - maxLi) / 2), int(maxCi)])

        maxLi = tmaxLi
        dividers.append(int(maxLi))
    finGrid = np.zeros([max(finChars,key=lambda x:x[1][0])[1][0]+dgDims[0]+pd*2,
                        max([pos[1]+c.shape[1]+1+pd*2 for c,pos in finChars])],
                       dtype=np.uint8)
    for c,pos in finChars:
        y,x = pos
        fh,fw = finGrid.shape
        ch,cw = c.shape
        finGrid[y:y+ch,fw-1-cw-x:fw-1-x] = np.max([c,finGrid[y:y+ch,fw-1-cw-x:fw-1-x]],axis=0)


    #add edge lines
    for yedge,edge in edgeLines:
        eline = buildLine(1,finGrid.shape[1] - edge,completion=1,thickness=15,noise=0.1,full=False)
        finGrid[yedge-2:yedge-2+eline.shape[0],0:finGrid.shape[1]-edge] =\
            np.max([eline,finGrid[yedge-2:yedge-2+eline.shape[0],0:finGrid.shape[1]-edge]],axis = 0)

    #add dividing lines
    completion = min(1,random.normalvariate(0.8,0.1))
    for div in dividers:
        dline =buildLine(1,finGrid.shape[1],thickness=20,completion=completion,noise=0.1)
        dstart = max(0,int(div-dline.shape[0]*2/3))
        dend = max(0,int(div-dline.shape[0]*2/3)) + dline.shape[0]
        finGrid[dstart:dend] = np.max([dline,finGrid[dstart:dend]],axis=0)

    #crossed out vals
    for cross in crossLines:
        cline = buildLine(1,finGrid.shape[1],thickness=15,completion=1,noise=0.05)
        finGrid[cross - 1:cross - 1 + cline.shape[0]] = np.max([cline, finGrid[cross - 1:cross - 1 + cline.shape[0]]], axis=0)

    #noise
    '''noise = np.clip(np.random.normal(0, 30, finGrid.shape),0,255)
    ffinGrid = np.max([finGrid,noise],axis=0).astype(np.uint8)'''

    #bounding boxes
    #YOLO V3 bounding boxes
    bBounds = [[n,[finGrid.shape[1]-1-fc[0].shape[1]-fc[1][1]-pd,fc[1][0]-pd,
                   fc[0].shape[1]+pd*2,fc[0].shape[0]+pd*2]]
               for n,fc in zip(myDigits,finChars)]

    #YOLO V5 bounding boxes
    finBounds = []
    for cl,bd in bBounds:
        cx,cy = bd[0]+bd[2]/2,bd[1]+bd[3]/2
        normCx,normCy = round(cx/finGrid.shape[1],6),round(cy/finGrid.shape[0],6)
        normW,normH = round(bd[2]/finGrid.shape[1],6),round(bd[3]/finGrid.shape[0],6)
        finBounds.append(" ".join([str(x) for x in [CLASSES.index(cl),normCx,normCy,normW,normH]])+"\n")

    #threshhold
    _, thfGrid = cv2.threshold(finGrid, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #thicken
    ffinGrid = cv2.dilate(thfGrid,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),
                          iterations = math.floor(random.expovariate(1.5)))


    #show me the bounds
    '''for chr,b in bBounds:
        cv2.rectangle(finGrid,(b[0],b[1]),(b[0]+b[2],b[1]+b[3]),(127),10)'''


    return ffinGrid,finBounds







def createDigitGridAlpha(*args):
    noiseMaps,templatesIms,allImgs = args
    talliesBg,IEBCStamp = templatesIms
    allImgs = allImgs[0]
    nps, chnps = noiseMaps
    gw,gh = [1000,1000]
    yPad = int(random.normalvariate(150,30))
    xPad = int(random.normalvariate(100,20))

    edgeWidth = xPad-70
    edgeGrad = random.normalvariate(0,0.02)
    upperTxtPos = yPad-150

    edgeDetail, edgeScale = int(random.normalvariate(16,2)), 1
    ofst = int(min(5,max(1,random.normalvariate(3,0.5))))
    lHeight = random.normalvariate(100,7)
    grid = np.zeros([gh,gw],dtype=np.uint8)

    #add background pattern
    bgOffset = [random.randrange(-30, 30), random.randrange(-30, 30)]
    x, y, tx, ty = 300 + bgOffset[0], 450 + bgOffset[1], 300 + 1000 + bgOffset[0], 450 + 1000 + bgOffset[1]
    bg = cv2.resize((255 - talliesBg[x:tx, y:ty]) // 2, (gw, gh))
    bgK = int(random.normalvariate(10,2))*2+1
    bg = cv2.GaussianBlur(bg,(bgK,bgK),0)
    grid = np.max([grid,bg],axis=0)

    # add edge background
    if edgeWidth > 0:
        p1 = [grid.shape[1] - (edgeWidth + edgeGrad * grid.shape[0]), 0]
        p2 = [grid.shape[1] - 1, 0]
        p3 = [grid.shape[1] - 1, grid.shape[0] - 1]
        p4 = [grid.shape[1] - (edgeWidth - edgeGrad * grid.shape[0]), grid.shape[0] - 1]
        pts = np.array([p1, p2, p3, p4], dtype=np.int32)
        cv2.fillPoly(grid, [pts], (random.randrange(30, 220)))

    #add text above form
    cv2.putText(grid,"Code : 332423",(grid.shape[1]-800-yPad,upperTxtPos),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)

    #placing a stamp
    if random.random() < 0.1:
        IEBCStamp = 255-cv2.resize(IEBCStamp,(250,250))
        utilityFunctions.rotateImage(IEBCStamp,random.normalvariate(0,20),255)
        sx,sy = random.randint(0,grid.shape[1]-IEBCStamp.shape[1]),random.randint(0,grid.shape[0]-IEBCStamp.shape[0])
        grid[sy:sy+IEBCStamp.shape[0],sx:sx+IEBCStamp.shape[1]] = \
            np.max([IEBCStamp,grid[sy:sy+IEBCStamp.shape[0],sx:sx+IEBCStamp.shape[1]]],axis=0)

    l = 0
    #vertical line
    xPos = int(gw-xPad)
    cv2.line(grid,(xPos,int(yPad)),(xPos,gh-1),(200),ofst*2+1)
    while True:
        yPos = int(yPad+lHeight*l)
        xPosV = int(gw-xPad)
        #horizontal line
        cv2.line(grid,(0,yPos),(xPosV,yPos),(200),ofst*2-1)
        #edge curve
        for edge in range(edgeDetail):
            cv2.line(grid,(xPosV-ofst-((edgeDetail*edgeScale)-(edge*edgeScale)),yPos+ofst),
                     (xPosV-ofst,yPos+(edge*edgeScale)+ofst),(200),2)
            if l != 0:
                cv2.line(grid,(xPosV-ofst-((edgeDetail*edgeScale)-(edge*edgeScale)),yPos-ofst),
                     (xPosV-ofst,yPos-(edge*edgeScale)-ofst),(200),2)
        if l >= 5:
            cv2.line(grid,(0,yPos+int(lHeight*0.7)),(int(gw-xPad),yPos),(200),7)
        l+=1

        if (yPad+lHeight*l) > gh:
            break






    grid = cv2.blur(np.random.normal(1,0.5,(gh,gw))*grid,(5,5))
    grid  = np.clip(grid.astype(np.uint16)*cv2.blur(np.random.normal(1,0.4,(gw,gh)),(7,7)),0,255)



    noiseMap,charNoiseMap = utilityFunctions.pickNoiseMap(nps),utilityFunctions.pickNoiseMap(chnps)


    grid += grid*noiseMap*random.normalvariate(0.6,0.2)


    global CLASSES
    lines = 4 + (1 if random.random() > 0.9 else 0)  # number of rows (4 candidates) (occasionally 5 for variation)
    perLine = 3  # digits per row/line



    potentialVotes = [
        int(random.normalvariate(250,60)),#raila
        int(random.normalvariate(250,60)),#ruto
        int(random.expovariate(0.75)),#waihiga
        int(random.expovariate(0.7))#wajackoyah
    ]
    if lines == 5:
        potentialVotes.insert(0,potentialVotes[0])
    myDigits = [d for v in potentialVotes for d in reversed(list(padString(v,3)))]

    #myDigits = [random.choice(pDigits,) for i in range(lines * perLine)]  # random set of digits (rows * perLine)
    baseChars = [cv2.imread(f"../{random.choice(allImgs[nm])}", 0) for nm in myDigits]
    bx,by = (gw-xPad,yPad+(lHeight if lines == 4 else 0))
    dDist = min(0.4,abs(random.normalvariate(0,0.3)))
    bounds = []


    for li in range(lines):
        boffset = 0
        for ci in range(perLine):
            timg = baseChars.pop(0).astype(np.float64)
            timg+=timg*np.rot90(charNoiseMap,random.randrange(0,3))
            tdig = myDigits.pop(0)
            x, y, w, h = cv2.boundingRect(cv2.findNonZero(timg))
            scale = int(lHeight - random.expovariate(1))/lHeight
            cimg = cv2.resize(timg[y:y + h, x:x + w],(0,0),fx = scale,fy = scale)

            ch,cw = cimg.shape
            idx,idy = bx-boffset-cw//2,by+lHeight*li+lHeight//2
            cx,cy = map(lambda x:int(round(x)),
                        [random.normalvariate(idx,lHeight//20),random.normalvariate(idy,lHeight//20)])
            grid[cy-ch//2:cy-ch//2+ch,cx-cw//2:cx-cw//2+cw] = \
                np.max([cimg,grid[cy-ch//2:cy-ch//2+ch,cx-cw//2:cx-cw//2+cw]],axis=0)
            bounds.append([CLASSES.index(tdig),[cx/gw,cy/gh,(cw*1.2)/gw,(ch*1.2)/gh]])

            boffset += cw - cw*dDist
        ly = int(by+lHeight*li+lHeight/2)
        '''cv2.line(grid,(0,ly),(cx-cw//2,ly),
                 (int(random.normalvariate(200,19))),
                 int(random.normalvariate(10,2)))'''
        edgeLine = buildLine(1,cx,20,1,full=False)
        edgeLine+=edgeLine*cv2.resize(np.rot90(charNoiseMap,random.randrange(0,3)),
                                      (edgeLine.shape[1],edgeLine.shape[0]))
        eh,ew = edgeLine.shape
        grid[ly-eh//2:ly-eh//2+eh,:ew] = np.max([
            grid[ly - eh // 2:ly - eh // 2 + eh, :ew],
            edgeLine
        ],axis=0)
    lBounds = [f"{cl} {' '.join([str(round(x,6)) for x in bd])}\n" for cl,bd in bounds]
    fingrid = 255-np.clip(grid,0,255).astype(np.uint8)
    return fingrid,lBounds



def buildLine(axis,length,thickness = 3,completion = 0.8,gapSize = 0.1,noise = 0.1,full = True):
    dims = [thickness*2+1,length] if axis == 1 else [length,thickness*2+1]
    line = np.zeros(dims,dtype=np.uint8)
    padding = 0
    if not full:
        padding = length // 10
    gaps = []
    gs = math.ceil(gapSize*(length-padding*2))
    for i in range(length - padding * 2 + gs):
        chance = random.random()
        if chance > completion ** (1 / gs):
            l, r = max(0, i - gs), min(length - 1, i)
            for g in range(l, r):
                gaps.append(g)

    if axis == 0:
        u = (padding,line.shape[1]//2)
        v = (line.shape[0] - padding - 1,line.shape[1]//2)


    else:
        u = (line.shape[0] // 2,padding)
        v = (line.shape[0] // 2,line.shape[1] - padding - 1)


    cv2.line(line,(u[1],u[0]),(v[1],v[0]),255,thickness//2)
    for g in gaps:
        if axis == 1:
            line[:,padding+g] = 0
        else:
            line[g+padding,:] = 0
    k = max(int(length*noise)+(1 if int(length*noise)%2 == 0 else 0),3)

    line = cv2.GaussianBlur(line,(5,5),0)
    fline = np.clip(line*cv2.blur(1-np.random.normal(0,0.3,line.shape),(5,5)),0,255)

    return fline









def extractData():
    global CLASSES
    file_path = f"../mnistData/mnist_train.csv"
    csvfile = csv.reader(open(file_path,"r"))
    data = {v:[] for v in CLASSES[:-1]}
    i = 0
    for row in csvfile:
        if i%1000 == 0:
            print(f"created {i} training images")
        i+=1
        if row[0] == "label":
            continue
        img = np.array(row[1:],dtype=np.uint8).reshape([28,28])
        fimg = cv2.resize(img,(0,0),fx=4,fy=4)
        impath = f"mnistData/rawTrainData/{str(row[0])}_{padString(len(data[str(row[0])]),4)}.jpg"
        data[str(row[0])].append(impath)
        cv2.imwrite(impath,fimg)

    file_path = f"../mnistData/mnist_test.csv"
    csvfile = csv.reader(open(file_path, "r"))
    for row in csvfile:
        if i%1000 == 0:
            print(f"created {i} training images")
        i+=1
        if row[0] == "label":
            continue
        img = np.array(row[1:],dtype=np.uint8).reshape([28,28])
        fimg = cv2.resize(img,(0,0),fx=4,fy=4)
        impath = f"mnistData/rawTrainData/{str(row[0])}_{padString(len(data[str(row[0])]),4)}.jpg"
        data[str(row[0])].append(impath)
        cv2.imwrite(impath,fimg)

    pickle.dump(data, open(f"../mnistData/trainImPaths.p", "wb"))

def generateAnnotationLine(path,bbounds):
    global CLASSES
    fpath = path
    fBounds = []
    for bound in bbounds:
        fBounds.append(f"{bound[1][0]},{bound[1][1]},{bound[1][0]+bound[1][2]},{bound[1][1]+bound[1][3]},"
                       f"{CLASSES.index(bound[0])}")
    fline = f"{fpath} {' '.join(fBounds)}\n"
    return fline



def createDataset(n,split = [0.8,0.15,0.05]):
    #set up locations
    reqPaths = [["mnist"],["images","labels"],["trainSet","valSet","testSet"]]
    for root in reqPaths[0]:
        if not os.path.exists(root):
            os.mkdir(root)
        for mid in reqPaths[1]:
            if not os.path.exists(os.path.join(root,mid)):
                os.mkdir(os.path.join(root,mid))
            for leaf in reqPaths[2]:
                if not os.path.exists(os.path.join(root,mid,leaf)):
                    os.mkdir(os.path.join(root,mid,leaf))


    global CLASSES
    chars = pickle.load(open(f"../mnistData/trainImPaths.p", "rb"))
    counts = [0,0,0]
    noiseMaps = utilityFunctions.createNoiseMaps(100, 1000, 1000) #change this if the size of the imgs
    charNoiseMaps = utilityFunctions.createNoiseMaps(100,112,112,[10],[1],div=4)
    for b in range(n):
        for si in range(len(split)):
            if b/n < sum(split[:si+1]):
                thisSet = reqPaths[2][si]
                break

        if b%(n//10) == 0:
            print(f"Created {b} out of {n} samples of training data")
        #im,bounds = createDigitGrid(chars)
        im,bounds = createDigitGridAlpha(chars,noiseMaps,charNoiseMaps)
        impath = f"mnist/images/{thisSet}/{padString(counts[si]+1,4)}.jpg"
        lbpath = f"mnist/labels/{thisSet}/{padString(counts[si]+1,4)}.txt"
        cv2.imwrite(impath,cv2.cvtColor(im,cv2.COLOR_GRAY2RGB))
        with open(lbpath,"w") as file:
            for bd in bounds:
                file.write(bd)
        counts[si]+=1


    print("done")


if __name__ == '__main__':
    imPaths = pickle.load(open(f"../mnistData/trainImPaths.p", "rb"))
    generateDataset(
        n=100,
        generator=createDigitGridAlpha,
        folder="mnist",
        noiseMaps=[[10, 1000,1000],[10,112,112,[10],[1],4]],
        templatePaths=['../configData/talliesBg.png','../configData/IEBCStamp.png'],
        extraData=[imPaths]
    )
