import cv2,os,random,time
import numpy as np

import utilityFunctions

def createQRCode(size,padding = 0.2):
    black,white = int(255-random.expovariate(0.07)),int(random.expovariate(0.07))
    qrPadding = int(padding*size)
    qr = np.zeros([size+qrPadding*2,size+qrPadding*2],dtype=np.uint8)
    qr[qrPadding:qrPadding+size,qrPadding:qrPadding+size] = white

    #random qr grid
    for y in range(1,31):
        for x in range(1,31):
            qr[int(y/32 * size)+qrPadding:int((y+1)/32 * size)+qrPadding,
            int(x/32 * size)+qrPadding:int((x+1)/32 * size)+qrPadding] = \
            black if random.random() < random.normalvariate(0.5,0.1) else white


    crnSize = int(size*int(random.normalvariate(11,0.7))/32)
    nests = [random.normalvariate(2,0.3),random.normalvariate(3.5,0.3)]
    corner = np.full([crnSize,crnSize],white,dtype=np.uint8)


    corner[int(size*1/32):int(crnSize-size*1/32),int(size*1/32):int(crnSize-size*1/32)] = black
    corner[int(size*(nests[0])/32):int(crnSize-size*(nests[0])/32),
    int(size*(nests[0])/32):int(crnSize-size*(nests[0])/32)] = white
    corner[int(size*(nests[1])/32):int(crnSize-size*(nests[1])/32),
    int(size*(nests[1])/32):int(crnSize-size*(nests[1])/32)] = black



    #place the corners
    crnLocations = []
    for x,y in [(0,0),(1,0),(1,1)]:
        xl,yl = qrPadding+x*(size-crnSize-1),qrPadding+y*(size-crnSize-1)
        qr[yl:yl+crnSize,xl:xl+crnSize] = corner
        crnLocations.append([xl,yl,crnSize,crnSize])

    return qr,crnLocations

def createForm(ftemplate,formBg,lg,noiseMaps):
    #layer images to create form
    lgbg = np.clip(cv2.blur(formBg,(5,5)).astype(np.uint16)+random.randint(100,200),0,255).astype(np.uint8)
    rbg = cv2.resize(lgbg[360:3100, 300:2100, 0].astype(np.uint8), (ftemplate.shape[1], ftemplate.shape[0]))
    form = np.min([rbg,ftemplate],axis=0)
    allcols = np.unique(form)
    fbgCol = random.choice([x for x in range(0,256) if x not in allcols])

    #create train data
    qr,crnLocs = createQRCode(80,padding=0)

    #paste train data onto form
    start,gap = np.random.randint(200,370),np.random.randint(70,100)
    lgx,lgy,lgw,lgh = start,20,lg.shape[1],lg.shape[0]
    qrx,qry,qrw,qrh = start+gap,10,qr.shape[1],qr.shape[0]
    form[lgy:lgy+lgh,lgx:lgx+lgw] = lg
    form[qry:qry+qrh,qrx:qrx+qrw] = qr
    qrLoc,lgLoc = [qrx,qry,qrw,qrh],[lgx,lgy,lgw,lgh]


    #add noise to base form
    noiseMul = random.random()*0.3 + 0.1
    formNoise = cv2.blur(np.random.normal(0,noiseMul,form.shape),(3,3))
    fk = random.choice([0,1,2])*2+1
    form = np.clip(cv2.blur(form,(fk,fk)).astype(np.int32) + formNoise*form,0,255).astype(np.uint8)


    #transform form
    fbgsize = [700,700]
    fsize = [600,600]
    bform = np.full(fbgsize,fbgCol,dtype=np.uint8)
    fLoc = [int(fbgsize[i]/2-fsize[i]/2+random.randint(-25,25)) for i in range(2)]
    fRot = random.normalvariate(0,7.5)
    bform[fLoc[1]:fLoc[1]+fsize[1],fLoc[0]:fLoc[0]+fsize[0]] =\
        cv2.resize(form[:min(form.shape),:min(form.shape)],tuple(fsize))
    cform = utilityFunctions.rotateImage(bform,fRot,fbgCol)


    #tranform points
    bounds = []
    locKeys = ["logo","qr","qrTl","qrTr","qrBr"]
    crnLocs = [[cd[0]+qrx,cd[1]+qry,cd[2],cd[3]] for cd in crnLocs]
    for x,y,w,h in [lgLoc,qrLoc,*crnLocs]:
        ax,ay = utilityFunctions.rotatePoint([d//2 for d in fbgsize],(fLoc[0]+x+w/2,fLoc[1]+y+h/2),-fRot)
        aw,ah = w*1.25,h*1.25
        bounds.append([ax/fbgsize[0],ay/fbgsize[1],aw/fbgsize[0],ah/fbgsize[1]])
    fBounds = [f"{cl} {' '.join([str(round(x, 6)) for x in bd])}\n" for cl, bd in enumerate(bounds)]


    #add noise to form
    bgNoiseMap = utilityFunctions.pickNoiseMap(noiseMaps)
    fform = (((cform == fbgCol) * fbgCol - ((bgNoiseMap - np.min(bgNoiseMap)) *
            (cform == fbgCol) * fbgCol ))+(cform != bgNoiseMap)*cform).astype(np.uint8)

    #rImg = utilityFunctions.viewBounds(fform,fBounds)

    return fform,fBounds


def createDataset(n,split = [0.8,0.15,0.05]):
    #set up locations
    reqPaths = [["qr"],["images","labels"],["trainSet","valSet","testSet"]]
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
    counts = [0,0,0]
    noiseMaps = utilityFunctions.createNoiseMaps(50,700,700) #change this if the size of the imgs

    # import og images
    formTemplate = cv2.imread("configData/FormTemplate.png", cv2.IMREAD_GRAYSCALE)
    formBackground = cv2.imread("configData/talliesBg.png")
    iebcLogo = cv2.imread("configData/iebcLogo.png", cv2.IMREAD_GRAYSCALE)

    for b in range(n):
        for si in range(len(split)):
            if b/n < sum(split[:si+1]):
                thisSet = reqPaths[2][si]
                break

        if b%(n//10) == 0:
            print(f"Created {b} out of {n} samples of training data")
        #im,bounds = createDigitGrid(chars)
        im,bounds = createForm(formTemplate,formBackground,iebcLogo,noiseMaps)
        impath = f"{reqPaths[0][0]}/{reqPaths[1][0]}/{thisSet}/{utilityFunctions.padVal(counts[si]+1,4)}.jpg"
        lbpath = f"{reqPaths[0][0]}/{reqPaths[1][1]}/{thisSet}/{utilityFunctions.padVal(counts[si]+1,4)}.txt"
        cv2.imwrite(impath,cv2.cvtColor(im,cv2.COLOR_GRAY2RGB))
        with open(lbpath,"w") as file:
            for bd in bounds:
                file.write(bd)
        counts[si]+=1


    print("done")


if __name__ == '__main__':
    createDataset(1000)