import csv
from tqdm import tqdm
from PyPDF2 import PdfFileWriter, PdfFileReader
import pdf2image,os,cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

import neuralNet
from utilityFunctions import padVal

POPPLER_PATH = "C:/Users/Admin/Documents/poppler-22.04.0/Library/bin"

def splitPdfPages(path="targetData/Form34C.pdf",target="targetData/pdfPages"):
    inputpdf = PdfFileReader(open(path, "rb"))

    for i in range(inputpdf.numPages):
        if i%100 == 0:
            print(f"extracted {i+1} pages out of {inputpdf.numPages}")
        output = PdfFileWriter()
        output.addPage(inputpdf.getPage(i))
        with open(f'{target}/Page_{padVal(i+1,5)}.pdf', "wb") as outputStream:
            output.write(outputStream)


def dividePage(im,axis):
    divs = np.mean(im, axis=axis)
    inEntry = True
    entries = [[0]]
    for ri, row in enumerate(divs):
        if ri == len(divs) - 1 and inEntry:
            entries[-1].append(ri)

        if row < 50:
            if inEntry:
                entries[-1].append(ri)
                inEntry = False
        else:
            if not inEntry:
                entries.append([ri])
                inEntry = True
    return entries

def calculateRowsColumns(im,tBox):
    gIm = cv2.cvtColor(im[tBox[1]:tBox[3], tBox[0]:tBox[2]], cv2.COLOR_BGR2GRAY)
    tIm = cv2.erode(cv2.threshold(gIm, 150, 255, type=cv2.THRESH_BINARY)[1], np.ones([3, 3]))

    rows = dividePage(tIm,1)
    columns = dividePage(tIm,0)
    return rows,columns


def extractPdfData(path="targetData/pdfPages"):
    #ocr = PaddleOCR(use_angle_cls=True, lang='en',show_log = False)
    model = neuralNet.YOLOModel("mnist")
    rowEntries,columnEntries = None,None
    bBox = [185,495,2128,3113]
    columns = ["CountyCode","CountyName","ConstituencyCode","ConstituencyName","PollingStationCode",
               "PollingStationName","RegisteredVoters","Raila Odinga","William Ruto","David Waihiga",
               "George Wajackoyah","Total Valid Votes","Rejected Ballots"]
    pagesNo = len(os.listdir(path))
    for pi,pdf in enumerate(os.listdir(path)):
        print(f"Page {pi+1} out of {pagesNo}")
        pageEntries = []
        im = pdf2image.convert_from_path(os.path.join(path,pdf), poppler_path=POPPLER_PATH)[0]
        cvIm = np.rot90(np.array(im)[:, :, ::-1],3)
        maskIm = np.reshape(cv2.dilate(1 - cv2.threshold(cv2.cvtColor(cvIm,cv2.COLOR_BGR2GRAY),
                                              100,255,cv2.THRESH_BINARY)
        [1]/255,np.ones([5,5])),[*cvIm.shape[:2],1])

        filIm = (maskIm*cvIm)+(1-maskIm)*np.full(cvIm.shape,255,np.uint8)

        if rowEntries == None or columnEntries == None:
            rowEntries,columnEntries = calculateRowsColumns(cvIm,bBox)
        for ri,row in tqdm(enumerate(rowEntries)):
            #print(f"Page {pi+1} out of {pagesNo}: Entry {ri+1} out of {len(rowEntries)}")

            rowIm = cvIm[row[0]+bBox[1]:row[1]+bBox[1],bBox[0]:bBox[2]]
            rowData = {col:'' for col in columns}
            for ci,col in [[i,columnEntries[i]] for i in [4,6,7,8,9,10,11,12]]:
                colIm = cv2.resize(rowIm[:,col[0]:col[1]], (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                #value = ocr.ocr(colIm, cls=True)
                inference = model.inferImage(colIm,'000')
                if len(inference) > 0:
                    value = ''.join([v[0] for v in sorted(inference, key=lambda x: x[1][0])])
                    rowData[columns[ci]] = value
            pageEntries.append(rowData)
        with open("targetData/targetTallies.csv",'a',newline='') as file:
            csvfile = csv.writer(file)
            if pi == 0:
                csvfile.writerow(["PollingStationCode",
                                "RegisteredVoters", "Raila Odinga", "William Ruto",
                            "David Waihiga","George Wajackoyah", "Total Valid Votes", "Rejected Ballots"])
            for entry in pageEntries:
                csvfile.writerow([entry[key] for key in ["PollingStationCode",
               "RegisteredVoters","Raila Odinga","William Ruto","David Waihiga",
               "George Wajackoyah","Total Valid Votes","Rejected Ballots"]])
        print("done")


        #results = ocr.ocr(gIm, cls=True)
        
    print("now this")


if __name__ == '__main__':
    extractPdfData()