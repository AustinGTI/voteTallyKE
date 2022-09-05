import pytesseract,pdf2image
from PIL import Image

import readForm

POPPLER_PATH = "C:/Users/Admin/Documents/poppler-22.04.0/Library/bin"
pdfpath = "C:/Users/Admin//PycharmProjects/voteTally/ALL_FORMS/NAROK_COUNTY" \
          "/1_34_A_033177088100102_Y3KK12CQKIA02VO_20220809_193916.pdf"
pytesseract.pytesseract.tesseract_cmd = r"C:/Users/Admin/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

im = pdf2image.convert_from_path(pdfpath,poppler_path=POPPLER_PATH)[0]
res = pytesseract.image_to_boxes(im)
print("done")