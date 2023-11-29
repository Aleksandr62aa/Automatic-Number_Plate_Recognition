# https://docs.opencv.org/3.4/d0/d90/group__highgui__window__flags.html
# https://github.com/opencv/opencv/tree/master/data/haarcascades

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
from imutils import contours


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def show_img(img, cmap = None, axis = False):
    plt.imshow(img, cmap=cmap)
    plt.axis(axis)
    plt.show()

def open_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def carplate_detect(image, carplate_rects):
    carplate_overlay = image.copy()   
    for x,y,w,h in carplate_rects:
        cv2.rectangle(carplate_overlay, (x,y), (x+w,y+h), (255, 0, 0), 5)
    return carplate_overlay

def carplate_extract(image, carplate_rects):
    for x,y,w,h in carplate_rects:
        carplate_img = image[y+15:y+h-10 ,x+15:x+w-20]
    return carplate_img

def enlarge_img(image, scale_percent):
    width = int(image.shape[1]*scale_percent/100)
    height = int(image.shape[0]*scale_percent/100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)    
    return resized_image

def main():
    # load a image
    img_path="IMAGE\car3.jpg"
    carplate_img_rgb = open_img(img_path)
    show_img(carplate_img_rgb)

    #Search for a number plate
    carplate_haar_cascade = cv2.CascadeClassifier("haarcascade\haarcascade_plate_number.xml")
    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_img_rgb, scaleFactor=1.1, minNeighbors=5)
    carplate_extract_img = carplate_detect(carplate_img_rgb, carplate_rects)
    show_img(carplate_extract_img)
    
    # Normalizing numbers plate
    carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_rects)
    show_img(carplate_extract_img)
    carplate_extract_img = enlarge_img(carplate_extract_img, 150)
    carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)
    show_img(carplate_extract_img_gray, cmap='gray')
  
    # Recognising text for a number plate
    carplate_text = pytesseract.image_to_string(
        carplate_extract_img_gray, 
        config='--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    carplate_text = str(carplate_text).rstrip()
    print(carplate_text)    
    
    for x,y,w,h in carplate_rects:
        k = w / 180
        cv2.rectangle(carplate_img_rgb, (x-10,y-h), (x+w-10,y), (255, 178, 90), -1)
        cv2.putText(carplate_img_rgb,carplate_text, (x-10, y-15), cv2.FONT_HERSHEY_TRIPLEX, 0.9*k, (0, 0, 0), 1)
    show_img(carplate_img_rgb)

    filename = 'car_Haarcascade_TesseractOCR.jpg'
    carplate_img_bgr = cv2.cvtColor(carplate_img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, carplate_img_bgr) 

if __name__ == '__main__':
    main()