from yolo import YOLO
from PIL import Image
import numpy as np
import glob
import cv2

yolo = YOLO()

imgPath = "img/img.jpg"
image = Image.open(imgPath)
r_image = yolo.detect_image(image)
img = cv2.cvtColor(np.asarray(r_image),cv2.COLOR_RGB2BGR) 
cv2.imwrite("01.jpg", img)
cv2.imshow("img",img)
cv2.waitKey(0)  
