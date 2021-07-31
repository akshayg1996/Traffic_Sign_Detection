import numpy as np
import cv2

img = cv2.imread('sign1.png',0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#lab_planes[0] = clahe.apply(lab_planes[0])

#lab = cv2.merge(lab_planes)

cl1 = clahe.apply(img)

#cl2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cl1 = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
cv2.imwrite('output.png',cl1)
