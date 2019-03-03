import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy import signal
from skimage.measure import compare_ssim as ssim
plt.rcParams['figure.figsize'] = 15, 8

def intensity_based_matching(img1, img2, img3, window_size=64):
    img4 = img3.copy()
    window_size = window_size
    for i in range(0,img1.shape[0]-window_size, window_size):
        for j in range(0, img1.shape[1]-window_size, window_size):
            temp = img1[i:i+window_size, j:j+window_size]
            #print(temp)
            #plt.imshow(temp)
            res = cv2.matchTemplate(img2, temp, cv2.TM_CCORR_NORMED)
            #print(res)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            mid1 = (max_loc[0] + window_size//2+w//2, max_loc[1] + window_size//2)
            mid2 = (i + window_size//2, j + window_size//2)
            #print(mid1, mid2)
            cv2.circle(img4,mid1, 2, (0,0,255), -1)
            cv2.circle(img4,mid2, 2, (0,0,255), -1)
            cv2.line(img4, mid1, mid2, (0, 255, 0), thickness=1, lineType=8)
            #plt.imshow(img4)
            #cv2.drawMatches(cropped_img1,mid2,cropped_img2,mid1,None,img3,flags=2)
            #break
        #break
    return img4

if __name__ == '__main__':
    path = input("Enter path to image:")
    img1 = cv2.imread(path)
    h,w = (img1.shape[:2])
    cropped_img11 = img1[0:h, 0:w//2]
    cropped_img21 = img1[0:h, w//2:w]
    i11 = cv2.cvtColor(cropped_img11, cv2.COLOR_BGR2GRAY)
    i21 = cv2.cvtColor(cropped_img21, cv2.COLOR_BGR2GRAY)
    res1 = intensity_based_matching(i11, i21, img1)
    cv2.imwrite('intensity_result.jpg', res1)