import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy import signal
from skimage.measure import compare_ssim as ssim
plt.rcParams['figure.figsize'] = 15, 8

def ComputeDescriptors(img, step):
    kps = []
    startSize = 8 if step < 8 else step
    for i in range(step, img.shape[0]-step, step):
        for j in range(step, img.shape[1]-step, step):
            for z in range(startSize, startSize*5, startSize):
                kps.append(cv2.KeyPoint(float(i), float(j), float(z)))
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.compute(img,kps)

def draw_matches(img1, img2,flag=False):
    if flag == True:
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
    else:
        kp1, des1 = ComputeDescriptors(img1,8)
        kp2, des2 = ComputeDescriptors(img2,8)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    #bf = cv2.BFMatcher()
    #matches = bf.knnMatch(des1[1],des2[1], k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    #print(matches)
    good = sorted(good, key = lambda x:x.distance)
    #print(kp1)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:], None,flags=2)
    #plt.imshow(img3)
    return img3

if __name__ == '__main__':
    path = input("Enter path to image:")
    img1 = cv2.imread(path)
    h,w = (img1.shape[:2])
    cropped_img11 = img1[0:h, 0:w//2]
    cropped_img21 = img1[0:h, w//2:w]
    matched_img1 = draw_matches(cropped_img11, cropped_img21)
    cv2.imwrite('dense_sift_result.jpg', matched_img1)