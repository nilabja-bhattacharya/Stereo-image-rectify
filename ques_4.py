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


def rectify_images(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    pts1 = []
    pts2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    F,mask= cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    pts1 = pts1[:,:][mask.ravel()==1]
    pts2 = pts2[:,:][mask.ravel()==1]

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    p1fNew = pts1.reshape((pts1.shape[0] * 2, 1))
    p2fNew = pts2.reshape((pts2.shape[0] * 2, 1))

    retBool ,rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(p1fNew,p2fNew,F,img1.shape[:2])

    dst11 = cv2.warpPerspective(img1,rectmat1, img1.shape[:2])
    dst22 = cv2.warpPerspective(img2,rectmat2, img2.shape[:2])
    #plt.imshow(dst22)
    return dst11, dst22

def draw_matches(img1, img2,match_points = 1, flag=False):
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
    if match_points == 0:
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:0], None,flags=2)
    else:
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:], None,flags=2)
    #plt.imshow(img3)
    return img3

if __name__ == '__main__':
    path = input("Enter path to image:")
    img1 = cv2.imread(path)
    h,w = (img1.shape[:2])
    cropped_img11 = img1[0:h, 0:w//2]
    cropped_img21 = img1[0:h, w//2:w]
    i31,i41 = rectify_images(cropped_img11,cropped_img21)
    i51 = draw_matches(i31,i41,flag=True)
    #plt.imshow(i51)
    matched_imgr1 = draw_matches(i31, i41, 1000)
    #plt.imshow(matched_imgr1)
    cv2.imwrite('dense_sift_result_rectified.jpg', matched_imgr1)
    i11r = cv2.cvtColor(i31, cv2.COLOR_BGR2GRAY)
    i21r = cv2.cvtColor(i41, cv2.COLOR_BGR2GRAY)
    res1r = intensity_based_matching(i11r, i21r, i51)
    cv2.imwrite('intensity_result_rectified.jpg', res1r)
    #plt.imshow(res1r)