import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy import signal
from skimage.measure import compare_ssim as ssim
plt.rcParams['figure.figsize'] = 15, 8



def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def greedy_matching(img1, img2, img3):
    window_size = 128
    img4 = img3.copy()
    for i in range(0, img1.shape[1]-window_size, window_size):
        for j in range(0, img1.shape[0]-window_size, window_size):
            tmp1 = img1[i:i+window_size, j:j+window_size]
            cor = -1e19
            mt = 0
            for k in range(0, img2.shape[0]-window_size, window_size):
                tmp2 = img2[i:i+window_size, k:k+window_size]
                cort = correlation_coefficient (tmp1, tmp2)
                #cv2.rectangle(img2,(k,i), (k+window_size,i+window_size), 255, 20)
                #print(cort)
                #plt.imshow(img2)
                if cort>cor:
                    #print(k)
                    cor=cort
                    mt = k
            mid1 = (j + window_size//2, i + window_size//2)
            mid2 = (mt + window_size//2+w//2, i + window_size//2)
            #print(mid1, mid2)
            cv2.circle(img4,mid1, 5, (0,0,255), -1)
            cv2.circle(img4,mid2, 5, (0,0,255), -1)
            cv2.line(img4, mid1, mid2, (0, 255, 0), thickness=2, lineType=8)
            #plt.imshow(img4)
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

def draw_matches(img1, img2,match_points = 0, flag=False):
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
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:min(match_points, len(good))], None,flags=2)
    #plt.imshow(img3)
    return img3

def distance_cost_plot(distances):
    im = plt.imshow(distances, interpolation='nearest', cmap='Reds') 
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.colorbar();

def dtw_matching(img1, img2, img3):
    window_size = 128
    img4 = img3.copy()
    for i in range(0, img1.shape[1]-window_size, window_size):
        distances = np.zeros((img1.shape[0]//window_size, img2.shape[0]//window_size))
        for k1 in range(img1.shape[0]//window_size):
            for k2 in range(img2.shape[0]//window_size):
                tmp1 = img1[i:i+window_size, k1*window_size:k1*window_size+window_size]
                tmp2 = img2[i:i+window_size, k2*window_size:k2*window_size+window_size]
                distances[k1,k2] = correlation_coefficient(tmp1, tmp2)
        #distance_cost_plot(distances)
        accumulated_cost = np.zeros((img1.shape[0]//window_size, img2.shape[0]//window_size))
        accumulated_cost[0,0] = distances[0,0]
        for k1 in range(1,img2.shape[0]//window_size):
            accumulated_cost[0,k1] = distances[0,k1] + accumulated_cost[0, k1-1]   
        #distance_cost_plot(accumulated_cost)
        for k1 in range(1, img1.shape[0]//window_size):
            accumulated_cost[k1,0] = distances[k1, 0] + accumulated_cost[k1-1, 0] 
        #distance_cost_plot(accumulated_cost)
        for k1 in range(1, img1.shape[0]//window_size):
            for k2 in range(1, img2.shape[0]//window_size):
                accumulated_cost[k1, k2] = min(accumulated_cost[k1-1, k2-1], 
                                               accumulated_cost[k1-1, k2], accumulated_cost[k1, k2-1]) + distances[k1, k2]
        #distance_cost_plot(accumulated_cost)
        path = [[img2.shape[0]//window_size-1, img1.shape[0]//window_size-1]]
        k1 = img1.shape[0]//window_size-1
        k2 = img2.shape[0]//window_size-1
        while k1>0 and k2>0:
            if k1==0:
                k2 = k2 - 1
            elif k2==0:
                k1 = k1 - 1
            else:
                if accumulated_cost[k1-1, k2] == min(accumulated_cost[k1-1, k2-1], accumulated_cost[k1-1, k2], accumulated_cost[k1, k2-1]):
                    k1 = k1 - 1
                elif accumulated_cost[k1, k2-1] == min(accumulated_cost[k1-1, k2-1], accumulated_cost[k1-1, k2], accumulated_cost[k1, k2-1]):
                    k2 = k2-1
                else:
                    k1 = k1 - 1
                    k2= k2- 1
            path.append([k2, k1])
        path.append([0,0])
        for k in path:
            mid1 = (k[0]*window_size + window_size//2, i + window_size//2)
            mid2 = (k[1]*window_size + window_size//2+w//2, i + window_size//2)
            #print(mid1, mid2)
            cv2.circle(img4,mid1, 5, (0,0,255), -1)
            cv2.circle(img4,mid2, 5, (0,0,255), -1)
            cv2.line(img4, mid1, mid2, (0, 255, 0), thickness=2, lineType=8)
        #plt.imshow(img4)
        #break
    return img4
#         for j in range(0, img1.shape[0]-window_size, window_size):
#             for k in range(0, img2.shape[0]-window_size, window_size):
#                 tmp1 = img1[i:i+window_size, j:j+window_size]
#                 tmp2 = img2[i:i+window_size, k:k+window_size]
#                 cort = correlation_coefficient (tmp1, tmp2)
#                 #cv2.rectangle(img2,(k,i), (k+window_size,i+window_size), 255, 20)
#                 #print(cort)
#                 #plt.imshow(img2)
#                 if cort>cor:
#                     #print(k)
#                     cor=cort
#                     mt = k
#             mid1 = (j + window_size//2, i + window_size//2)
#             mid2 = (mt + window_size//2+w//2, i + window_size//2)
#             #print(mid1, mid2)
#             cv2.circle(img4,mid1, 5, (0,0,255), -1)
#             cv2.circle(img4,mid2, 5, (0,0,255), -1)
#             cv2.line(img4, mid1, mid2, (0, 255, 0), thickness=2, lineType=8)
#             #plt.imshow(img4)
#     return img4

if __name__ == '__main__':
    path = input("Enter path to image:")
    img1 = cv2.imread(path)
    h,w = (img1.shape[:2])
    cropped_img11 = img1[0:h, 0:w//2]
    cropped_img21 = img1[0:h, w//2:w]
    i31,i41 = rectify_images(cropped_img11,cropped_img21)
    i51 = draw_matches(i31,i41, flag=True)

    i61 = greedy_matching(cv2.cvtColor(i31, cv2.COLOR_BGR2GRAY), cv2.cvtColor(i41, cv2.COLOR_BGR2GRAY), i51)
    cv2.imwrite("greedy_matches.jpg", i61)


    i71 = dtw_matching(i31, i41, i51)
    cv2.imwrite("dtw_matches.jpg", i71)
    #plt.imshow(i61)
    