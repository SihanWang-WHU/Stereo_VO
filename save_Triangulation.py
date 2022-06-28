import cv2
import numpy as np
import yaml
import math
from cv2 import RANSAC
from matplotlib import pyplot as plt

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
from cv2 import RANSAC
from matplotlib import pyplot as plt

def read_yaml(yaml_fp):
    with open(yaml_fp, "r") as file:
        parameter = yaml.load(file.read(), Loader=yaml.Loader)
        mtx = parameter['camera_matrix']
        dist = parameter['dist_coeff']
        newcameramtx = parameter['new_camera_matrix']
        mtx = np.array(mtx)
        dist = np.array(dist)
        newcameramtx = np.array(newcameramtx)
    return mtx, dist, newcameramtx

def sift_detect(img1, img2):
    # 使用SIFT特征检测器和描述符，计算关键点和描述符
    sift = cv2.xfeatures2d.SIFT_create()
    kp1= sift.detect(img1, None)
    des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # kps是关键点。它所包含的信息有：
    # angle：角度，表示关键点的方向，为了保证方向不变形，SIFT算法通过对关键点周围邻域进行梯度运算，求得该点方向。-1为初值。
    # class_id：当要对图片进行分类时，我们可以用class_id对每个特征点进行区分，未设定时为-1，需要靠自己设定
    # octave：代表是从金字塔哪一层提取的得到的数据。
    # pt：关键点点的坐标
    # response：响应程度，代表该点强壮大小，更确切的说，是该点角点的程度。
    # size：该点直径的大小
    # des是描述子，是每个特征点的128维向量
    return kp1, des1[1], kp2, des2

def Match_Flann(kp1, des1, kp2, des2):
    # FLANN特征匹配
    # FLANN parameters
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 进行匹配
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []

    for m, n in matches:
        # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
        if m.distance < 0.7 * n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    matchImg = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:20],
                                  None, flags=2)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    return pts1, pts2, matchImg

def FindF_E_Mat(pts1, pts2, mtx):
    # 计算基础矩阵F和本质矩阵E
    F, maskF = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    print("基础矩阵为：")
    print(F)
    E, maskE = cv2.findEssentialMat(pts1, pts2, cameraMatrix=mtx, method=cv2.RANSAC)
    print("本质矩阵为")
    print(E)
    retval2, R, T, mask = cv2.recoverPose(E, pts1, pts2, mtx)
    print("旋转角度为")
    print(R)
    print("平移量为")
    print(T)
    baseline = math.sqrt(T[0][0] * T[0][0] +
                         T[1][0] * T[1][0] + T[2][0] * T[2][0])
    print(baseline)
    return F, E, R, T

#draw Epilines
def drawlines(image1, image2, lines, pts1, pts2):
    r, c = image1.shape
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = (np.random.randint(0, 255), np.random.randint(0,
                255), np.random.randint(0, 255))
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        image1 = cv2.line(image1, (x0, y0), (x1, y1), color, 1)
        image1 = cv2.circle(image1, tuple(pt1), 5, color, -1)
        image2 = cv2.circle(image2, tuple(pt2), 5, color, -1)
    return image1, image2

def triangulation(kps1, kps2, R, t, mtx):
    # projMatr1 和 proMatr2 都是3*4的矩阵
    projMatr1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 第一个相机参数
    projMatr2 = np.concatenate((R, t), axis=1)  # 第二个相机参数
    # 将像素坐标系转换至相机坐标系
    projMatr1 = np.matmul(mtx, projMatr1)  # 相机内参 相机外参
    projMatr2 = np.matmul(mtx, projMatr2)  #
    kps1 = np.array(kps1)
    kps2 = np.array(kps2)
    points4D = cv2.triangulatePoints(projMatr1, projMatr2, kps1.T, kps1.T)
    points4D /= points4D[3]  # 归一化
    points4D = points4D.T[:, 0:3]  # 取坐标点
    print("三角测量三维坐标")
    print(points4D)
    return points4D

if __name__ == '__main__':
    # 图像的读取和显示
    img1_fp = 'Epipolar-Geometry\cam0img0.png'
    img2_fp = 'Epipolar-Geometry\cam1img0.png'
    yaml_fp = 'parameter.yaml'
    img1 = cv2.imread(img1_fp, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_fp, cv2.IMREAD_GRAYSCALE)
    mtx, dist, newcameramtx = read_yaml(yaml_fp)
    kp1, des1, kp2, des2 = sift_detect(img1, img2)
    pts1, pts2, matchimg = Match_Flann(kp1, des1, kp2, des2)
    F, E, R, T = FindF_E_Mat(pts1, pts2, mtx)
    points4D = triangulation(pts1, pts2, R, T, mtx)