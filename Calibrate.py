import cv2
import numpy as np
import glob
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def Calibration(fp, width_num, height_num):
    # 找棋盘格角点
    len_of_grids = 25.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 阈值
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((width_num * height_num, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width_num, 0:height_num].T.reshape(-1, 2)
    objp = objp * len_of_grids
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点
    images = sorted(glob.glob(fp + "/*.jpg"))  # 拍摄的十几张棋盘图片所在目录
    lenth = len(images)
    i = 1
    for fname in range(0, lenth):
        img = cv2.imread(images[fname])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (width_num, height_num), None)
        # 如果找到足够点对，将其存储起来
        if ret:
            filename = "Calibrate_Res/" + format(i) + ".jpg"
            i = i + 1
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (width_num, height_num), corners, ret)
            # cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('findCorners', 810, 540)
            # cv2.imshow('findCorners', gray)
            cv2.imwrite(filename, img)
            # cv2.waitKey(1)
    cv2.destroyAllWindows()

    # %% 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints, gray.shape[::-1], None, None)

    # 计算误差
    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_error += error

    return mtx, dist, tot_error, lenth

#write yaml
def write_yaml(mtx, dist):
    mtx = mtx.tolist()
    dist = dist.tolist()
    data = {"camera_matrix": mtx, "dist_coeff": dist}
    with open("Calibrate_Res/ parameter.yaml", "w") as file:
        yaml.dump(data, file)

if __name__ == '__main__':
    fp = "F:\Opencv-Imgs\Calibration"
    # 棋盘格模板规格
    width_num = 9  # 10 - 1
    height_num = 5  # 6  - 1
    mtx, dist, tot_error = Calibration(fp, width_num, height_num)
    write_yaml(mtx, dist)
    print("fx =", format(mtx[0][0]))
    print("fy =", format(mtx[1][1]))
    print("cx =", format(mtx[0][2]))
    print("cy =", format(mtx[1][2]))
    print("distortion coefficients")
    print("k_1 = {0}, k_2 = {1}, p_1 = {2}, p_2 = {3}, "
          "k_3 = {4}".format(dist[0][0], dist[0][1], dist[0][2], dist[0][3], dist[0][4]))
