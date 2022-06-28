import cv2
import numpy as np
import glob
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 找棋盘格角点
len_of_grids = 25.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
# 棋盘格模板规格
width_num = 9   # 10 - 1
height_num = 5    # 6  - 1
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((width_num * height_num, 3), np.float32)
objp[:, :2] = np.mgrid[0:width_num, 0:height_num].T.reshape(-1, 2)
objp = objp * len_of_grids
# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点
images = glob.glob('.\\Calibration\\*.jpg')  #   拍摄的十几张棋盘图片所在目录
i = 1
for fname in images:
     img = cv2.imread(fname)
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     # 找到棋盘格角点
     ret, corners = cv2.findChessboardCorners(gray, (width_num, height_num), None)
     # 如果找到足够点对，将其存储起来
     if ret:
        i = i+1
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (width_num, height_num), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 810, 540)
        cv2.imshow('findCorners', img)
        cv2.waitKey(2000)
cv2.destroyAllWindows()

#%% 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                   imgpoints, gray.shape[::-1], None, None)

# print("ret:", ret)
# print("mtx:\n", mtx)      # 内参数矩阵
# print("dist:\n", dist)   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# print("rvecs:\n", rvecs)   # 旋转向量  # 外参数
# print("tvecs:\n", tvecs)  # 平移向量  # 外参数

testimg = cv2.imread('testimg.jpg')
h, w = testimg.shape[:2]
'''
优化相机内参（camera matrix），这一步可选。
参数1表示保留所有像素点，同时可能引入黑色像素，
设为0表示尽可能裁剪不想要的像素，这是个scale，0-1都可以取。
'''
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# print("newcameramtx:\n", newcameramtx)
# 纠正畸变
dst = cv2.undistort(testimg, mtx, dist, None, newcameramtx)
#输出纠正畸变以后的图片
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('undistorted_testimg.png', dst)

#计算误差
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print("total error: ", tot_error/len(objpoints))

#write yaml
def write_yaml(mtx, dist, newcameramtx):
    mtx = mtx.tolist()
    dist = dist.tolist()
    newcameramtx = newcameramtx.tolist()
    data = {"camera_matrix": mtx, "dist_coeff": dist, "new_camera_matrix": newcameramtx}
    with open("parameter.yaml", "w") as file:
        yaml.dump(data, file)

if __name__ == '__main__':
    write_yaml(mtx, dist, newcameramtx)