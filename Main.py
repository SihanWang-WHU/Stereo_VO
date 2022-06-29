import cv2
import numpy as np
from glob import glob
import sys
import json
import time
import os
import yaml
from PyQt5.QtGui import QFont, QPainter, QImage, QPixmap
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *

import StereoPnP_VO
import frame
from MainWindow import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
import MainWindow

import frame
import track
from StereoPnP_VO import Stereo_PnPVO
import Calibrate
matplotlib.use('Qt5Agg')
# 使用 matplotlib中的FigureCanvas (在使用 Qt5 Backends中 FigureCanvas继承自QtWidgets.QWidget)

class Window_Param:
    calibpath = []
    cali_lenth = 0
    width_num = 9
    height_num = 5
    mtx = np.zeros([3, 3])
    dist = np.zeros([1, 4])
    tot_error = 0.0

    def get_calibration_fp(self):
        filePath = []
        self.calibpath = QtWidgets.QFileDialog.getExistingDirectory(None, "选取棋盘格文件夹", os.getcwd())
        filePath.append(self.calibpath)
        ui.califp_char.setFont(QFont('Times', 14, QFont.Black))
        ui.califp_char.setText(self.calibpath)

    def button_Calibrate(self):
        self.scene = QGraphicsScene()
        ui.Calibimgs.setScene(self.scene)
        ui.Calibimgs.show()
        self.mtx, self.dist, self.tot_error, self.cali_lenth = \
            Calibrate.Calibration(self.calibpath, self.width_num, self.height_num)
        for i in range(1, self.cali_lenth + 1):
            qfile = "Calibrate_Res/" + format(i) + ".jpg"
            img = cv2.imread(qfile)
            img = cv2.resize(img, [440, 330])
            cv2.waitKey(100)
            y, x = img.shape[:-1]
            frame = QImage(img, x, y, QImage.Format_Grayscale16)
            self.scene.clear()
            self.pix = QPixmap.fromImage(frame)
            self.scene.addPixmap(self.pix)
        ui.fx_double.setText(format(self.mtx[0][0]))
        ui.fy_double.setText(format(self.mtx[1][1]))
        ui.cx_double.setText(format(self.mtx[0][2]))
        ui.cy_double.setText(format(self.mtx[1][2]))
        ui.k1_double.setText(format(self.dist[0][0]))
        ui.k2_double.setText(format(self.dist[0][1]))
        ui.p1_double.setText(format(self.dist[0][2]))
        ui.p2_double.setText(format(self.dist[0][3]))
        ui.k4_double.setText(format(self.dist[0][4]))
        ui.error_double.setText(format(self.tot_error))



    def click_califp_button(self):
        if ui.califp_button.isEnabled():
            self.get_calibration_fp()

    def click_start_Calib_button(self):
        # get the width num and height num (int32) of the grid from the spinbox input
        self.width_num = ui.widthnum_int.value()
        self.height_num = ui.heightnum_int.value()
        if ui.Start_Calib_button.isEnabled():
            self.button_Calibrate()


    def click_start_writeyaml_button(self):
        if ui.Start_Writeyaml_button.isEnabled():
            Calibrate.write_yaml(self.mtx, self.dist)

if __name__ == '__main__':
    # ############################## Init Window ###############################
    # 实例化，传参
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    para = Window_Param()
    # 创建ui，引用demo1文件中的Ui_MainWindow类
    ui = MainWindow.Ui_MainWindow()
    # 调用Ui_MainWindow类的setupUi，创建初始组件
    ui.setupUi(mainWindow)
    # 创建窗口
    mainWindow.show()
    ui.califp_button.clicked.connect(para.click_califp_button)
    ui.Start_Calib_button.clicked.connect(para.click_start_Calib_button)
    ui.Start_Writeyaml_button.clicked.connect(para.click_start_writeyaml_button)
    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())

    ############################### VO ################################
    # Dataset
    dataset = "kitti"
    yaml_fp = 'kitti.yaml'
    detecting_method = 'ORB'
    matching_alg = 'flann'
    imgL_files = sorted(glob("data/" + dataset.lower() + "/image_0/*.png"))
    imgR_files = sorted(glob("data/" + dataset.lower() + "/image_1/*.png"))

    if dataset.lower() == "kitti":
        mtx, dist_coeffs = StereoPnP_VO.read_yaml(yaml_fp)
    else:
        # Specify intrinsics of other camera/dataset used
        print("Intrinsics required")
        sys.exit()

    show_trajectory = True
    show_3Dpts = False
    plot3D = False
    write2file = True

    prev_imgL, prev_imgR = None, None
    total_duration = 0
    initialized = False

    svo = Stereo_PnPVO(mtx, dist_coeffs)

    print()
    start_frame = 0
    end_frame = len(imgL_files)
    for i in range(start_frame, end_frame):
        print("frame_ID->{}".format(i))
        start_time = time.time()

        imgL = cv2.imread(imgL_files[i])
        imgR = cv2.imread(imgR_files[i])

        cur_frame = frame.Frame(i, imgL, imgR, mtx, dist_coeffs, detecting_method, matching_alg)
        # frame = Frame(i, imgL, imgR, mtx, dist_coeffs)
        cur_frame.extract_features()

        if not initialized:
            # Get points
            svo.initialize(cur_frame)
            initialized = True

        else:
            # Track
            svo.track(cur_frame, prev_imgL, prev_imgR)

        # Keep previous frames for tracking
        prev_imgL = imgL.copy()
        prev_imgR = imgR.copy()

        # local and global durations
        duration = time.time() - start_time
        total_duration += duration
        print("duration: {}s".format(duration));
        print()

    print("--------------------------------------------------------------")
    print("Total duration: {}s".format(total_duration))
    print("No. of keyframes:", svo.keyframe_ID)
    print("No. of frames:", end_frame - start_frame);
    print()

    # Plot camera trajectory
    if show_trajectory:
        print("Plotting trajectory...\n")
        svo.plot_trajectory(start_frame, end_frame, plot3D, show_3Dpts)

    # Write pose and points to file
    if write2file:
        path = "outputs/"
        if os.path.exists(path + "poses_" + dataset + ".txt"):
            os.remove(path + "poses_" + dataset + ".txt")
        if os.path.exists(path + "point_data_" + dataset + ".txt"):
            os.remove(path + "point_data_" + dataset + ".txt")
        svo.write2file(start_frame, end_frame, path, dataset)
