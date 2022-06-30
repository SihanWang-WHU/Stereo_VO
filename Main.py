import cv2
import numpy as np
from glob import glob

from PyQt5.QtCore import QRectF, QPointF, Qt
from PyQt5.uic import loadUi
import sys
import time
import os
from PyQt5.QtGui import QFont, QPainter, QImage, QPixmap, QPen
import matplotlib
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *
import StereoPnP_VO
from MainWindow import Ui_MainWindow
from VO import Ui_VO_show
from PyQt5.QtWidgets import QApplication, QMainWindow
import frame
from StereoPnP_VO import Stereo_PnPVO
import Calibrate
import matplotlib
matplotlib.use('Qt5Agg')


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class Calib_Param(QMainWindow, Ui_MainWindow):
    calibpath = []
    cali_lenth = 0
    width_num = 9
    height_num = 5
    mtx = np.zeros([3, 3])
    dist = np.zeros([1, 4])
    tot_error = 0.0

    def __init__(self):
        super(Calib_Param, self).__init__()
        loadUi("MainWindow.ui", self)
        self.setWindowTitle("Calibration Window")
        self.califp_button.clicked.connect(self.click_califp_button)
        self.Start_Calib_button.clicked.connect(self.click_start_Calib_button)
        self.Start_Writeyaml_button.clicked.connect(self.click_start_writeyaml_button)
        self.gotovo_button.clicked.connect(self.click_VO_button)

    def get_calibration_fp(self):
        filePath = []
        self.calibpath = QtWidgets.QFileDialog.getExistingDirectory(None, "选取棋盘格文件夹", os.getcwd())
        filePath.append(self.calibpath)
        self.califp_char.setFont(QFont('Times', 12, QFont.Black))
        self.califp_char.setText(self.calibpath)

    def button_Calibrate(self):
        self.scene = QGraphicsScene()
        self.Calibimgs.setScene(self.scene)
        self.Calibimgs.show()
        self.mtx, self.dist, self.tot_error, self.cali_lenth = \
            Calibrate.Calibration(self.calibpath, self.width_num, self.height_num)
        for i in range(1, self.cali_lenth + 1):
            qfile = "Calibrate_Res/" + format(i) + ".jpg"
            img = cv2.imread(qfile)
            img = cv2.resize(img, [440, 330])
            cv2.waitKey(100)
            y, x = img.shape[:-1]
            frame = QImage(img, x, y, QImage.Format_BGR888)
            self.scene.clear()
            self.pix = QPixmap.fromImage(frame)
            self.scene.addPixmap(self.pix)
        self.fx_double.setFont(QFont('Times', 10, QFont.Black))
        self.fy_double.setFont(QFont('Times', 10, QFont.Black))
        self.cx_double.setFont(QFont('Times', 10, QFont.Black))
        self.fx_double.setFont(QFont('Times', 10, QFont.Black))
        self.cy_double.setFont(QFont('Times', 10, QFont.Black))
        self.k1_double.setFont(QFont('Times', 10, QFont.Black))
        self.k2_double.setFont(QFont('Times', 10, QFont.Black))
        self.p1_double.setFont(QFont('Times', 10, QFont.Black))
        self.p2_double.setFont(QFont('Times', 10, QFont.Black))
        self.k4_double.setFont(QFont('Times', 10, QFont.Black))
        self.error_double.setFont(QFont('Times', 10, QFont.Black))
        self.fx_double.setText(format(self.mtx[0][0]))
        self.fy_double.setText(format(self.mtx[1][1]))
        self.cx_double.setText(format(self.mtx[0][2]))
        self.cy_double.setText(format(self.mtx[1][2]))
        self.k1_double.setText(format(self.dist[0][0]))
        self.k2_double.setText(format(self.dist[0][1]))
        self.p1_double.setText(format(self.dist[0][2]))
        self.p2_double.setText(format(self.dist[0][3]))
        self.k4_double.setText(format(self.dist[0][4]))
        self.error_double.setText(format(self.tot_error))

    def click_VO_button(self):
        widget.setCurrentIndex(1)

    def click_califp_button(self):
        if self.califp_button.isEnabled():
            self.get_calibration_fp()

    def click_start_Calib_button(self):
        # get the width num and height num (int32) of the grid from the spinbox input
        self.width_num = self.widthnum_int.value()
        self.height_num = self.heightnum_int.value()
        if self.Start_Calib_button.isEnabled():
            self.button_Calibrate()

    def click_start_writeyaml_button(self):
        if self.Start_Writeyaml_button.isEnabled():
            Calibrate.write_yaml(self.mtx, self.dist)


class VO_Param(QDialog, Ui_VO_show):
    # default parameters
    dataset = "kitti"
    yaml_fp = 'kitti.yaml'
    detecting_method = 'ORB'
    matching_alg = 'flann'
    mtx = np.zeros([3, 3])
    dist_coeffs = np.zeros([4, 1])
    VO_filepath = []
    imgL_files = sorted(glob("data/" + dataset.lower() + "/image_0/*.png"))
    imgR_files = sorted(glob("data/" + dataset.lower() + "/image_1/*.png"))
    show_trajectory = True
    show_pts = False
    plot3D = False
    slice_points = False
    write2file = True
    cur_posx = []
    cur_posz = []

    prev_imgL, prev_imgR = None, None
    total_duration = 0
    initialized = False

    def __init__(self):
        super(VO_Param, self).__init__()
        loadUi("VO.ui", self)
        self.lscene = QGraphicsScene()
        self.traj_scene = QGraphicsScene()
        self.rscene = QGraphicsScene()
        self.setWindowTitle("Video Odometry Window")
        self.gobacktomain_button.clicked.connect(self.click_Calib_button)
        self.vofp_button.clicked.connect(self.get_calibration_fp)
        self.startVO_button.clicked.connect(self.click_vostart_button)
        self.SIFT_check.stateChanged.connect(self.check_sift)
        self.ORB_check.stateChanged.connect(self.check_orb)
        self.BruteForce_check.stateChanged.connect(self.check_bf)
        self.Flann_check.stateChanged.connect(self.check_flann)
        self.plot3d_check.stateChanged.connect(self.check_plot3d)
        self.plotpts_check.stateChanged.connect(self.check_plot_pts)
        self.slicepoints_check.stateChanged.connect(self.check_slicepts)
        self.write2file_check.stateChanged.connect(self.check_write2file)

    def click_Calib_button(self):
        widget.setCurrentIndex(0)

    def get_calibration_fp(self):
        filePath = []
        self.VO_filepath = QtWidgets.QFileDialog.getExistingDirectory(None, "选取VO文件夹", os.getcwd())
        filePath.append(self.VO_filepath)
        self.VOfp_char.setFont(QFont('Times', 12, QFont.Black))
        self.VOfp_char.setText(self.VO_filepath)
        self.imgL_files = sorted(glob(self.VO_filepath + "/image_0/*.png"))
        self.imgR_files = sorted(glob(self.VO_filepath + "/image_1/*.png"))

    def check_sift(self):
        if self.SIFT_check.isChecked():
            self.detecting_method = "SIFT"
            self.ORB_check.setChecked(False)
        else:
            self.ORB_check.setChecked(True)

    def check_orb(self):
        if self.ORB_check.isChecked():
            self.detecting_method = "ORB"
            self.SIFT_check.setChecked(False)
        else:
            self.SIFT_check.setChecked(True)

    def check_bf(self):
        if self.BruteForce_check.isChecked():
            self.matching_alg = "BF"
            self.Flann_check.setChecked(False)
        else:
            self.Flann_check.setChecked(True)

    def check_flann(self):
        if self.Flann_check.isChecked():
            self.matching_alg = "FLANN"
            self.BruteForce_check.setChecked(False)
        else:
            self.BruteForce_check.setChecked(True)

    def check_plot3d(self):
        if self.plot3d_check.isChecked():
            self.plot3D = True
        else:
            self.plot3D = False

    def check_plot_pts(self):
        if self.plotpts_check.isChecked():
            self.show_pts = True
        else:
            self.show_pts = False

    def check_write2file(self):
        if self.write2file_check.isChecked():
            self.write2file = True
        else:
            self.write2file = False

    def check_slicepts(self):
        if self.slicepoints_check.isChecked():
            self.slice_points = True
        else:
            self.slice_points = False

    def show_trackedPoints(self, tag, is_keyframe):
        # tag = 0 for left img
        if tag == 0:
            self.imgL_show = self.imgL.copy()
            for pt in self.trackedpts:
                pt = np.array(pt)
                pt = pt.reshape(2, )
                coord = (int(pt[0]), int(pt[1]))
                self.imgL_show = cv2.circle(self.imgL_show, coord, radius=3, color=(2, 51, 196), thickness=2)
            if is_keyframe:
                for pt2dl in self.points2dl:
                    pt2dl = np.array(pt2dl)
                    pt2dl = pt2dl.reshape(2, )
                    coord2dl = (int(pt2dl[0]), int(pt2dl[1]))
                    self.imgL_show = cv2.circle(self.imgL_show, coord2dl, radius=3, color=(196, 2, 51), thickness=2)
            self.imgL_show = cv2.resize(self.imgL_show, [500, 200])
            self.imgL_show = cv2.cvtColor(self.imgL_show, cv2.COLOR_BGR2RGB)
            self.h, self.w = self.imgL_show.shape[:-1]
            self.lframe = QImage(self.imgL_show, self.w, self.h, QImage.Format_RGB888)
            self.lscene.clear()
            self.lpix = QPixmap.fromImage(self.lframe)
            self.lscene.addPixmap(self.lpix)

        # tag = 0 for right img
        if tag == 1:
            self.imgR_show = self.imgR.copy()
            if is_keyframe:
                for pt in self.points2dr:
                    pt = np.array(pt)
                    pt = pt.reshape(2, )
                    coord = (int(pt[0]), int(pt[1]))
                    self.imgR_show = cv2.circle(self.imgR_show, coord, radius=3, color=(196, 2, 51), thickness=2)
            self.imgR_show = cv2.resize(self.imgR_show, [500, 200])
            self.imgR_show = cv2.cvtColor(self.imgR_show, cv2.COLOR_BGR2RGB)
            self.h, self.w = self.imgR_show.shape[:-1]
            self.rframe = QImage(self.imgR_show, self.w, self.h, QImage.Format_RGB888)
            self.rscene.clear()
            self.rpix = QPixmap.fromImage(self.rframe)
            self.rscene.addPixmap(self.rpix)
            cv2.waitKey(20)

    def show_traj(self, cur_x, cur_z):
        self.trajmat = np.zeros(shape=(260, 500, 3), dtype=np.uint8) + 255
        for i in range(0, len(cur_x)):
            x = 180 - int(cur_x[i] / 3)
            y = int(cur_z[i] / 3) + 250
            B, G, R = cv2.split(self.trajmat)
            B[x: x+3, y: y+3] = 0
            G[x: x+3, y: y+3] = 0
            R[x: x+3, y: y+3] = 0
            self.trajmat = cv2.merge((B, G, R))
        self.trajmat = cv2.cvtColor(self.trajmat, cv2.COLOR_BGR2RGB)
        self.trajh, self.trajw = self.trajmat.shape[:-1]
        self.trajframe = QImage(self.trajmat, self.trajw, self.trajh, QImage.Format_RGB888)
        self.traj_scene.clear()
        self.trajpix = QPixmap.fromImage(self.trajframe)
        self.traj_scene.addPixmap(self.trajpix)


    def click_vostart_button(self):
        initialized = False

        # initialize QgraphicsScene
        self.leftimg.setScene(self.lscene)
        self.rightimg.setScene(self.rscene)
        self.trajectoryimg.setScene(self.traj_scene)
        self.leftimg.show()
        self.rightimg.show()
        self.trajectoryimg.show()


        if self.dataset.lower() == "kitti":
            self.mtx, self.dist_coeffs = StereoPnP_VO.read_yaml(self.yaml_fp)
        else:
            # Specify intrinsics of other camera/dataset used
            print("Intrinsics required")
            sys.exit()

        svo = Stereo_PnPVO(self.mtx, self.dist_coeffs)

        self.start_frame = 0
        self.end_frame = len(self.imgL_files)
        start_time = time.time()

        # Main Loop of stereo VO
        for i in range(self.start_frame, self.end_frame):
            epoch_start_time = time.time()
            svo.is_keyframe = False
            self.imgL = cv2.imread(self.imgL_files[i])
            self.imgR = cv2.imread(self.imgR_files[i])

            cur_frame = frame.Frame(i, self.imgL, self.imgR, self.mtx, self.dist_coeffs,
                                    self.detecting_method, self.matching_alg)
            cur_frame.extract_features()

            if not initialized:
                # Get points
                svo.initialize(cur_frame)
                self.right_double.setText(format(0.0000))
                self.down_double.setText(format(0.0000))
                self.front_double.setText(format(0.0000))
                self.cur_posx.append(0)
                self.cur_posz.append(0)
                initialized = True

            else:
                # Track
                self.position, self.trackedpts, self.points2dl, self.points2dr = svo.track(cur_frame,
                                                                                           prev_imgL,
                                                                                           prev_imgR)
                self.show_trackedPoints(0, svo.is_keyframe)
                self.show_trackedPoints(1, svo.is_keyframe)
                self.right_double.setText(format(self.position[0]))
                self.down_double.setText(format(self.position[1]))
                self.front_double.setText(format(self.position[2]))
                self.cur_posx.append(self.position[0])
                self.cur_posz.append(self.position[2])
                self.show_traj(self.cur_posz, self.cur_posx)

            frameid = cur_frame.frame_ID
            # Keep previous frames for tracking
            prev_imgL = self.imgL.copy()
            prev_imgR = self.imgR.copy()

            # local and global durations
            duration = time.time() - epoch_start_time
            total_duration = time.time() - start_time

            self.FrameID_int.setText(format(frameid) + "/" + format(self.end_frame))
            self.duration_double.setText(format(duration))
            self.totalduration_double.setText(format(total_duration))
            self.kfid_int.setText(format(svo.keyframe_ID))



        print("ploting trajectory")
        svo.plot_trajectory(self.start_frame, self.end_frame, self.plot3D, self.show_pts)

        # Write pose and points to file
        if self.write2file:
            path = "VO_Res/"
            if os.path.exists(path + "poses_" + self.dataset + ".txt"):
                os.remove(path + "poses_" + self.dataset + ".txt")
            if os.path.exists(path + "point_data_" + self.dataset + ".txt"):
                os.remove(path + "point_data_" + self.dataset + ".txt")
            svo.write2file(self.start_frame, self.end_frame, path, self.dataset)


if __name__ == '__main__':
    # ############################## Init Window ###############################
    # 实例化，传参
    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()
    calib_window = Calib_Param()
    vo_window = VO_Param()
    widget.addWidget(calib_window)
    widget.addWidget(vo_window)
    widget.setFixedWidth(1061)
    widget.setFixedHeight(776)
    widget.show()

    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())
