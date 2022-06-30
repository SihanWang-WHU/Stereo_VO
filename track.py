import cv2
import numpy as np


class Track:

    def __init__(self, frame, prev_imgL, prev_imgR, points2track):
        self.frame = frame
        self.prev_imgL = prev_imgL
        self.prev_imgR = prev_imgR
        self.points2track = points2track

        self.trackErr_threshold = 15.0
        """
        nextPts, status, err = calcOpticalFlowPyrLK(prevImg, nextImg, prevPts[, nextPts[, status[, 
        err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]])
        参数说明：
        prevImage 前一帧8-bit图像
        nextImage 当前帧8-bit图像
        prevPts 待跟踪的特征点向量
        nextPts 输出跟踪特征点向量
        status 特征点是否找到，找到的状态为1，未找到的状态为0
        err 输出错误向量
        winSize 搜索窗口的大小
        maxLevel 最大的金字塔层数
        flags 可选标识：OPTFLOW_USE_INITIAL_FLOW   OPTFLOW_LK_GET_MIN_EIGENVALS
        """
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def track_features(self, pts3D, id_list):
        pts, status, errs = self.compute_opticalFlow()
        self.tracked_pts = []
        self.corres_3Dpts = []  # corresponding 3D points
        self.point_IDs = []
        for i in range(len(pts)):
            if status[i] == 1 and errs[i] < self.trackErr_threshold:
                self.tracked_pts.append(pts[i].tolist())
                self.corres_3Dpts.append(pts3D[id_list[i]])
                self.point_IDs.append(id_list[i])

        # show_tracked_pts = True
        # if show_tracked_pts:
        #     self.show_trackedPoints()

    def compute_opticalFlow(self):
        points2track = np.float32(np.array(self.points2track).reshape(-1, 1, 2))
        return cv2.calcOpticalFlowPyrLK(self.prev_imgL, self.frame.imgL, \
                                        points2track, None, **self.lk_params)

    # def show_trackedPoints(self):
    #     img = self.frame.imgL.copy()
    #     for pt in self.tracked_pts:
    #         pt = np.array(pt)
    #         pt = pt.reshape(2, )
    #         coord = (int(pt[0]), int(pt[1]))
    #         img = cv2.circle(img, coord, radius=3, color=(2, 51, 196), thickness=2)
    #     cv2.imshow('left_frame', img);
    #     cv2.waitKey(10)