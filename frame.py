import math
import cv2
import sys
import numpy as np
import yaml


class Frame:
    # 类的构造函数
    def __init__(self, ID, imgL, imgR, mtx, dist_coeffs, detecting_method, matching_algorithm):
        self.basline = 1
        self.frame_ID = ID
        self.imgL = imgL
        self.imgR = imgR
        self.mtx = mtx
        self.dist_coeffs = dist_coeffs
        self.method = detecting_method
        self.match_alg = matching_algorithm


    def extract_features(self):
        if self.method.lower() == "orb":
            detector = cv2.ORB_create(1000)
            extractor = detector
        elif self.method.lower() == "sift":
            # 使用SIFT特征检测器和描述符，计算关键点和描述符
            detector = cv2.SIFT_create(contrastThreshold=0.06, edgeThreshold=0.15)
            extractor = detector
        else:
            # TO DO: define other types of features extractors
            print("Invalid feature extractor")
            sys.exit()

        self.kpl, self.desl = self.feature_extractor(self.imgL, detector, extractor)
        self.kpr, self.desr = self.feature_extractor(self.imgR, detector, extractor)


    @staticmethod
    def feature_extractor(image, detector, extractor):
        keypoints = detector.detect(image)
        descriptors = extractor.compute(image, keypoints)
        return keypoints, descriptors[1]

    def get_measurements(self):
        self.match_features()
        self.traingulate_points()

    def match_features(self):
        self.matches = []
        if self.match_alg.lower() == "bf":
            bfm = cv2.BFMatcher()
            bfMatches = bfm.knnMatch(self.desl, self.desr, k=2)
            for m, n in bfMatches:
                if self.method.lower() == "orb":
                    if m.distance < 0.75 * n.distance:
                        # matches属性中包含了distace，图像的索引imgIdx
                        # 在左边图像中描述子的编号queryIdx和在右边图像中描述子的编号trainIdx
                        self.matches.append(m)
                elif self.method.lower() == "sift":
                    if m.distance < 0.6 * n.distance:
                        # matches属性中包含了distace，图像的索引imgIdx
                        # 在左边图像中描述子的编号queryIdx和在右边图像中描述子的编号trainIdx
                        self.matches.append(m)



        elif self.match_alg.lower() == "flann":
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary

            desl = np.float32(self.desl)
            desr = np.float32(self.desr)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desl, desr, k=2)

            for i, (m, n) in enumerate(matches):
                if m.distance < 0.75 * n.distance:
                    self.matches.append(m)

        else:
            print("Invalid matching method. Use 'flann' or 'brute_force')")
            sys.exit()

    def convertMatchP2xy(self):
        # 左边影像和右边影像的特征点（在像素坐标系上）
        self.lpts, self.rpts = [], []
        # 左边影像和右边影像的特征点（在图像坐标系上）
        self.mkpl, self.mkpr = [], []
        # 左边影像和右边影像的特征描述子
        self.mdesl, self.mdesr = [], []
        for idx in range(len(self.matches)):
            # (u,v) points
            self.lpts.append(self.kpl[self.matches[idx].queryIdx].pt)
            self.rpts.append(self.kpr[self.matches[idx].trainIdx].pt)
            # Keypoints
            self.mkpl.append(self.kpl[self.matches[idx].queryIdx])
            self.mkpr.append(self.kpr[self.matches[idx].trainIdx])
            # Descriptors
            self.mdesl.append(self.desl[self.matches[idx].queryIdx])
            self.mdesr.append(self.desr[self.matches[idx].trainIdx])

        assert len(self.lpts) == len(self.rpts),\
        "Mismatch left and right point count"

    def FindF_E_Mat(self):
        self.Fundamentalmtx = np.eye(3)
        self.Eseentialmtx = np.eye(3)
        self.CamR = np.eye(3)
        self.CamT = np.zeros([3, 1])
        pts1 = np.array(self.lpts)
        pts2 = np.array(self.rpts)
        #计算基础矩阵F和本质矩阵E
        self.Fundamentalmtx, maskF = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        self.Eseentialmtx, maskE = cv2.findEssentialMat(pts1, pts2, self.mtx, method=cv2.RANSAC)
        retval2, self.CamR, self.CamT, mask = cv2.recoverPose(self.Eseentialmtx, pts1, pts2, self.mtx)
        self.baseline = math.sqrt(self.CamT[0][0] * self.CamT[0][0] +
                                  self.CamT[1][0] * self.CamT[1][0] + self.CamT[2][0] * self.CamT[2][0])

    def traingulate_points(self):
        self.convertMatchP2xy()
        self.FindF_E_Mat()

        fx = self.mtx[0][0]
        fy = self.mtx[1][1]
        cx = self.mtx[0][2]
        cy = self.mtx[1][2]
        baseline = self.baseline

        self.points3d = []

        for lp, rp in zip(self.lpts, self.rpts):
            lp = np.array(lp)
            rp = np.array(rp)
            disparity = lp[0] - rp[0]

            if disparity > 0:
                depth = (fx * baseline) / disparity
                x = ((lp[0] - cx) * depth) / fx
                y = ((lp[1] - cy) * depth) / fy
                self.points3d.append([x, y, depth])
            else:
                self.points3d.append([0, 0, 0])


    def transform_3Dpoints(self, pose):
        R = np.array(pose["rmat"])
        t = np.array(pose["tvec"])
        transformed_points = np.array([])
        for idx in range(len(self.points3d)):
            if self.points3d[idx][0] is not None:
                transformed_points = np.append(transformed_points, \
                                               R.T.dot(self.points3d[idx]) \
                                               + (-R.T.dot(t)))
        #从齐次坐标转换到非齐次坐标
        self.points3d = transformed_points.reshape(
            int(len(transformed_points) / 3), 3).tolist()