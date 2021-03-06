import cv2
import numpy as np
from glob import glob
import sys
import json
import time
import os
import yaml
from matplotlib import pyplot as plt
from frame import Frame
from track import Track
import Calibrate


class Stereo_PnPVO:
    def __init__(self, mtx, dist_coeffs):
        # pose里面包括R矩阵和T矩阵（历元间的）
        self.poses = {}
        # measurements里面包括frame_ID， points2dL， points2dR，points3d信息
        self.measurements = {}
        # point_data里面包括frame_ID，左边相机的2D点，以及Point3d
        self.point_data = {}
        self.keyframe_ID = 0
        self.prev_trackedPts2D = None
        self.prev_trackedPts3D = None
        # 相机内参数矩阵
        K_mat = np.array(mtx)
        self.cam_params = {"camera_matrix": K_mat,
                           "dist_coeffs": dist_coeffs}
        # Percentage of keyframe points to decide new keyframe
        self.min_pt4kf = 0.7

    def initialize(self, frame):
        self.is_keyframe = True
        self.update_keyframeID()

        frame.get_measurements()
        rotation_matrix = np.eye(3)
        translation_vec = np.zeros(3)
        pose = np.hstack((rotation_matrix, translation_vec.reshape(3, 1)))
        self.poses[frame.frame_ID] = {"frame_ID": frame.frame_ID,
                                      "pose": pose.tolist(),
                                      "rmat": rotation_matrix.tolist(),
                                      "tvec": translation_vec.tolist(),
                                      "position": translation_vec.tolist()}
        self.update_measurements(frame)
        self.point_data[frame.frame_ID] = {"frame_ID": frame.frame_ID,
                                           "points2d": frame.lpts,
                                           "points3d": frame.points3d}

    def update_measurements(self, frame):
        self.measurements[self.keyframe_ID] = {"frame_ID": frame.frame_ID,
                                               "points2dL": frame.lpts,
                                               "points2dR": frame.rpts,
                                               "points3d": frame.points3d}
        self.prev_trackedPts2D = frame.lpts
        self.prev_ptIDs = [i for i in range(len(frame.lpts))]

    def update_keyframeID(self):
        self.keyframe_ID += 1

    def track(self, frame, prev_imgL, prev_imgR):
        tracker = Track(frame, prev_imgL, prev_imgR, self.prev_trackedPts2D)
        tracker.track_features(self.measurements[self.keyframe_ID]["points3d"],
                               self.prev_ptIDs)

        # print("tracked points {} of {}".format(len(tracker.tracked_pts),
        #                                         len(self.measurements[self.keyframe_ID]["points2dL"])))

        # Camera pose estimation
        position = self.pnp_solver(tracker.tracked_pts, tracker.corres_3Dpts, frame)

        # Required for data association
        self.prev_trackedPts2D = tracker.tracked_pts
        self.prev_ptIDs = tracker.point_IDs

        # Store points used
        self.point_data[frame.frame_ID] = {"frame_ID": frame.frame_ID,
                                           "points2d": tracker.tracked_pts,
                                           "points3d": tracker.corres_3Dpts}

        # Check if new keyframe should be defined
        new_kfFactor = len(self.measurements[self.keyframe_ID]["points2dL"]) * self.min_pt4kf

        if len(tracker.tracked_pts) < new_kfFactor:
            # print("new keyframe at {}. kf_ID: {}".format(frame.frame_ID, self.keyframe_ID))
            self.is_keyframe = True
            self.update_keyframeID()
            frame.get_measurements()
            frame.transform_3Dpoints(self.poses[frame.frame_ID])
            self.update_measurements(frame)
        return position, tracker.tracked_pts, self.measurements[self.keyframe_ID]["points2dL"], \
               self.measurements[self.keyframe_ID]["points2dR"]

    def pnp_solver(self, points2d, points3d, frame):
        points2d = np.array(points2d)
        points3d = np.array(points3d)
        intrinsic_matrix = self.cam_params["camera_matrix"]
        dist_coeffs = self.cam_params["dist_coeffs"]

        try:
            # flags: cv2.SOLVEPNP_P3P; cv2.SOLVEPNP_EPNP; cv2.SOLVEPNP_ITERATIVE
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points3d, points2d,
                intrinsic_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=2,
                iterationsCount=100)
        except Exception as e:
            print("\n-->PnP failed possibly due to less number of points\n")
            print(e)
            sys.exit()

        if success:
            rmat, Jacobian = cv2.Rodrigues(rvec)
            pose = cv2.hconcat((rmat, tvec))

            rotation_matrix = pose[:3, :3]
            translation_vec = pose[:3, 3]
            cur_position = -rotation_matrix.T.dot(translation_vec)
            print("position:", cur_position)

            self.poses[frame.frame_ID] = {"frame_ID": frame.frame_ID,
                                          "pose": pose.tolist(),
                                          "rmat": rotation_matrix.tolist(),
                                          "tvec": translation_vec.tolist(),
                                          "position": cur_position.tolist()}
        else:
            print("failed to estimate pose!")
            sys.exit()
        return cur_position

    def plot_trajectory(self, sframe, end_frame, plot3D = False, plot_points = False):
        #camera Y-axis is vertical axis
        xc, yc, zc = [], [], []
        for idx in range(sframe, end_frame):
            position = self.poses[idx]["position"]
            xc.append(position[0])
            yc.append(position[1])
            zc.append(position[2])

        # Reduce number of 3D points to plot
        slicing_percentage = 0.5 # percentage
        slice_points = True

        xp, yp, zp = [], [], []
        if plot_points:
            for kfID in range(1, self.keyframe_ID+1):
                points = self.measurements[kfID]["points3d"]

                if slice_points:
                    count = int(len(points)*slicing_percentage)
                    indices = np.random.choice(range(0,len(points)),
                                                count, replace=False)
                    for idx in indices:
                        xp.append(points[idx][0])
                        yp.append(points[idx][1])
                        zp.append(points[idx][2])
                else:
                    for pt in points:
                        xp.append(pt[0])
                        yp.append(pt[1])
                        zp.append(pt[2])

        if plot3D:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(xc, zc, yc, 'green')
            # s1 = [2*2**n for n in range(len(xc))]
            ax.scatter(xc, zc, yc, c='red')
            if plot_points:
                s2 = [2*1**n for n in range(len(xp))]
                # right, front, downwards
                ax.scatter(xp, zp, yp, c='gray',s=s2)
        else:
            s1 = [2*2**n for n in range(len(xc))]
            plt.plot(xc, zc, 'green')
            plt.scatter(xc, zc, c='red')
            if plot_points:
                s2 = [2*1**n for n in range(len(xp))]
                plt.scatter(xp, zp, c='gray', s=s2)
        plt.show()

    def write2file(self, sframe, end_frame, path, dataset):
        print("Writing data to ~/{}".format(path))

        for f in range(sframe, end_frame):
            pose_file = open(path+"poses_"+dataset+".txt","a+")
            pose_json = json.dumps(self.poses[f])
            pose_file.write(pose_json + "\n")

            point_file = open(path+"point_data_"+dataset+".txt","a+")
            point_json = json.dumps(self.point_data[f])
            point_file.write(point_json + "\n")
        print("Files wrtten\n")


def read_yaml(yaml_fp):
    with open(yaml_fp, "r") as file:
        parameter = yaml.load(file.read(), Loader=yaml.Loader)
        mtx = parameter['camera_matrix']
        dist = parameter['dist_coeff']
        mtx = np.array(mtx)
        dist = np.array(dist)
        dist = dist.reshape([4, 1])
    return mtx, dist


if __name__ == '__main__':
    ############################### Calibrate ################################
    # 棋盘格模板规格
    calib_fp = "Calibration"
    width_num = 9
    height_num = 5
    mtx, dist, tot_error = Calibrate.Calibration(calib_fp, width_num, height_num)
    Calibrate.write_yaml(mtx, dist)
    print("fx =", format(mtx[0][0]))
    print("fy =", format(mtx[1][1]))
    print("cx =", format(mtx[0][2]))
    print("cy =", format(mtx[1][2]))
    print("distortion coefficients")
    print("k_1 = {0}, k_2 = {1}, p_1 = {2}, p_2 = {3}, "
          "k_3 = {4}".format(dist[0][0], dist[0][1], dist[0][2], dist[0][3], dist[0][4]))


    ############################### VO ################################
    # Dataset
    dataset = "kitti"
    yaml_fp = 'kitti.yaml'
    detecting_method = 'ORB'
    matching_alg = 'flann'
    imgL_files = sorted(glob("data/" + dataset.lower() + "/image_0/*.png"))
    imgR_files = sorted(glob("data/" + dataset.lower() + "/image_1/*.png"))

    if dataset.lower() == "kitti":
        mtx, dist_coeffs = read_yaml(yaml_fp)
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

        frame = Frame(i, imgL, imgR, mtx, dist_coeffs, detecting_method, matching_alg)
        # frame = Frame(i, imgL, imgR, mtx, dist_coeffs)
        frame.extract_features()

        if not initialized:
            # Get points
            svo.initialize(frame)
            initialized = True

        else:
            # Track
            svo.track(frame, prev_imgL, prev_imgR)

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
        if os.path.exists(path+"poses_"+dataset+".txt"):
            os.remove(path+"poses_"+dataset+".txt")
        if os.path.exists(path+"point_data_"+dataset+".txt"):
            os.remove(path+"point_data_"+dataset+".txt")
        svo.write2file(start_frame, end_frame, path, dataset)
