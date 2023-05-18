import sys
sys.path.append('../')
import cv2
import numpy as np
import pickle
import os
from calibration.utils import *


class oneCamData:
    def __init__(self, data_dir, cam, square_size, pattern_size, start_idx, num_frame, depth_only):
        print('---------------------------------------------------------')
        print('>>> Initilize camera %s' % cam)
        self.data_dir = data_dir
        self.cam = cam
        self.square_size = square_size
        self.pattern_size = pattern_size  # (col, row)
        self.pattern_points = self.init_pattern_points()
        self.cns_pattern = None
        self.start_idx = start_idx
        self.num_frame = num_frame

        if 'kinect' in self.cam:
            self.imgs_d = None
            self.imgs_c = None
            self.imgs_gray = None

            self.intr_c = None
            self.intr_d = None
            self.intr = None
            self.T_d2c = None

            self.cns_c = None  # color corners, np.array [num_imgs, num_corners, 2]
            self.cns_gray = None  # infrared norners, np.array [num_imgs, num_corners, 2]
            self.cns_flag = None  # indicate if corners can be detected, np.array [num_imgs]
        else:
            self.imgs_gray = None
            self.intr_c = None  # not use
            self.intr_d = None  # not use
            self.intr = None  # intrinsic parameters
            self.cns_gray = None  # corners, np.array [num_imgs, num_corners, 2]
            self.cns_flag = None  # indicate if corners can be detected, np.array [num_imgs]

        self.load_images(depth_only=depth_only)
        self.load_corners()
        self.load_cam_params()

    def init_pattern_points(self):
        col, row = self.pattern_size
        objp = np.zeros((col * row, 3), np.float32)
        objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * float(self.square_size)
        return objp

    def load_images(self, depth_only):
        file_path = '%s/%s_calib_snap' % (self.data_dir, self.cam)
        if 'kinect' in self.cam and depth_only:
            # only depth image is loaded
            imgs_d = []
            for i in range(self.num_frame):
                fname = '%s/depth%4i.png' % (file_path, i + self.start_idx)
                img_d = cv2.imread(fname, -1).astype(np.float32)
                imgs_d.append(img_d)
            self.imgs_d = np.stack(imgs_d, axis=0)
            print('[%s] only depth images are loaded.' % self.cam, 'shape ', self.imgs_d.shape)

        elif 'azure_kinect' in self.cam and not depth_only:
            imgs_d, imgs_c, imgs_gray = [], [], []
            for i in range(self.num_frame):
                fname = '%s/depth%04i.png' % (file_path, i + self.start_idx)
                # print(fname)
                img_d = cv2.imread(fname, -1).astype(np.float32)
                imgs_d.append(img_d)

                fname = '%s/infrared%04i.png' % (file_path, i + self.start_idx)
                # print(fname)
                img_i = np.clip(cv2.imread(fname, -1).astype(np.float32) * 0.25, 0, 255).astype(np.uint8)
                imgs_gray.append(img_i)

                fname = '%s/color%04i.jpg' % (file_path, i + self.start_idx)
                # print(fname)
                img_c = cv2.imread(fname)
                imgs_c.append(img_c)

            self.imgs_d = np.stack(imgs_d, axis=0)
            self.imgs_gray = np.stack(imgs_gray, axis=0)
            self.imgs_c = np.stack(imgs_c, axis=0)
            print('[%s] images are loaded. ' % self.cam,
                  'shape ', self.imgs_c.shape, self.imgs_gray.shape, self.imgs_d.shape)

        elif 'kinect_v2' in self.cam and not depth_only:
            imgs_d, imgs_c, imgs_gray = [], [], []
            for i in range(self.num_frame):
                fname = '%s/depth%04i.png' % (file_path, i + self.start_idx)
                # print(fname)
                img_d = cv2.imread(fname, -1).astype(np.float32)
                imgs_d.append(img_d)

                fname = '%s/infrared%04i.png' % (file_path, i + self.start_idx)
                # print(fname)
                img_i = np.clip(cv2.imread(fname, -1).astype(np.float32) * 0.15, 0, 255).astype(np.uint8)
                imgs_gray.append(img_i)

                fname = '%s/color%04i.jpg' % (file_path, i + self.start_idx)
                # print(fname)
                img_c = cv2.imread(fname)
                imgs_c.append(img_c)

            self.imgs_d = np.stack(imgs_d, axis=0)
            self.imgs_gray = np.stack(imgs_gray, axis=0)
            self.imgs_c = np.stack(imgs_c, axis=0)
            print('[%s] images are loaded. ' % self.cam,
                  'shape ', self.imgs_c.shape, self.imgs_gray.shape, self.imgs_d.shape)

        else:
            # polar or event camera
            imgs_gray = []
            for i in range(self.num_frame):
                if 'event' in self.cam:
                    fname = '%s/fullpic%04i.jpg' % (file_path, i + self.start_idx)
                else:
                    fname = '%s/polar0_%04i.jpg' % (file_path, i + self.start_idx)
                # print(fname)
                img_g = cv2.imread(fname)
                imgs_gray.append(img_g)
            self.imgs_gray = np.stack(imgs_gray, axis=0)
            print('[%s] images are loaded.' % self.cam, 'shape ', self.imgs_gray.shape)

    def load_cam_params(self):
        if 'kinect' in self.cam:
            with open('%s/intrinsic_param.pkl' % self.data_dir, 'rb') as f:
                data = pickle.load(f)
                self.intr_c = data['%s_color' % self.cam]
                self.intr_d = data['%s_depth' % self.cam]

            with open('%s/kinect_extrinsic_param.pkl' % self.data_dir, 'rb') as f:
                data = pickle.load(f)
                r, t = data['%s_d2c' % self.cam]
                self.T_d2c = Transform(r=r, t=t)
            print('[%s] camera params are loaded.' % self.cam)

        else:
            with open('%s/intrinsic_param.pkl' % self.data_dir, 'rb') as f:
                data = pickle.load(f)
                self.intr = data['%s' % self.cam]
            print('[%s] camera intrinsic params are loaded.' % self.cam)

    def detect_corners(self):
        col, row = self.pattern_size
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        if 'kinect' in self.cam:
            # kinect
            obj_points, img_points_c, img_points_i, flags = [], [], [], []
            for i in range(self.num_frame):
                img_c = cv2.cvtColor(self.imgs_c[i], cv2.COLOR_BGR2GRAY)
                img_i = self.imgs_gray[i]

                ret_i, corners_i = cv2.findChessboardCorners(img_i, (col, row), None)
                ret_c, corners_c = cv2.findChessboardCorners(img_c, (col, row), None)
                obj_points.append(self.pattern_points)
                if ret_i and ret_c:
                    flags.append(True)
                    corners2_i = cv2.cornerSubPix(img_i, corners_i, (5, 5), (-1, -1), criteria)
                    img_points_i.append(np.squeeze(corners2_i))
                    corners2_c = cv2.cornerSubPix(img_c, corners_c, (5, 5), (-1, -1), criteria)
                    img_points_c.append(np.squeeze(corners2_c))

                    # cv2.drawChessboardCorners(img_c, (col, row), corners2_c, ret_c)
                    # cv2.imshow('img_c', cv2.resize(img_c, (int(img_c.shape[1]/2), int(img_c.shape[0]/2))))
                    # cv2.waitKey(50)

                    # cv2.drawChessboardCorners(img_i, (col, row), corners2_i, ret_i)
                    # cv2.imshow('img_i', img_i)
                    # cv2.waitKey(50)
                else:
                    flags.append(False)
                    img_points_i.append(np.zeros_like(self.pattern_points[:, 0:2]))
                    img_points_c.append(np.zeros_like(self.pattern_points[:, 0:2]))
            cv2.destroyAllWindows()
            obj_points = np.stack(obj_points, axis=0)
            img_points_i = np.stack(img_points_i, axis=0)
            img_points_c = np.stack(img_points_c, axis=0)
            flags = np.asarray(flags)
            print('[%s] finish detecting corners, [%i True, %i False]'
                  % (self.cam, np.sum(flags==True), np.sum(flags==False)))

            # save as .pkl
            # file_path = '%s/%s_corners.pkl' % (self.data_dir, self.cam)
            data = [obj_points, img_points_c, img_points_i, flags]
            # with open(file_path, 'wb') as f:
            #     pickle.dump(data, f)
            #     print('saved as %s' % file_path)
            return data

        else:
            obj_points, img_points, flags = [], [], []
            for i in range(self.num_frame):
                img = self.imgs_gray[i]
                ret, corners = cv2.findChessboardCorners(img[:, :, 0], (col, row), None)
                flags.append(ret)
                obj_points.append(self.pattern_points)
                if ret:
                    corners2 = cv2.cornerSubPix(img[:, :, 0], corners, (5, 5), (-1, -1), criteria)
                    img_points.append(np.squeeze(corners2))

                    # cv2.drawChessboardCorners(img, (col, row), corners2, ret)
                    # cv2.imshow('img', cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2))))
                    # cv2.waitKey(50)
                else:
                    img_points.append(np.zeros_like(self.pattern_points[:, 0:2]))
            cv2.destroyAllWindows()
            obj_points = np.stack(obj_points, axis=0)
            img_points = np.stack(img_points, axis=0)
            flags = np.asarray(flags)
            print('[%s] finish detecting corners, [%i True, %i False]'
                  % (self.cam, np.sum(flags==True), np.sum(flags==False)))

            # save as .pkl
            # file_path = '%s/%s_corners.pkl' % (self.data_dir, self.cam)
            data = [obj_points, img_points, flags]
            # with open(file_path, 'wb') as f:
            #     pickle.dump(data, f)
            #     print('saved as %s' % file_path)
            return data

    def load_corners(self):
        if 'kinect' in self.cam:
            # detect corners
            self.cns_pattern, self.cns_c, self.cns_gray, self.cns_flag = self.detect_corners()
            print('shape ', self.cns_pattern.shape, self.cns_c.shape, self.cns_gray.shape, self.cns_flag.shape)

        else:
            # detect corners
            self.cns_pattern, self.cns_gray, self.cns_flag = self.detect_corners()
            print('shape ', self.cns_pattern.shape, self.cns_gray.shape, self.cns_flag.shape)


class onePairCamsData:
    def __init__(self, data_dir, cam1, cam2, square_size, pattern_size, start_idx, num_frame, depth_only):
        self.cam1 = cam1
        self.cam2 = cam2
        self.data_dir = data_dir
        self.pattern_size = pattern_size
        self.cam_pair_data = self.align_corners_pair(square_size, pattern_size, start_idx, num_frame, depth_only)

    def align_corners_pair(self, square_size, pattern_size, start_idx, num_frame, depth_only):
        cam1_data = oneCamData(self.data_dir, self.cam1, square_size, pattern_size, start_idx, num_frame, depth_only)
        cam2_data = oneCamData(self.data_dir, self.cam2, square_size, pattern_size, start_idx, num_frame, depth_only)

        flags = cam1_data.cns_flag & cam2_data.cns_flag
        cns_pattern = cam1_data.cns_pattern[flags]

        cns_gray1 = cam1_data.cns_gray[flags]
        imgs_gray1 = cam1_data.imgs_gray[flags]
        if 'kinect' in self.cam1:
            cns_c1 = cam1_data.cns_c[flags]
            imgs_c1 = cam1_data.imgs_c[flags]
            imgs_d1 = cam1_data.imgs_d[flags]
        else:
            imgs_c1, imgs_d1, cns_c1 = None, None, None

        cns_gray2 = cam2_data.cns_gray[flags]
        imgs_gray2 = cam2_data.imgs_gray[flags]
        if 'kinect' in self.cam2:
            cns_c2 = cam2_data.cns_c[flags]
            imgs_c2 = cam2_data.imgs_c[flags]
            imgs_d2 = cam2_data.imgs_d[flags]
        else:
            cns_c2, imgs_c2, imgs_d2 = None, None, None

        cns_gray1, cns_gray2, cns_c1, cns_c2 = self.flip_corners(cns_gray1, cns_gray2, cns_c1, cns_c2)
        print('---------------------------------------------------------')
        print('aligned [%s] and [%s], [%i True, %i False]' %
              (self.cam1, self.cam2, np.sum(flags==True), np.sum(flags==False)))

        data_cam_pairs = {
            'flags': flags,
            'pattern_size': cam1_data.pattern_size,
            'valid_num_frame': np.sum(flags).astype(np.int32),
            'num_frame': cam1_data.num_frame,
            'cns_pattern': cns_pattern,
            'cns_gray1': cns_gray1,
            'cns_c1': cns_c1,
            'imgs_gray1': imgs_gray1,
            'imgs_c1': imgs_c1,
            'imgs_d1': imgs_d1,
            'intr_params_1': cam1_data.intr,
            'intr_params_c1': cam1_data.intr_c,
            'intr_params_d1': cam1_data.intr_d,
            'cns_gray2': cns_gray2,
            'cns_c2': cns_c2,
            'imgs_gray2': imgs_gray2,
            'imgs_c2': imgs_c2,
            'imgs_d2': imgs_d2,
            'intr_params_2': cam2_data.intr,
            'intr_params_c2': cam2_data.intr_c,
            'intr_params_d2': cam2_data.intr_d}
        return data_cam_pairs

    def flip_corners(self, cns_gray1, cns_gray2, cns_c1, cns_c2):
        _cns_gray1, _cns_gray2 = cns_gray1.copy(), cns_gray2.copy()
        if cns_c1 is not None:
            _cns_c1 = cns_c1.copy()
        else:
            _cns_c1 = cns_c1
        if cns_c2 is not None:
            _cns_c2 = cns_c2.copy()
        else:
            _cns_c2 = cns_c2

        num_imgs = _cns_gray1.shape[0]
        for i in range(num_imgs):
            cn_gray1 = cns_gray1[i]
            cn_gray2 = cns_gray2[i]

            vec_1 = (cn_gray1[0, :] - cn_gray1[-1, :]) / np.linalg.norm(cn_gray1[0, :] - cn_gray1[-1, :])
            vec_2 = (cn_gray2[0, :] - cn_gray2[-1, :]) / np.linalg.norm(cn_gray2[0, :] - cn_gray2[-1, :])
            if np.dot(vec_1, vec_2) < 0:
                _cns_gray1[i] = cn_gray1[::-1]
                _cns_gray2[i] = cn_gray2[::-1]
                if cns_c1 is not None:
                    cn_c1 = cns_c1[i]
                    _cns_c1[i] = cn_c1[::-1]
                if cns_c2 is not None:
                    cn_c2 = cns_c2[i]
                    _cns_c2[i] = cn_c2[::-1]

        return _cns_gray1, _cns_gray2, _cns_c1, _cns_c2

    def observe_corners(self):
        data = self.cam_pair_data
        for j in range(data.get('valid_num_frame')):
            if 'kinect' in self.cam1:
                pass
            else:
                img_gray1 = data['imgs_gray1'][j]
                corners_gray1 = data['cns_gray1'][j]
                cv2.drawChessboardCorners(img_gray1, self.pattern_size, corners_gray1, True)
                cv2.imshow('img_gray1',
                           cv2.resize(img_gray1, (int(img_gray1.shape[1]/2), int(img_gray1.shape[0]/2))))
                cv2.waitKey()

            if 'kinect' in self.cam2:
                pass
            else:
                img_gray2 = data['imgs_gray2'][j]
                corners_gray2 = data['cns_gray2'][j]
                cv2.drawChessboardCorners(img_gray2, self.pattern_size, corners_gray2, True)
                cv2.imshow('img_gray2',
                           cv2.resize(img_gray2, (int(img_gray2.shape[1]/2), int(img_gray2.shape[0]/2))))
                cv2.waitKey()

            if data['imgs_c1'] is not None:
                img_c1 = data['imgs_c1'][j]
                corners_c1 = data['cns_c1'][j]
                cv2.drawChessboardCorners(img_c1, self.pattern_size, corners_c1, True)
                cv2.imshow('img_color1', cv2.resize(img_c1, (int(img_c1.shape[1]/2), int(img_c1.shape[0]/2))))
                cv2.waitKey()

            if data['imgs_c2'] is not None:
                img_c2 = data['imgs_c2'][j]
                corners_c2 = data['cns_c2'][j]
                cv2.drawChessboardCorners(img_c2, self.pattern_size, corners_c2, True)
                cv2.imshow('img_color2', cv2.resize(img_c2, (int(img_c2.shape[1]/2), int(img_c2.shape[0]/2))))
                cv2.waitKey()
        cv2.destroyAllWindows()


def estimate_transform(data_cam_pairs, cam1, cam2):
    print('---------------------------------------------------------')
    if 'kinect' not in cam1:
        ls_T_cam1w = get_img2w_transform(data_cam_pairs['cns_gray1'], data_cam_pairs['cns_pattern'],
                                         data_cam_pairs['intr_params_1'], cam1)
    else:
        ls_T_cam1w = get_depth2w_transform(data_cam_pairs['imgs_d1'], data_cam_pairs['cns_gray1'],
                                           data_cam_pairs['cns_pattern'], data_cam_pairs['intr_params_d1'],
                                           data_cam_pairs['pattern_size'], cam1)

    if 'kinect' not in cam2:
        ls_T_cam2w = get_img2w_transform(data_cam_pairs['cns_gray2'], data_cam_pairs['cns_pattern'],
                                         data_cam_pairs['intr_params_2'], cam2)
    else:
        ls_T_cam2w = get_depth2w_transform(data_cam_pairs['imgs_d2'], data_cam_pairs['cns_gray2'],
                                           data_cam_pairs['cns_pattern'], data_cam_pairs['intr_params_d2'],
                                           data_cam_pairs['pattern_size'], cam2)

    # for i in range(data_cam_pairs['valid_num_frame']):
    #     T_cam1w = ls_T_cam1w[i]
    #     cn_pattern = data_cam_pairs['cns_pattern'][i]
    #     pts_cam1 = T_cam1w.transform(cn_pattern)
    #     if 'kinect' in cam1:
    #         uv_cam1 = projection(pts_cam1, data_cam_pairs['intr_params_d1'], False)
    #     else:
    #         uv_cam1 = projection(pts_cam1, data_cam_pairs['intr_params_1'], False)
    #     error = np.mean(np.sqrt(np.sum((data_cam_pairs.get('cns_gray1')[i] - uv_cam1) ** 2, axis=1)))
    #     print('error:', error)
    #
    # print('---------------------------------------------------------')
    # for i in range(data_cam_pairs['valid_num_frame']):
    #     T_cam2w = ls_T_cam2w[i]
    #     cn_pattern = data_cam_pairs['cns_pattern'][i]
    #     pts_cam2 = T_cam2w.transform(cn_pattern)
    #     if 'kinect' in cam2:
    #         uv_cam2 = projection(pts_cam2, data_cam_pairs['intr_params_d2'], False)
    #     else:
    #         uv_cam2 = projection(pts_cam2, data_cam_pairs['intr_params_2'], False)
    #     error = np.mean(np.sqrt(np.sum((data_cam_pairs.get('cns_gray2')[i] - uv_cam2) ** 2, axis=1)))
    #     print('error:', error)


    # get transform
    ls_T_cam1cam2 = []  # transform from cam2 to cam1
    for i in range(data_cam_pairs['valid_num_frame']):
        # print('-------------------------------------------------------')
        T_cam1w = ls_T_cam1w[i]
        # print(T_cam1w.r, '\n', T_cam1w.t, '\n')
        T_cam2w = ls_T_cam2w[i]
        # print(T_cam2w.r, '\n', T_cam2w.t, '\n')
        _T_cam1cam2 = T_cam1w * T_cam2w.inv()
        # print(_T_cam1cam2.r, '\n', _T_cam1cam2.t, '\n')
        ls_T_cam1cam2.append(_T_cam1cam2)

    # clustering
    T_cam1cam2 = cluster_pose(ls_T_cam1cam2)
    print(T_cam1cam2.r, '\n', T_cam1cam2.t, '\n')

    # check errors
    for i in range(data_cam_pairs['valid_num_frame']):
        T_cam2w = ls_T_cam2w[i]
        cn_pattern = data_cam_pairs['cns_pattern'][i]
        pts_cam2 = T_cam2w.transform(cn_pattern)
        pts_cam1 = T_cam1cam2.transform(pts_cam2)

        if 'kinect' in cam1:
            uv_cam1 = projection(pts_cam1, data_cam_pairs['intr_params_d1'], False)[:, 0:2]
        else:
            uv_cam1 = projection(pts_cam1, data_cam_pairs['intr_params_1'], False)[:, 0:2]
        error = np.mean(np.sqrt(np.sum((data_cam_pairs.get('cns_gray1')[i] - uv_cam1) ** 2, axis=1)))
        print('error:', error)

    return T_cam1cam2


if __name__ == '__main__':
    date = '1101'
    data_dir = '/data/shihao/data_event/calib_%s' % date
    # cam1 = 'event_camera'
    # cam2 = 'azure_kinect_0'
    # start_idx = 60
    square_size = 60.
    pattern_size = (8, 11)
    num_frame = 20
    depth_only = False

    # data = onePairCamsData(data_dir, cam1, cam2, square_size, pattern_size, start_idx, num_frame, depth_only)
    # data.observe_corners()
    # T_cam2cam1 = estimate_transform(data.cam_pair_data, cam1, cam2)

    extr_param = {}
    # from cam1 to cam2
    cam_pairs = [('azure_kinect_2', 'kinect_v2_2', 0),
                 ('azure_kinect_0', 'azure_kinect_2', 20),
                 ('polar', 'azure_kinect_0', 40),
                 ('event_camera', 'azure_kinect_0', 60),
                 ('azure_kinect_0', 'azure_kinect_1', 80),
                 ('azure_kinect_1', 'kinect_v2_1', 100)]

    for (cam1, cam2, start_idx) in cam_pairs:
        data = onePairCamsData(data_dir, cam1, cam2, square_size, pattern_size, start_idx, num_frame, depth_only)
        T_cam1cam2 = estimate_transform(data.cam_pair_data, cam1, cam2)
        extr_param['%s-%s' % (cam1, cam2)] = (T_cam1cam2.r, T_cam1cam2.t)

    for k, v in extr_param.items():
        print(k, v)
    with open('%s/extrinsic_param_%s.pkl' % (data_dir, date), 'wb') as f:
        pickle.dump(extr_param, f)
    print('save as %s/extrinsic_param_%s.pkl' % (data_dir, date))
