# contains functions defined in the utils file as well
import os
import cv2
import pickle
import numpy as np
import sklearn.cluster

class Transform:
    def __init__(self, r=np.eye(3, dtype='float'), t=np.zeros(3, 'float'), s=np.ones(3, 'float')):
        self.r = r.copy()
        self.t = t.reshape(-1).copy()
        self.s = s.copy()

    def __mul__(self, other):
        r = np.dot(self.r, other.r)
        t = np.dot(self.r, other.t * self.s) + self.t
        if not hasattr(other, 's'):
            other.s = np.ones(3, 'float').copy()
        s = other.s.copy()
        return Transform(r, t, s)

    def inv(self):
        r = self.r.T
        t = - np.dot(self.r.T, self.t)
        return Transform(r, t)

    def transform(self, xyz):
        if not hasattr(self, 's'):
            self.s = np.ones(3, 'float').copy()
        assert xyz.shape[-1] == 3
        assert len(self.s) == 3
        return np.dot(xyz * self.s, self.r.T) + self.t

    def getmat4(self):
        M = np.eye(4)
        M[:3, :3] = self.r * self.s
        M[:3, 3] = self.t
        return M


def substract_foreground(bg_img, img, min_depth=500, max_depth=4000):
    import scipy.ndimage
    bg_img = bg_img.astype(np.float32)
    img = img.astype(np.float32)

    # substraction
    img_bin = (np.abs(bg_img - img) > 60) * (img > min_depth) * (img < max_depth)
    img_bin = scipy.ndimage.median_filter(img_bin, 5)

    # connected component (useless)
    num_labels, labels_im = cv2.connectedComponents(img_bin.astype(np.uint8))
    # label 0 is the background with the largest area, second largest is the foreground
    areas = [np.sum(labels_im==(i+1)) for i in range(num_labels-1)]
    max_label = areas.index(max(areas))
    img_bin = labels_im == (max_label + 1)

    # plt.figure()
    # plt.imshow(img_bin, cmap='gray')
    # plt.show()
    return img_bin


def convert_params2mtx(params):
    # (fx, fy, cx, cy, h, w, k1, k2, p1, p2, k3, k4, k5, k6)
    fx, fy, cx, cy, h, w, k1, k2, p1, p2, k3, k4, k5, k6 = params
    mtx = np.zeros([3, 3])
    mtx[0, 0] = fx
    mtx[1, 1] = fy
    mtx[0, 2] = cx
    mtx[1, 2] = cy
    dist = (k1, k2, p1, p2, k3, k4, k5, k6)
    return mtx, dist


def get_img2w_transform(cns, cns_pattern, intr_params, cam):
    mtx, dist = convert_params2mtx(intr_params)
    ls_T_imgw = []
    for i in range(cns.shape[0]):
        retval, rvec, tvec = cv2.solvePnP(objectPoints=cns_pattern[i], imagePoints=cns[i],
                                          cameraMatrix=mtx, distCoeffs=dist)
        r = cv2.Rodrigues(rvec)[0]
        t = np.squeeze(tvec)
        ls_T_imgw.append(Transform(r, t))
        # print('[%s] R:\n' % cam, r, '\nt:\n', t)
    return ls_T_imgw


def get_depth2w_transform(depths, cns, cns_pattern, intr_param, pattern_size, cam):
    d1, d2 = 1000, 4000
    ls_T_dw = []  # world to depth camera coordinate
    for i in range(cns.shape[0]):
        cn = cns[i]
        depth = depths[i]
        cn_pattern = cns_pattern[i]

        # extract plane
        corners = extend_chessboard_corners(cn, pattern_size)
        mask = np.zeros(depth.shape, 'uint8')
        cv2.drawContours(mask, [corners], 0, 255, -1)
        mask = np.bitwise_and(np.bitwise_and(mask > 0, depth > d1), depth < d2)
        pts = depth2pts(depth, intr_param, False)  # xyz [H, W, 3]
        pts = pts[mask]  # [N, 3]
        # fit plane
        ct, norm = fit_plane_robust(pts)
        # intersection
        vecs = uv2xyz1(cn, intr_param, False)  # xyz, z=1, [N, 3]
        pts1 = intersect_lines_plane((np.zeros(3), vecs), (ct, norm))
        # alignment
        pts0 = cn_pattern
        r, t = rigid_align_3d(pts0, pts1)
        # print('[%s] R:\n' % cam, r, '\nt:\n', t)
        ls_T_dw.append(Transform(r, t))
    return ls_T_dw


def extend_chessboard_corners(cn, pattern_size):
    """
    Extend 4 corners of chessboard

    Parameters
    ----------
    cn: (M,2) array. M is the number of corner points on chessboard.

    Return
    ------
    corners: (4,2) int32 array. 4 outer corners

    Hint
    ----
    :type cn: np.ndarray
    """
    assert isinstance(cn, np.ndarray)

    w, h = pattern_size
    p1 = cn[0]
    p1i = cn[w+1]
    p1o = p1 + (p1-p1i)

    p2 = cn[w-1]
    p2i = cn[2*w-2]
    p2o = p2 + (p2-p2i)

    p3 = cn[-1]
    p3i = cn[-w-2]
    p3o = p3 + (p3-p3i)

    p4 = cn[-w]
    p4i = cn[-2*w+1]
    p4o = p4 + (p4-p4i)

    corners = np.stack((p1o, p2o, p3o, p4o)).astype('int32')
    return corners


def depth2uvd(depth):
    h, w = depth.shape
    u, v = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    uvd = np.stack([u, v, depth], axis=2)  # [H, W, 3]
    return uvd


def depth2pts(depth, intr_param, simple_mode=False):
    uvd = depth2uvd(depth)
    # intr_param (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
    fx, fy, cx, cy = intr_param[0:4]
    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_pp = (uvd[:, :, 0] - cx) / fx
        y_pp = (uvd[:, :, 1] - cy) / fy
        r2 = x_pp ** 2 + y_pp ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        a = a + (a == 0)
        di = b / a

        x_p = x_pp * di
        y_p = y_pp * di

        x = uvd[:, :, 2] * (x_p - p2 * (y_p ** 2 + 3 * x_p ** 2) - p1 * 2 * x_p * y_p)
        y = uvd[:, :, 2] * (y_p - p1 * (x_p ** 2 + 3 * y_p ** 2) - p2 * 2 * x_p * y_p)
        z = uvd[:, :, 2]

        return np.stack([x, y, z], axis=2)
    else:
        x = uvd[:, :, 2] * (uvd[:, :, 0] - cx) / fx
        y = uvd[:, :, 2] * (uvd[:, :, 1] - cy) / fy
        z = uvd[:, :, 2]
        return np.stack([x, y, z], axis=2)


def uv2xyz1(uv, intr_param, simple_mode=False):
    """
    convert uvd coordinates to xyz, z=1
    return:
        points in xyz coordinates, shape [N, 3]
    """
    # intr_param (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
    assert uv.shape[1] == 2
    fx, fy, cx, cy = intr_param[0:4]
    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_pp = (uv[:, 0] - cx) / fx
        y_pp = (uv[:, 1] - cy) / fy
        r2 = x_pp ** 2 + y_pp ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        a = a + (a == 0)
        di = b / a

        x_p = x_pp * di
        y_p = y_pp * di

        x = x_p - p2 * (y_p ** 2 + 3 * x_p ** 2) - p1 * 2 * x_p * y_p
        y = y_p - p1 * (x_p ** 2 + 3 * y_p ** 2) - p2 * 2 * x_p * y_p
        z = np.ones_like(uv[:, 0])

        return np.stack([x, y, z], axis=1)
    else:
        x = (uv[:, 0] - cx) / fx
        y = (uv[:, 1] - cy) / fy
        z = np.ones_like(uv[:, 0])
        return np.stack([x, y, z], axis=1)


def projection(xyz, intr_param, simple_mode=False):
    # xyz: [N, 3]
    # intr_param: (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
    assert xyz.shape[1] == 3
    fx, fy, cx, cy = intr_param[0:4]

    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_p = xyz[:, 0] / xyz[:, 2]
        y_p = xyz[:, 1] / xyz[:, 2]
        r2 = x_p ** 2 + y_p ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        b = b + (b == 0)
        d = a / b

        x_pp = x_p * d + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p ** 2)
        y_pp = y_p * d + p1 * (r2 + 2 * y_p ** 2) + 2 * p2 * x_p * y_p

        u = fx * x_pp + cx
        v = fy * y_pp + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)
    else:
        u = xyz[:, 0] / xyz[:, 2] * fx + cx
        v = xyz[:, 1] / xyz[:, 2] * fy + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)


def unprojection(uvd, intr_param, simple_mode=False):
    # uvd: [N, 3]
    # cam_param: (fx, fy, cx, cy)
    # dist_coeff: (k1, k2, p1, p2, k3, k4, k5, k6)
    assert uvd.shape[1] == 3
    fx, fy, cx, cy = intr_param[0:4]

    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_pp = (uvd[:, 0] - cx) / fx
        y_pp = (uvd[:, 1] - cy) / fy
        r2 = x_pp ** 2 + y_pp ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        a = a + (a == 0)
        di = b / a

        x_p = x_pp * di
        y_p = y_pp * di

        x = uvd[:, 2] * (x_p - p2 * (y_p ** 2 + 3 * x_p ** 2) - p1 * 2 * x_p * y_p)
        y = uvd[:, 2] * (y_p - p1 * (x_p ** 2 + 3 * y_p ** 2) - p2 * 2 * x_p * y_p)
        z = uvd[:, 2]

        return np.stack([x, y, z], axis=1)
    else:
        x = uvd[:, 2] * (uvd[:, 0] - cx) / fx
        y = uvd[:, 2] * (uvd[:, 1] - cy) / fy
        z = uvd[:, 2]
        return np.stack([x, y, z], axis=1)


def fit_plane(pts):
    """
    Description
    ===========
    Fit a plane from 3D points.

    Parameters
    ==========
    :param pts: (M,3) array. 3D points

    Returns
    =======
    :return ct: (3,) array. Center of the plane.
    :return norm: (3,) array. Normal of the plane.
    :return err: (M,) array. Fitting error of each point.
    """
    ct = np.mean(pts, axis=0)
    A = pts - ct
    ATA = A.T.dot(A)
    w, V = np.linalg.eig(ATA)
    i = np.argmin(w)
    norm = V[:, i]
    norm *= 1./np.linalg.norm(norm)
    err = A.dot(norm)
    return ct, norm, err


def fit_plane_robust(pts, percent=0.7):
    """
    Fit 3D plane robustly using given percent inliers.
    Please refer to fit_plane for details.

    Paramers
    ========
    :param pts: (M,3) array. 3D points
    :param percent: float scalar. Percentage of inlying points used.

    Returns
    =======
    :return ct: (3,) array. Center of the plane.
    :return norm: (3,) array. Normal of the plane.
    """
    ct, norm, err = fit_plane(pts)
    err = np.abs(err)
    th = np.sort(err)[int(len(err)*percent)]
    idx = err < th
    pts = pts[idx]
    ct, norm, err = fit_plane(pts)
    return ct, norm


def intersect_lines_plane(lines, plane):
    """
    Description
    ===========
    Calculate intersection points between lines and plane.

    Algorithm
    =========
    a point on line is parametered as: :math:`v_l t + p_l`,

    we have :math:`(v_l t + p_l - p_p) \cdot n_p = 0`,

    and then :math:`t = (p_p - p_l) \cdot n_p / (v_l \cdot n_p)`,

    where _p means plane, and _l means line.

    Parameters
    ==========
    :param lines: (p_l,v_l) tuple

        p_l: (M,3) or (3,) array. 3D base points of lines

        v_l: (M,3) array. 3D directions of lines

    :param plane: (p_p,n_p) tuple

        p_p (3,) array. 3D base point of a plane

        n_p (3,) array. 3D direction of a plane

    Returns
    =======
    :return: (M,3) array. Intersection points
    """

    p_l, v_l = lines
    p_p, n_p = plane

    t = (p_p - p_l).dot(n_p) / (v_l.dot(n_p))
    pts = p_l + v_l * t.reshape((-1, 1))
    return pts


def rigid_align_3d(X, Y):
    """
    Description
    ===========
    Estimation a rigid transformation to align 2 3D point set.

    :math:`Y=R X + t`,

    where :math:`R` is a rotation and :math:`t` is a translation.

    Parameters
    ==========
    :param X: (M,3) array. src point set.
    :param Y: (M,3) array. dst point set.

    Returns
    =======
    :return R: (3,3) array
    :return t: (3,) array

    :rtype: (np.ndarray, np.ndarray)
    """
    Xbar = np.mean(X, axis=0)
    Ybar = np.mean(Y, axis=0)
    X1 = X - Xbar
    Y1 = Y - Ybar
    S = X1.T.dot(Y1)
    U, s, VT = np.linalg.svd(S)
    V = VT.T
    UT = U.T
    R = V.dot(np.diag([1., 1., np.linalg.det(V.dot(UT))])).dot(UT)
    t = Ybar - R.dot(Xbar)
    return R, t


def cluster_pose(ls_T):
    """
    :type ls_T: list[Transform]
    """
    Rs = [T.r for T in ls_T]
    ts = [T.t for T in ls_T]

    # cluster t
    meanshift_t = sklearn.cluster.MeanShift(bandwidth=1000, bin_seeding=True)
    meanshift_t.fit(ts)
    print(meanshift_t.labels_)
    assert np.count_nonzero(meanshift_t.labels_ == 0) > (0.7 * len(ts))
    t = meanshift_t.cluster_centers_[0]
    print(t)

    # cluster R
    _Rs = np.array(Rs)[meanshift_t.labels_ == 0]
    _Rs = _Rs.reshape((-1, 9))
    meanshift_R = sklearn.cluster.MeanShift(bandwidth=0.1)
    meanshift_R.fit(_Rs)
    print(meanshift_R.labels_)
    _tmp = meanshift_t.labels_ == 0
    assert isinstance(_tmp, np.ndarray)
    assert np.count_nonzero(_tmp) > (0.7 * len(_Rs))
    R = meanshift_R.cluster_centers_[0].reshape((3, 3))

    # normalize
    R = r2R(R2r(R))
    return Transform(R, t)


# 3D transform
def r2R(r):
    R, _ = cv2.Rodrigues(r)
    return R


def R2r(R):
    r, _ = cv2.Rodrigues(R)
    return r

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

        elif 'kinect' in self.cam and not depth_only:
            imgs_d, imgs_c, imgs_gray = [], [], []
            for i in range(self.num_frame):
                fname = '%s/depth%04i.png' % (file_path, i + self.start_idx)
                # print(fname)
                img_d = cv2.imread(fname, -1).astype(np.float32)
                imgs_d.append(img_d)

                fname = '%s/infrared%04i.png' % (file_path, i + self.start_idx)
                # print(fname)
                img_i = np.clip(cv2.imread(fname, -1).astype(np.float32) * 0.2, 0, 255).astype(np.uint8)
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
                self.intr_c = data['azure_kinect1_2_calib_snap_depth']
                self.intr_d = data['azure_kinect1_2_calib_snap_color']

            with open('%s/kinect_extrinsic_param.pkl' % self.data_dir, 'rb') as f:
                data = pickle.load(f)
                r, t = data['azure_kinect1_2_calib_snap_d2c']
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

                    cv2.drawChessboardCorners(img_c, (col, row), corners2_c, ret_c)
                    cv2.imshow('img_c', cv2.resize(img_c, (int(img_c.shape[1]/2), int(img_c.shape[0]/2))))
                    cv2.waitKey(50)

                    cv2.drawChessboardCorners(img_i, (col, row), corners2_i, ret_i)
                    cv2.imshow('img_i', img_i)
                    cv2.waitKey(50)
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

                    cv2.drawChessboardCorners(img, (col, row), corners2, ret)
                    cv2.imshow('img', cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2))))
                    cv2.waitKey(50)
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
        print('-------------------------------------------------------')
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

# assert (cam1, cam2) in [('azure_kinect_2', 'kinect_v2_2'), ('azure_kinect_2', 'azure_kinect_0'),
#                         ('polar', 'azure_kinect_0'), ('event_camera', 'azure_kinect_0'),
#                         ('azure_kinect_1', 'azure_kinect_0'), ('azure_kinect_1', 'kinect_v2_1')]

data_dir = ".\AzureKinectRecord_30_05"
# cam1 = 'event_camera'
# cam2 = 'azure_kinect_0'
# start_idx = 60
col = 11
row = 8
square_size = 59
pattern_size = (8, 11)
num_frame = 25
depth_only = False

# data = onePairCamsData(data_dir, cam1, cam2, square_size, pattern_size, start_idx, num_frame, depth_only)
# data.observe_corners()
# T_cam2cam1 = estimate_transform(data.cam_pair_data, cam1, cam2)

extr_param = {}
# from cam1 to cam2
cam_pairs = [
    # ('azure_kinect1_2', 'azure_kinect2_3', 0),
            #  ('azure_kinect2_3', 'azure_kinect2_2', 26),
             ('azure_kinect2_2', 'azure_kinect3_4', 51),
             ('azure_kinect3_5', 'azure_kinect3_4', 81),
            #  ('azure_kinect3_5', 'azure_kinect1_3', 105),
            #  ('azure_kinect1_3', 'azure_kinect1_2', 131)
             ]

for (cam1, cam2, start_idx) in cam_pairs:
    data = onePairCamsData(data_dir, cam1, cam2, square_size, pattern_size, start_idx, num_frame, depth_only)
    T_cam1cam2 = estimate_transform(data.cam_pair_data, cam1, cam2)
    extr_param['%s-%s' % (cam1, cam2)] = (T_cam1cam2.r, T_cam1cam2.t)

for k, v in extr_param.items():
    print(k, v)
with open('%s/extrinsic_param.pkl' % data_dir, 'wb') as f:
    pickle.dump(extr_param, f)

# intr_param = pickle.load(open('%s/intrinsic_param.pkl' % data_dir, 'rb'))
# print(intr_param)

# extr_param = pickle.load(open('%s/kinect_extrinsic_param.pkl' % data_dir, 'rb'))
# print(extr_param)


