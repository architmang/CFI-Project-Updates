# contains functions defined in the utils file as well
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# 3D transform
def r2R(r):
    R, _ = cv2.Rodrigues(r)
    return R

def R2r(R):
    r, _ = cv2.Rodrigues(R)
    return r

def to_xyz(depth_img, mtx):
    rows, cols = depth_img.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    z = depth_img / 1000.  # convert depth to meters
    x = (c - mtx[0, 2]) * z / mtx[0, 0]
    y = (r - mtx[1, 2]) * z / mtx[1, 1]
    xyz_points = np.stack((x, y, z), axis=-1)
    return xyz_points

def transform_points(points, transformation):
    # Reshape the points to (N, 3)
    reshaped_points = points.reshape(-1, 3)

    # Append 1's to the points to convert them to homogeneous coordinates
    homogeneous_points = np.concatenate([reshaped_points, np.ones((reshaped_points.shape[0], 1))], axis=1)

    # Transform the points
    transformed_points = np.dot(transformation, homogeneous_points.T).T

    # Convert the points back to Cartesian coordinates
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3, np.newaxis]

    # Reshape the points back to the original shape
    transformed_points = transformed_points.reshape(*points.shape)

    return transformed_points

def create_transformation_matrix(R, T):
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = np.squeeze(T)
    return M

def inverse_extrinsic_params(R_12, t_12):
    """
    Compute the inverse of the extrinsic parameters.
    R_12: rotation matrix from camera 1 to camera 2
    t_12: translation vector from camera 1 to camera 2
    Returns: R_21 and t_21
    """
    # Compute the inverse rotation matrix
    R_21 = np.transpose(R_12)
    
    # Compute the inverse translation vector
    t_21 = -np.matmul(R_21, t_12)
    
    return R_21, t_21

def from_xyz_to_uvd(xyz_points, new_mtx_d):
    fx = new_mtx_d[0,0]
    fy = new_mtx_d[1,1]
    cx = new_mtx_d[0,2]
    cy = new_mtx_d[1,2]

    uvd_points = []

    for i in range(xyz_points.shape[0]):
        for j in range(xyz_points.shape[1]):
            # Ensure point is not None and does not contain NaN values
                X, Y, Z = xyz_points[i][j]
                print(f"X, Y, Z is {X, Y, Z}")
                if Z != 0: # To avoid division by zero
                    u = fx * (X / Z) + cx
                    v = fy * (Y / Z) + cy
                    depth = Z
                    uvd_points.append((u, v, depth))
                else:
                    uvd_points.append((0, 0, 0))
            
    return np.array(uvd_points)

root_dir = "./AzureKinectRecord_30_05/1"
master_dir = "./AzureKinectRecord_30_05/"

intr_param = pickle.load(open(master_dir + "/intrinsic_param.pkl", 'rb'))
kinect_extr_param = pickle.load(open(master_dir + "/kinect_extrinsic_param.pkl", 'rb'))
extr_param = pickle.load(open(master_dir + "/extrinsic_param.pkl", 'rb'))

# Define the camera list
cam_list = ['azure_kinect1_2','azure_kinect1_3','azure_kinect2_4', 
            'azure_kinect2_5', 'azure_kinect3_2', 'azure_kinect3_3']

frame_idx = 20  # Specify the frame index for depth-to-color registration

all_xyz_points = []

for cam in cam_list:
    print(cam)
    dir = f"{root_dir}/{cam}"

    # Read depth and color images for the specified frame
    depth_img = cv2.imread(f"{dir}/depth/depth{frame_idx:04d}.png", -1)

    if depth_img is None:
        print("Error reading depth image.")
        continue

    # Retrieve camera intrinsic parameters
    mtx_d = np.array([[intr_param['%s_depth' % cam][0], 0, intr_param['%s_depth' % cam][2]],
                      [0, intr_param['%s_depth' % cam][1], intr_param['%s_depth' % cam][3]],
                      [0, 0, 1]])

    dist_d = intr_param['%s_depth' % cam][6:14]

    h,  w = depth_img.shape[:2]
    new_mtx_d, _ = cv2.getOptimalNewCameraMatrix(mtx_d, dist_d, (w,h), 0, (w,h))

    # Convert depth to xyz coordinates
    xyz_points = to_xyz(depth_img, new_mtx_d)

    if cam == 'azure_kinect1_2':
        all_xyz_points.append(xyz_points)
        
    if cam == 'azure_kinect1_3':
        # Transform coordinates to camera 1
        d2c_transformation = create_transformation_matrix(kinect_extr_param['%s_d2c' % cam][0],
                                                    kinect_extr_param['%s_d2c' % cam][1])

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect1_2', cam)][0],
                    extr_param['%s-%s' % ('azure_kinect1_2', cam)][1])
        transformation_to_cam1 = create_transformation_matrix( r, t)

        # Apply the depth-to-color transformation
        xyz_points_d2c = transform_points(xyz_points, d2c_transformation)        # Transform the coordinates to camera 1's frame


        xyz_points_cam1 = transform_points(xyz_points_d2c, transformation_to_cam1)

        all_xyz_points.append(xyz_points_cam1)

    if cam == 'azure_kinect3_2':
        # Transform coordinates to camera 1
        d2c_transformation = create_transformation_matrix(kinect_extr_param['%s_d2c' % cam][0],
                                                    kinect_extr_param['%s_d2c' % cam][1])

        transformation_to_cam1 = create_transformation_matrix(extr_param['%s-%s' % (cam, 'azure_kinect1_2')][0],
                                                        extr_param['%s-%s' % (cam, 'azure_kinect1_2')][1])

        # Apply the depth-to-color transformation
        xyz_points_d2c = transform_points(xyz_points, d2c_transformation)        # Transform the coordinates to camera 1's frame


        xyz_points_cam1 = transform_points(xyz_points_d2c, transformation_to_cam1)

        all_xyz_points.append(xyz_points_cam1)

    if cam == 'azure_kinect2_4':
        # Transform coordinates to camera 1
        d2c_transformation = create_transformation_matrix(kinect_extr_param['%s_d2c' % cam][0],
                                                    kinect_extr_param['%s_d2c' % cam][1])

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect1_3', cam)][0],
                    extr_param['%s-%s' % ('azure_kinect1_3', cam)][1])
        transformation_to_cam1a = create_transformation_matrix( r, t)

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][0],
                    extr_param['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][1])
        transformation_to_cam1b = create_transformation_matrix( r, t)

        # Apply the depth-to-color transformation
        xyz_points_d2c = transform_points(xyz_points, d2c_transformation)        # Transform the coordinates to camera 1's frame
        xyz_points_cam1 = transform_points(xyz_points_d2c, transformation_to_cam1a)
        xyz_points_cam1 = transform_points(xyz_points_cam1, transformation_to_cam1b)

        all_xyz_points.append(xyz_points_cam1)

    if cam == 'azure_kinect2_5':
        # Transform coordinates to camera 1
        d2c_transformation = create_transformation_matrix(kinect_extr_param['%s_d2c' % cam][0],
                                                    kinect_extr_param['%s_d2c' % cam][1])

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect2_4', 'azure_kinect2_5')][0],
                    extr_param['%s-%s' % ('azure_kinect2_4', 'azure_kinect2_5')][1])
        transformation_to_cam1a = create_transformation_matrix( r, t)

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect1_3', 'azure_kinect2_4')][0],
                    extr_param['%s-%s' % ('azure_kinect1_3', 'azure_kinect2_4')][1])
        transformation_to_cam1b = create_transformation_matrix( r, t)

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][0],
                    extr_param['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][1])
        transformation_to_cam1c = create_transformation_matrix( r, t)

        # Apply the depth-to-color transformation
        xyz_points_d2c = transform_points(xyz_points, d2c_transformation)        # Transform the coordinates to camera 1's frame
        xyz_points_cam1 = transform_points(xyz_points_d2c, transformation_to_cam1a)
        xyz_points_cam1 = transform_points(xyz_points_cam1, transformation_to_cam1b)
        xyz_points_cam1 = transform_points(xyz_points_cam1, transformation_to_cam1c)

        all_xyz_points.append(xyz_points_cam1)

    if cam == 'azure_kinect3_3':
        # Transform coordinates to camera 1
        d2c_transformation = create_transformation_matrix(kinect_extr_param['%s_d2c' % cam][0],
                                                    kinect_extr_param['%s_d2c' % cam][1])

        r,t = extr_param['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][0], extr_param['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][1]
        transformation_to_cam1a = create_transformation_matrix( r, t)

        r,t = extr_param['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][0], extr_param['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][1]
        transformation_to_cam1b = create_transformation_matrix( r, t)

        # Apply the depth-to-color transformation
        xyz_points_d2c = transform_points(xyz_points, d2c_transformation)        # Transform the coordinates to camera 1's frame
        xyz_points_cam1 = transform_points(xyz_points_d2c, transformation_to_cam1a)
        xyz_points_cam1 = transform_points(xyz_points_cam1, transformation_to_cam1b)

        all_xyz_points.append(xyz_points_cam1)
    

# Compute the average point cloud
avg_xyz_points = np.mean(all_xyz_points, axis=0)
# print("Shape of avg_xyz_points: ", avg_xyz_points)

# Convert xyz to uvd and display
# avg_uvd_points = from_xyz_to_uvd(avg_xyz_points, new_mtx_d)

# Displaying a depth map from the uvd points
# depth_map = avg_uvd_points[..., 2]
# cv2.imshow('Depth Map', depth_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Separate x, y, z coordinates for plotting
xs = avg_xyz_points[:, 0]
ys = avg_xyz_points[:, 1]
zs = avg_xyz_points[:, 2]

ax.scatter(xs, ys, zs)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()




    if cam == 'azure_kinect1_2':
        all_xyz_points.append(xyz_points)
        
    if cam == 'azure_kinect1_3':
        # Transform coordinates to camera 1
        d2c_transformation = create_transformation_matrix(kinect_extr_param['%s_d2c' % cam][0],
                                                    kinect_extr_param['%s_d2c' % cam][1])

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect1_2', cam)][0],
                    extr_param['%s-%s' % ('azure_kinect1_2', cam)][1])
        transformation_to_cam1 = create_transformation_matrix( r, t)

        # Apply the depth-to-color transformation
        xyz_points_d2c = transform_points(xyz_points, d2c_transformation)        # Transform the coordinates to camera 1's frame


        xyz_points_cam1 = transform_points(xyz_points_d2c, transformation_to_cam1)

        all_xyz_points.append(xyz_points_cam1)

    if cam == 'azure_kinect3_2':
        # Transform coordinates to camera 1
        d2c_transformation = create_transformation_matrix(kinect_extr_param['%s_d2c' % cam][0],
                                                    kinect_extr_param['%s_d2c' % cam][1])

        transformation_to_cam1 = create_transformation_matrix(extr_param['%s-%s' % (cam, 'azure_kinect1_2')][0],
                                                        extr_param['%s-%s' % (cam, 'azure_kinect1_2')][1])

        # Apply the depth-to-color transformation
        xyz_points_d2c = transform_points(xyz_points, d2c_transformation)        # Transform the coordinates to camera 1's frame


        xyz_points_cam1 = transform_points(xyz_points_d2c, transformation_to_cam1)

        all_xyz_points.append(xyz_points_cam1)

    if cam == 'azure_kinect2_4':
        # Transform coordinates to camera 1
        d2c_transformation = create_transformation_matrix(kinect_extr_param['%s_d2c' % cam][0],
                                                    kinect_extr_param['%s_d2c' % cam][1])

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect1_3', cam)][0],
                    extr_param['%s-%s' % ('azure_kinect1_3', cam)][1])
        transformation_to_cam1a = create_transformation_matrix( r, t)

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][0],
                    extr_param['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][1])
        transformation_to_cam1b = create_transformation_matrix( r, t)

        # Apply the depth-to-color transformation
        xyz_points_d2c = transform_points(xyz_points, d2c_transformation)        # Transform the coordinates to camera 1's frame
        xyz_points_cam1 = transform_points(xyz_points_d2c, transformation_to_cam1a)
        xyz_points_cam1 = transform_points(xyz_points_cam1, transformation_to_cam1b)

        all_xyz_points.append(xyz_points_cam1)

    if cam == 'azure_kinect2_5':
        # Transform coordinates to camera 1
        d2c_transformation = create_transformation_matrix(kinect_extr_param['%s_d2c' % cam][0],
                                                    kinect_extr_param['%s_d2c' % cam][1])

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect2_4', 'azure_kinect2_5')][0],
                    extr_param['%s-%s' % ('azure_kinect2_4', 'azure_kinect2_5')][1])
        transformation_to_cam1a = create_transformation_matrix( r, t)

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect1_3', 'azure_kinect2_4')][0],
                    extr_param['%s-%s' % ('azure_kinect1_3', 'azure_kinect2_4')][1])
        transformation_to_cam1b = create_transformation_matrix( r, t)

        r,t = inverse_extrinsic_params(extr_param['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][0],
                    extr_param['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][1])
        transformation_to_cam1c = create_transformation_matrix( r, t)

        # Apply the depth-to-color transformation
        xyz_points_d2c = transform_points(xyz_points, d2c_transformation)        # Transform the coordinates to camera 1's frame
        xyz_points_cam1 = transform_points(xyz_points_d2c, transformation_to_cam1a)
        xyz_points_cam1 = transform_points(xyz_points_cam1, transformation_to_cam1b)
        xyz_points_cam1 = transform_points(xyz_points_cam1, transformation_to_cam1c)

        all_xyz_points.append(xyz_points_cam1)

    if cam == 'azure_kinect3_3':
        # Transform coordinates to camera 1
        d2c_transformation = create_transformation_matrix(kinect_extr_param['%s_d2c' % cam][0],
                                                    kinect_extr_param['%s_d2c' % cam][1])

        r,t = extr_param['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][0], extr_param['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][1]
        transformation_to_cam1a = create_transformation_matrix( r, t)

        r,t = extr_param['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][0], extr_param['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][1]
        transformation_to_cam1b = create_transformation_matrix( r, t)

        # Apply the depth-to-color transformation
        xyz_points_d2c = transform_points(xyz_points, d2c_transformation)        # Transform the coordinates to camera 1's frame
        xyz_points_cam1 = transform_points(xyz_points_d2c, transformation_to_cam1a)
        xyz_points_cam1 = transform_points(xyz_points_cam1, transformation_to_cam1b)

        all_xyz_points.append(xyz_points_cam1)