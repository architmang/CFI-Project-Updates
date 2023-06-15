import sys
sys.path.append('../')
from calibration.utils import *
import pickle
import os
import json
import matplotlib.pyplot as plt
import joblib


class KinectOpenPoseSythesis:
    def __init__(self, root_dir, visualize=False,
                    cam_list = ('azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2')):
        
        self.root_dir = root_dir
        self.action = None
        self.data_dir = None  # '%s/%s' % (self.root_dir, self.action)
        self.openpose_pred_dir = None  # '%s/%s/' % (root_dir, action)
        self.openpose_save_dir = None  # '%s/%s/openpose' % (root_dir, action)
        self.calib_dir = None  # '%s/calib_1024' % (root_dir)
        self.bg_frame_idx = None
        self.cam_list = cam_list
        self.visualize = visualize
        self.joint_idx_openpose = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.kinematic_tree_openpose = [(0, 1, 'salmon'), (1, 2, 'darkorange'), (2, 3, 'orange'),
                                        (3, 4, 'orange'), (1, 5, 'limegreen'), (5, 6, 'lime'), (6, 7, 'lime'),
                                        (1, 8, 'red'), (8, 9, 'cyan'), (9, 10, 'skyblue'), (10, 11, 'skyblue'),
                                        (8, 12, 'deeppink'), (12, 13, 'pink'), (13, 14, 'pink')]
        self.intr = None
        self.kinect_extr = None
        self.extr = None
        self.trans_dict = None
        self.bg_imgs_dict = None

    def set_action(self, action, calib_dir, bg_frame_idx, cam_list=None):
        self.action = action
        self.data_dir = '%s/%s' % (self.root_dir, self.action)
        self.calib_dir = '%s/calib_%s' % (self.root_dir, calib_dir)

        if 'group4' in action:
            # action: 'subject01_group4_time1'
            self.bg_dir = '%s/%s1' % (self.root_dir, action[:-1])
        else:
            self.bg_dir = '%s/calib_%s' % (self.root_dir, calib_dir)
        self.bg_frame_idx = bg_frame_idx

        self.openpose_pred_dir = '%s/%s' % (self.root_dir, action)
        self.openpose_save_dir = '%s/%s/openpose' % (self.root_dir, action)
        if not os.path.exists(self.openpose_save_dir):
            print('make direction: %s' % self.openpose_save_dir)
            os.mkdir(self.openpose_save_dir)
        if cam_list is not None:
            self.cam_list = cam_list

        self.intr = pickle.load(open('%s/intrinsic_param.pkl' % self.calib_dir, 'rb'))
        self.kinect_extr = pickle.load(open('%s/kinect_extrinsic_param.pkl' % self.calib_dir, 'rb'))
        self.extr = pickle.load(open('%s/extrinsic_param_%s.pkl' % (self.calib_dir, calib_dir), 'rb'))
        self.trans_dict = self.load_transform()
        self.bg_imgs_dict = self.load_background()

    def load_transform(self):
        trans_dict = {}
        # depth to color within each kinect
        for cam in self.cam_list:
            trans_dict['%s_cd' % cam] = \
                Transform(r=self.kinect_extr['%s_d2c' % cam][0], t=self.kinect_extr['%s_d2c' % cam][1])

        # extrinsic azure_kinect_0 color to other color
        key = 'azure_kinect_0-azure_kinect_0'
        trans_dict[key] = Transform(r=np.eye(3), t=np.zeros([3]))
        key = 'azure_kinect_0-azure_kinect_1'
        trans_dict[key] = Transform(r=self.extr[key][0], t=self.extr[key][1])
        key = 'azure_kinect_0-azure_kinect_2'
        trans_dict[key] = Transform(r=self.extr[key][0], t=self.extr[key][1])

        key = 'azure_kinect_0-kinect_v2_1'
        T_tmp = Transform(r=self.extr['azure_kinect_1-kinect_v2_1'][0], t=self.extr['azure_kinect_1-kinect_v2_1'][1])
        trans_dict[key] = trans_dict['azure_kinect_0-azure_kinect_1'] * T_tmp
        key = 'azure_kinect_0-kinect_v2_2'
        T_tmp = Transform(r=self.extr['azure_kinect_2-kinect_v2_2'][0], t=self.extr['azure_kinect_2-kinect_v2_2'][1])
        trans_dict[key] = trans_dict['azure_kinect_0-azure_kinect_2'] * T_tmp
        return trans_dict

    def synthesize_openpose_3d_single(self, frame_idx, reigion_size=2, plot_depth=False):
        if self.openpose_pred_dir is None:
            assert ValueError('Set action fold.')
        openpose_3d_dict = {}
        for cam in self.cam_list:
            _frame_idx = frame_idx
            if not os.path.exists('%s/%s/depth/depth%04i.png' % (self.data_dir, cam, _frame_idx)):
                print('[warning] not exist: %s/%s/depth/depth%04i.png' % (self.data_dir, cam, _frame_idx))
                continue

            ### depth2color registration
            # read images
            depth_img = cv2.imread('%s/%s/depth/depth%04i.png' % (self.data_dir, cam, _frame_idx), -1)
            img_mask = substract_foreground(self.bg_imgs_dict[cam], depth_img)
            depth_img = depth_img * img_mask
            if plot_depth:
                plt.figure()
                plt.imshow(depth_img, cmap='gray')
                plt.show()

            color_img = cv2.cvtColor(
                cv2.imread('%s/%s/color/color%04i.jpg' % (self.data_dir, cam, _frame_idx)), cv2.COLOR_BGR2RGB)

            # unproject depth image to 3d
            v_d, u_d = depth_img.shape
            uvd_d = depth2uvd(depth_img).reshape([v_d * u_d, 3])
            xyz_d = unprojection(uvd_d, self.intr['%s_depth' % cam], simple_mode=False)

            # transform to color and project
            T_cd = self.trans_dict['%s_cd' % cam]
            xyz_c = T_cd.transform(xyz_d)
            uvd_c = projection(xyz_c, self.intr['%s_color' % cam], simple_mode=False)

            # remove invalid point
            v_c, u_c, _ = color_img.shape
            valid_idx = (uvd_c[:, 0] >= 0) & (uvd_c[:, 0] < u_c) & \
                        (uvd_c[:, 1] >= 0) & (uvd_c[:, 1] < v_c) & \
                        (uvd_c[:, 1] > 0)
            uvd_c = uvd_c[valid_idx, :]
            rgb_d = np.zeros([color_img.shape[0], color_img.shape[1]], dtype=np.float32)
            rgb_d[uvd_c[:, 1].astype(np.int32), uvd_c[:, 0].astype(np.int32)] = uvd_c[:, 2]

            # process openpose keypoints
            openpose_file = '%s/%s_openpose/color%04i_keypoints.json' % (self.openpose_pred_dir, cam, _frame_idx)
            openpose_2d = self.encode_openpose_json(openpose_file, self.joint_idx_openpose) # [u, v, confidence]

            # get openpose_2d depth
            openpose_3d = self.get_openpose2d_depth(rgb_d, openpose_2d, region_size=reigion_size)
            openpose_3d_dict['%s' % cam] = openpose_3d  # [u, v, d, confidence] color camera coordinate


        # synthesize multi-view 3d joint
        flag_zero_depth, tmp = [], []  # joint depth is zero
        for cam in self.cam_list:
            openpose_3d = openpose_3d_dict[cam].copy()
            # unproject to 3d
            xyz_c2 = unprojection(openpose_3d[:, 0:3], self.intr['%s_color' % cam], simple_mode=False)
            flag_zero_depth.append(xyz_c2[:, 2] > 0)
            # print(xyz_c2)
            # transform from other to azure_kinect_0 color camera coordinate
            trans = self.trans_dict['%s_cd' % cam].inv() * \
                    self.trans_dict['azure_kinect_0-%s' % cam] *  \
                    self.trans_dict['azure_kinect_0_cd']
            xyz_c1 = trans.transform(xyz_c2)
            openpose_3d[:, 0:3] = xyz_c1
            tmp.append(openpose_3d)
        flag_zero_depth = np.expand_dims(np.stack(flag_zero_depth, axis=1), axis=-1)  # [15 joints, 5 cameras, 1]
        tmp = np.stack(tmp, axis=1)  # [15 joints, 5 cameras, 4]
        # print(flag_zero_depth[:, :, 0].astype(np.int16))

        flag1 = (tmp[:, :, 3:4] > 0.8) & flag_zero_depth
        flag2 = (tmp[:, :, 3:4] > 0.9) & flag_zero_depth
        flag_tmp = np.zeros_like(tmp[:, :, 3:4])
        flag_tmp[:, 0:3, :] = 1
        flag3 = (flag_tmp > 0) & flag_zero_depth
        # print(flag1[:, :, 0].astype(np.int16))
        # print(flag2[:, :, 0].astype(np.int16))

        # [15 joints, 5 cameras, 1]
        weight = flag1.astype(np.float32) + flag2.astype(np.float32) + flag3.astype(np.float32)
        weight_sum = np.sum(weight, axis=1)  # [15 joints, 1]
        # print(weight[:, :, 0])
        pose_3d = np.sum(tmp[:, :, 0:3] * weight, axis=1) / (weight_sum + (weight_sum==0).astype(np.float32))
        joint_flag = np.sum(weight_sum, axis=1) > 0

        # pose_3d = []
        # for i in range(tmp.shape[0]):
        #     flag = (tmp[:, :, 3:4] > 0.8) & (tmp[:, :, 2:3] > 0)
        #     if np.sum(flag) == 0:
        #         joint = np.array([0, 0, 0])
        #     else:
        #         joint = np.median(tmp[i, flag, 0:3], axis=0)
        #     pose_3d.append(joint)
        # pose_3d = np.stack(pose_3d, axis=0)

        # print(pose_3d)
        if self.visualize:
            # self.visualize_filtered_openpose(frame_idx, openpose_3d_dict, flag)
            self.visualize_3d_pose(frame_idx, pose_3d, joint_flag)
            pass

        return pose_3d, joint_flag

    def synthesize_openpose_3d(self, num_file=None):
        if num_file is None:
            import glob
            tmp = sorted(glob.glob('%s/azure_kinect_0_openpose/color*_keypoints.json' % self.openpose_pred_dir))
            tmp_img = sorted(glob.glob('%s/azure_kinect_0/color/color*.jpg' % self.data_dir))
            if len(tmp) == 0:
                print('[error] %s missing openpose' % self.action)
                num_file = -1
            else:
                num_file = int(tmp[-1].split('color')[-1].split('_keypoints')[0])
                num_img_file = int(tmp_img[-1].split('color')[-1].split('.')[0])
                if num_img_file != num_file:
                    print('[error] %s (openpose %i, img %i)' % (self.action, num_file, num_img_file))
                    num_file = -1

        # all_poses = {}
        trans_dc = self.trans_dict['azure_kinect_0_cd'].inv()
        for frame_idx in range(num_file+1):
            save_fname = '%s/openpose%04i.pkl' % (self.openpose_save_dir, frame_idx)
            if os.path.exists(save_fname):
                continue
            # if frame_idx % 300 == 0:
            #     print('%s, %04i' % (self.action, frame_idx))
            pose_3d, joint_flag = self.synthesize_openpose_3d_single(frame_idx)
            # all_poses['%04i' % frame_idx] = trans_dc.transform(pose_3d)
            save_pose_3d = trans_dc.transform(pose_3d)
            joblib.dump(save_pose_3d, save_fname, compress=3)

        # joblib.dump(all_poses, '%s/openpose.pkl' % self.openpose_save_dir, compress=3)

    def visualize_filtered_openpose(self, frame_idx, openpose_3d_dict, flag):
        for i, cam in enumerate(self.cam_list):
            if i != 3:
                continue
            # read color image
            color_img = cv2.cvtColor(
                cv2.imread('%s/%s/color/color%04i.jpg' % (self.data_dir, cam, frame_idx)), cv2.COLOR_BGR2RGB)
            openpose_3d = openpose_3d_dict[cam]

            plt.figure()
            plt.imshow(color_img)
            for j in self.joint_idx_openpose:
                if not flag[j, i, 0]:
                    print(flag[j, i, 0])
                    continue
                plt.scatter(openpose_3d[j, 0], openpose_3d[j, 1], color='r', marker='h', s=15)
            plt.show()

    def visualize_3d_pose(self, frame_idx, pose_3d, joint_flag):
        plt.figure(figsize=(24, 16))
        for idx, cam in enumerate(self.cam_list):
            if 'azure_kinect' in cam:
                # _frame_idx = frame_idx + 5
                _frame_idx = frame_idx
            else:
                _frame_idx = frame_idx

            # read color image
            color_img = cv2.cvtColor(
                cv2.imread('%s/%s/color/color%04i.jpg' % (self.data_dir, cam, _frame_idx)), cv2.COLOR_BGR2RGB)

            # transform and project
            trans = self.trans_dict['%s_cd' % cam].inv() * \
                    self.trans_dict['azure_kinect_0-%s' % cam] *  \
                    self.trans_dict['azure_kinect_0_cd']
            trans = trans.inv()  # azure_kinect_0 color to other color
            xyz_c = trans.transform(pose_3d)
            uvd_c = projection(xyz_c, self.intr['%s_color' % cam], simple_mode=False)

            plt.subplot(2, 3, idx+1)
            plt.imshow(color_img)
            for j1, j2, c in self.kinematic_tree_openpose:
                if (joint_flag[j1] == False) or (joint_flag[j2] == False):
                    print('[warning] joints (%i, %i) missing' % (j1, j2))
                    continue
                plt.scatter(uvd_c[j2, 0], uvd_c[j2, 1], color=c, marker='h', s=10)
                plt.plot([uvd_c[j1, 0], uvd_c[j2, 0]], [uvd_c[j1, 1], uvd_c[j2, 1]], color=c, linewidth=1.5)
        plt.show()

    def load_background(self):
        if self.openpose_pred_dir is None:
            assert ValueError('Set action fold.')
        bg_imgs = {}
        for cam in self.cam_list:
            if 'group4' in self.bg_dir:
                bg_file = '%s/%s/depth/depth%04i.png' % (self.bg_dir, cam, self.bg_frame_idx)
            else:
                bg_file = '%s/%s_calib_snap/depth%04i.png' % (self.bg_dir, cam, self.bg_frame_idx)
            bg_img = cv2.imread(bg_file, -1)
            bg_imgs[cam] = bg_img
        return bg_imgs

    @staticmethod
    def encode_openpose_json(openpose_file, joint_idx_openpose):
        # single person
        if not os.path.exists(openpose_file):
            # print('[warning] %s does not exist' % openpose_file)
            openpose_2d = np.zeros([len(joint_idx_openpose), 3])
        else:
            openpose_2d = json.load(open(openpose_file, 'r'))['people']
            if len(openpose_2d) == 0:
                # print('[error] %s no key-points detected.' % openpose_file)
                openpose_2d = np.zeros([len(joint_idx_openpose), 3]) # [u, v, confidence]
            elif len(openpose_2d) > 1:
                print('[warning] %s has more than one person.' % openpose_file)
                openpose_2d1 = np.reshape(np.asarray(openpose_2d[0]['pose_keypoints_2d']),
                                          [-1, 3])[joint_idx_openpose, :]
                openpose_2d2 = np.reshape(np.asarray(openpose_2d[1]['pose_keypoints_2d']),
                                          [-1, 3])[joint_idx_openpose, :]
                tmp = openpose_2d1[:, 2] < openpose_2d2[:, 2] # openpose_2d2 has higher confidence
                openpose_2d1[tmp, :] = openpose_2d2[tmp, :]
                openpose_2d = openpose_2d1.copy()  # [u, v, confidence]
            else:
                # [u, v, confidence] openpose_2d[i]['pose_keypoints_2d']
                openpose_2d = np.reshape(np.asarray(openpose_2d[0]['pose_keypoints_2d']), [-1, 3])[joint_idx_openpose, :]
        return openpose_2d

    @staticmethod
    def get_openpose2d_depth(rgb_d, openpose_2d, region_size):
        d = np.zeros([openpose_2d.shape[0], 1], dtype=np.float32)
        v_c, u_c = rgb_d.shape
        for i, joint in enumerate(openpose_2d):
            v_min = max(0, joint[1]-region_size)
            v_max = min(v_c, joint[1]+region_size)
            u_min = max(0, joint[0]-region_size)
            u_max = min(u_c, joint[0]+region_size)
            region = np.reshape(rgb_d[int(v_min):int(v_max), int(u_min):int(u_max)], [-1])
            region = region[region>0]
            if region.shape[0] == 0:
                d[i, 0] = 0
            else:
                d[i, 0] = np.mean(region)
        openpose_3d = np.concatenate([openpose_2d[:, 0:2], d, openpose_2d[:, 2:3]], axis=1)
        return openpose_3d


def openpose_3d_single_processor(synthesizer, action_names, cpu_id, subject_calib_dict):
    for idx, action in enumerate(action_names):
        calib_dir = subject_calib_dict[action.split('_')[0]]
        if 'group4' in action:
            bg_frame_idx = 10
        else:
            bg_frame_idx = 122
        print('[cpu %i] begin %s (%02i / %02i), calib_dir: %s, bg_frame_idx: %i' %
              (cpu_id, action, idx + 1, len(action_names), calib_dir, bg_frame_idx))
        synthesizer.set_action(action, calib_dir, bg_frame_idx)
        synthesizer.synthesize_openpose_3d()


def openpose_3d_multi_processor(root_dir, cam_list, subject_calib_dict, num_cpus=6):
    import multiprocessing
    action_names = []
    # filter out group4_time1
    for action in sorted(os.listdir(root_dir)):
        subject = action.split('_')[0]
        # if 'group4_time1' not in action and subject in subject_calib_dict.keys():
        if 'group4_time1' not in action and 'subject15_group3' in action:
            action_names.append(action)
    print(action_names)

    N = len(action_names)
    n_files_cpu = N // num_cpus
    synthesizer = KinectOpenPoseSythesis(root_dir, visualize=False, cam_list=cam_list)

    openpose_3d_single_processor(synthesizer, action_names, 0, subject_calib_dict)

    # results = []
    # pool = multiprocessing.Pool(num_cpus)
    # for i in range(num_cpus):
    #     idx1 = i * n_files_cpu
    #     idx2 = min((i + 1) * n_files_cpu, N)
    #     results.append(pool.apply_async(openpose_3d_single_processor,
    #                                     (synthesizer, action_names[idx1: idx2], i, subject_calib_dict)))
    # pool.close()
    # pool.join()
    # pool.terminate()
    #
    # for result in results:
    #     tmp = result.get()
    #     if tmp is not None:
    #         print(tmp)
    # print('Multi-cpu pre-processing ends.')


if __name__ == '__main__':
    # np.set_printoptions(suppress=True)
    # subject_calib_dict = {'subject01': '1024',
    #                       'subject02': '1024',
    #                       'subject03': '1024',
    #                       'subject04': '1028',
    #                       'subject05': '1028',
    #                       'subject06': '1028',
    #                       'subject07': '1028',
    #                       'subject08': '1028',
    #                       'subject09': '1028',
    #                       'subject10': '1028',
    #                       'subject11': '1101',
    #                       'subject12': '1101',
    #                       'subject13': '1101',
    #                       'subject14': '1101',
    #                       'subject15': '1101'}
    #
    # root_dir = '/data/shihao/data_event'
    # action = 'subject09_group3_time1'
    # calib_dir = subject_calib_dict[action.split('_')[0]]
    # bg_frame_idx = 122
    #
    # visual = True
    # cam_list = ('azure_kinect_0', 'azure_kinect_1', 'azure_kinect_2', 'kinect_v2_1', 'kinect_v2_2')
    #
    # synthesizer = KinectOpenPoseSythesis(root_dir, visualize=visual, cam_list=cam_list)
    # synthesizer.set_action(action, calib_dir, bg_frame_idx)
    # _, _ = synthesizer.synthesize_openpose_3d_single(249)
    # # synthesizer.synthesize_openpose_3d()

    # synthesizer = KinectOpenPoseSythesis(root_dir, visualize=visual, bg_frame_idx=121, cam_list=cam_list)
    # synthesizer.set_action('3')
    # synthesizer.synthesize_openpose_3d()
    # # _, _ = synthesizer.synthesize_openpose_3d_single(300)

    # synthesizer = KinectOpenPoseSythesis(root_dir, visualize=visual, bg_frame_idx=121, cam_list=cam_list)
    # synthesizer.set_action('4')
    # synthesizer.synthesize_openpose_3d()
    # # _, _ = synthesizer.synthesize_openpose_3d_single(300)

    # root_dir = 'D:/UoA_Research/data_10'
    # openpose_pred_dir = '%s/openpose_pred/1' % root_dir
    # with open('%s/openpose.txt' % openpose_pred_dir, 'r') as f:
    #     for line in f.readlines():
    #         tmp = line.split(' ')
    #         frame_idx = int(tmp[0])
    #         pose_3d = np.asarray([float(j) for j in tmp[1:-1]]).reshape([-1, 3])
    #         print(pose_3d)
    #
    # a = joblib.load('%s/openpose.pkl' % openpose_pred_dir)

    os.environ["OMP_NUM_THREADS"] = "1"
    subject_calib_dict = {'subject01': '1024',
                          'subject02': '1024',
                          'subject03': '1024',
                          'subject04': '1028',
                          'subject05': '1028',
                          'subject06': '1028',
                          'subject07': '1028',
                          'subject08': '1028',
                          'subject09': '1028',
                          'subject10': '1028',
                          'subject11': '1101',
                          'subject12': '1101',
                          'subject13': '1101',
                          'subject14': '1101',
                          'subject15': '1101'}

    cam_list = ('azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2')
    root_dir = '/data/shihao/data_event'
    # data_dir = './AzureKinectRecord_30_05'
    openpose_3d_multi_processor(root_dir, cam_list, subject_calib_dict, num_cpus=5)



