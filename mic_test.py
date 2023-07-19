import numpy as np

def avg_joints_distance(keypoints_3d_list, keypoints_3d_list_prev):
    assert keypoints_3d_list.shape == keypoints_3d_list_prev.shape, "Input arrays must have the same shape"
    
    distances = np.linalg.norm(keypoints_3d_list - keypoints_3d_list_prev, axis=1)
    avg_distance = np.mean(distances)
    return avg_distance

a=avg_joints_distance(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[100, 100, 100], [4, 5, 6]]))
print(a)