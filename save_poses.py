import matplotlib.pyplot as plt
import numpy as np
import queue
import pickle
import os
from datasets.kitti import KITTI
from datasets.utils import euler_to_rotation
from scipy.spatial.transform import Rotation as R
from datasets.utils import rotation_to_euler
def save_trajectory(poses, sequence, save_dir):
    """
    Save predicted poses in .txt file
    Args:
        poses {ndarray}: list with all 4x4 pose matrix
        sequence {str}: sequence of KITTI dataset
        save_dir {str}: path to save pose
    """
    # create directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_filename = os.path.join(save_dir, "{}.txt".format(sequence))
    with open(output_filename, "w") as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            line = " ".join([str(x) for x in pose]) + "\n"
            f.write(line)


def post_processing(pred_poses):
    pred_poses = pred_poses
    return np.asarray(pred_poses)

def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3,:]).reshape(1,12)
    return SE

def recover_trajectory_and_poses(poses):
    predicted_poses = []
    # recover predicted trajectory
    predicted_trajectory = []
    for i in range(len(poses)):
        # if i == 0:
        #     # T = np.array([[1.000000e+00, 1.197625e-11, 1.704638e-10, 5.551115e-17],
        #                 #  [1.197625e-11, 1.000000e+00, 3.562503e-10, 0.000000e+00],  
        #                 #  [1.704638e-10, 3.562503e-10, 1.000000e+00, 2.220446e-16],
        #                 #  [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])
        #     T = np.eye(4)

        # angles = poses[i, :3]
        # t = poses[i, 3:]

        # undo normalization
        # mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        # std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        # mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        # std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        # [x, y, z] = np.multiply(angles, std_angles)  + mean_angles
        # t = np.multiply(t, std_t) + mean_t
        # R = np.asarray(euler_to_rotation(x, y, z, seq = 'xyz'))

        # T_r = np.concatenate((np.concatenate([R, np.reshape(t, (3,1))], axis=1) , [[0.0, 0.0, 0.0, 1.0]] ), axis=0)
        # T_abs = np.dot(T,T_r)

        # # 更新累积位姿矩阵
        # T = T_abs
        pose = poses[i]
        T = pos_quat2SE(pose)
        T = np.reshape(T, (3,4))
        # R = T[:3, :3]
        # t = T[:3, 3]
        # angles = rotation_to_euler(R, seq='zyx')

        # normalization
        # [x, y, z] = np.multiply(angles, std_angles)  + mean_angles
        # t = np.multiply(t, std_t) + mean_t
        # R = np.asarray(euler_to_rotation(x, y, z, seq = 'xyz'))
        T_r = np.concatenate((T , [[0.0, 0.0, 0.0, 1.0]] ), axis=0)
        predicted_poses.append(T_r)
        # predicted_trajectory.append(T_abs[:3, 3])
    return predicted_poses
  

if __name__ == "__main__":
  
    ckpt_path = "path/to/checkpoint"
    ckpt_name = "brightvo"
    sequences = ["00"]

    # read hyperparameters and configuration

    ckpt_path = os.path.join(ckpt_path, ckpt_name)
    checkpoint_path = ckpt_path

    # plot trajectory and ground truth
    for sequence in sequences:
        # read ground test data and predicted poses
        pred_path = os.path.join(checkpoint_path, "pred_poses_{}.npy".format(sequence))
        pred_poses = np.load(pred_path)

        # post processing and recover trajectory
        poses = post_processing(pred_poses)
        pred_poses = recover_trajectory_and_poses(poses)
        
        save_trajectory(pred_poses, sequence, 
                        save_dir=os.path.join(checkpoint_path, "pred_poses"))

  