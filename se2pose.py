import numpy as np
# import cv2
# import pyrr
from scipy.spatial.transform import Rotation as R


def se3_to_transformation(se3_data):
    # se3_data 是形状为 [N, 6] 的张量
    batch_size = se3_data.shape[0]
    transformations = np.zeros((batch_size, 4, 4))
    for i in range(batch_size):
        rotvec = se3_data[i, 3:6]
        trans = se3_data[i, 0:3]
        rotmat = so2SO(rotvec)
        transformations[i, :3, :3] = rotmat
        transformations[i, :3, 3] = trans
        transformations[i, 3, 3] = 1
    return transformations


def line2mat(line_data):
    mat = np.eye(4)
    mat[0:3, :] = line_data.reshape(3, 4)
    return np.matrix(mat)


def motion2pose(data,initial_se):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size + 1, 12))
    temp = np.eye(4, 4).reshape(1, 16)
    all_pose[0, :] = line2mat(initial_se).reshape(1, 16)[0, 0:12]
    pose = np.matrix(np.eye(4, 4))
    for i in range(0, data_size):
        data_mat = line2mat(data[i, :])
        pose = pose * data_mat
        pose_line = np.array(pose[0:3, :]).reshape(1, 12)
        all_pose[i + 1, :] = pose_line
    return all_pose


# 定义函数pose2motion，用于将输入的pose数据转换为motion数据
# 参数data为输入的pose数据，skip为跳过的帧数
# 返回值为转换后的motion数据
def pose2motion(data, skip=0):
    # 获取输入pose数据的帧数
    data_size = data.shape[0]
    # 初始化转换后的motion数据
    all_motion = np.zeros((data_size - 1, 12))
    # 遍历输入pose数据
    for i in range(0, data_size - 1 - skip):
        # 将每一帧pose数据转换为矩阵
        pose_curr = line2mat(data[i, :])
        pose_next = line2mat(data[i + 1 + skip, :])
        # 计算当前帧到下一帧的motion
        motion = pose_curr.I * pose_next
        # 将motion转换为行向量
        motion_line = np.array(motion[0:3, :]).reshape(1, 12)
        # 将转换后的motion行向量添加到all_motion中
        all_motion[i, :] = motion_line
    # 返回转换后的motion数据
    return all_motion


# 定义SE2se函数，将SE_data转换为6维数组
def SE2se(SE_data):
    # 初始化结果数组
    result = np.zeros((6))
    # 将SE_data的第一行赋值给结果数组的0-2维
    result[0:3] = np.array(SE_data[0:3, 3].T)
    # 将SE_data的第一行赋值给结果数组的3-5维
    result[3:6] = SO2so(SE_data[0:3, 0:3]).T
    # 返回结果数组
    return result


# 定义SO2so函数，将SO_data转换为3维数组
def SO2so(SO_data):
    # 使用R.from_matrix函数将SO_data转换为旋转矩阵
    return R.from_matrix(SO_data).as_rotvec()


def so2SO(so_data):
    return R.from_rotvec(so_data).as_matrix()


def se2SE(se_data):
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3, 0:3] = so2SO(se_data[3:6])
    result_mat[0:3, 3] = np.matrix(se_data[0:3]).T
    return result_mat


### can get wrong result
def se_mean(se_datas):
    all_SE = np.matrix(np.eye(4))
    for i in range(se_datas.shape[0]):
        se = se_datas[i, :]
        SE = se2SE(se)
        all_SE = all_SE * SE
    all_se = SE2se(all_SE)
    mean_se = all_se / se_datas.shape[0]
    return mean_se


def ses_mean(se_datas):
    se_datas = np.array(se_datas)
    se_datas = np.transpose(
        se_datas.reshape(se_datas.shape[0], se_datas.shape[1], se_datas.shape[2] * se_datas.shape[3]), (0, 2, 1))
    se_result = np.zeros((se_datas.shape[0], se_datas.shape[2]))
    for i in range(0, se_datas.shape[0]):
        mean_se = se_mean(se_datas[i, :, :])
        se_result[i, :] = mean_se
    return se_result


def ses2poses(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size + 1, 12))
    temp = np.eye(4, 4).reshape(1, 16)
    all_pose[0, :] = temp[0, 0:12]
    pose = np.matrix(np.eye(4, 4))
    for i in range(0, data_size):
        data_mat = se2SE(data[i, :])
        pose = pose * data_mat
        pose_line = np.array(pose[0:3, :]).reshape(1, 12)
        all_pose[i + 1, :] = pose_line
    return all_pose


# 定义函数ses2poses_quat，用于将se(2)的6维数据转换为7维的quat
def ses2poses_quat(data, initial_pos_quat):
    '''
    ses: N x 6
    '''
    # 获取data的行数
    data_size = data.shape[0]
    # 初始化一个7维的数组，用于存放转换后的quat
    all_pose_quat = np.zeros((data_size + 1, 7))
    # 将第一个quat设置为[0, 0, 0, 0, 0, 0, 1]
    all_pose_quat[0, :] = initial_pos_quat
    # 初始化一个4x4的矩阵
    pose = np.matrix(np.eye(4, 4))
    # 遍历data中的每一行
    for i in range(0, data_size):
        # 将data中的每一行转换为se2SE矩阵
        data_mat = se2SE(data[i, :])
        # 将pose与data_mat相乘
        pose = pose * data_mat
        # 将pose的3x3矩阵转换为quat
        quat = SO2quat(pose[0:3, 0:3])
        # 将pose的平移转换为3维数组
        all_pose_quat[i + 1, :3] = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])
        # 将quat转换为4维数组
        all_pose_quat[i + 1, 3:] = quat
        # 返回转换后的quat
    return all_pose_quat


# 将motion_data转换为ses
def SEs2ses(motion_data):
    # 获取motion_data的行数
    data_size = motion_data.shape[0]
    # 初始化ses
    ses = np.zeros((data_size, 6))
    # 遍历每一行
    for i in range(0, data_size):
        # 初始化SE
        SE = np.matrix(np.eye(4))
        # 将motion_data的每一行转换为3*4的矩阵
        SE[0:3, :] = motion_data[i, :].reshape(3, 4)
        # 将SE转换为se
        ses[i, :] = SE2se(SE)
    # 返回ses
    return ses


def so2quat(so_data):
    so_data = np.array(so_data)
    theta = np.sqrt(np.sum(so_data * so_data))
    axis = so_data / theta
    quat = np.zeros(4)
    quat[0:3] = np.sin(theta / 2) * axis
    quat[3] = np.cos(theta / 2)
    return quat


def quat2so(quat_data):
    quat_data = np.array(quat_data)
    sin_half_theta = np.sqrt(np.sum(quat_data[0:3] * quat_data[0:3]))
    axis = quat_data[0:3] / sin_half_theta
    cos_half_theta = quat_data[3]
    theta = 2 * np.arctan2(sin_half_theta, cos_half_theta)
    so = theta * axis
    return so


# input so_datas batch*channel*height*width
# return quat_datas batch*numner*channel
def sos2quats(so_datas, mean_std=[[1], [1]]):
    so_datas = np.array(so_datas)
    so_datas = so_datas.reshape(so_datas.shape[0], so_datas.shape[1], so_datas.shape[2] * so_datas.shape[3])
    so_datas = np.transpose(so_datas, (0, 2, 1))
    quat_datas = np.zeros((so_datas.shape[0], so_datas.shape[1], 4))
    for i_b in range(0, so_datas.shape[0]):
        for i_p in range(0, so_datas.shape[1]):
            so_data = so_datas[i_b, i_p, :]
            quat_data = so2quat(so_data)
            quat_datas[i_b, i_p, :] = quat_data
    return quat_datas


def SO2quat(SO_data):
    rr = R.from_matrix(SO_data)
    # rr = R.from_dcm(SO_data)
    return rr.as_quat()


def quat2SO(quat_data):
    return R.from_quat(quat_data).as_matrix()


# 定义函数pos_quat2SE，用于将四元数转换为SE矩阵
def pos_quat2SE(quat_data):
    # 从四元数中获取SO矩阵
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    # 初始化SE矩阵
    SE = np.matrix(np.eye(4))
    # 将SO矩阵赋值给SE矩阵
    SE[0:3, 0:3] = np.matrix(SO)
    # 将四元数的前3个元素赋值给SE矩阵的第3列
    SE[0:3, 3] = np.matrix(quat_data[0:3]).T
    # 将SE矩阵转换为数组，并将其转换为1*12的数组
    SE = np.array(SE[0:3, :]).reshape(1, 12)
    # 返回SE矩阵
    return SE


# 定义函数pos_quats2SEs，用于将四元数组转换为SE矩阵
def pos_quats2SEs(quat_datas):
    # 获取四元数组的长度
    data_len = quat_datas.shape[0]
    # 初始化SE矩阵
    SEs = np.zeros((data_len, 12))
    # 遍历四元数组中的每一个元素
    for i_data in range(0, data_len):
        # 将四元数组转换为SE矩阵
        SE = pos_quat2SE(quat_datas[i_data, :])
        # 将SE矩阵存入SE矩阵中
        SEs[i_data, :] = SE
    # 返回SE矩阵
    return SEs


def pos_quats2SE_matrices(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = []
    for quat in quat_datas:
        SO = R.from_quat(quat[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3, 0:3] = SO
        SE[0:3, 3] = quat[0:3]
        SEs.append(SE)
    return SEs


def SE2pos_quat(SE_data):
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3, 0:3])
    pos_quat[:3] = SE_data[0:3, 3].T
    return pos_quat


def kitti2tartan(traj):
    '''
    traj: in kitti style, N x 12 numpy array, in camera frame
    output: in TartanAir style, N x 7 numpy array, in NED frame
    '''
    T = np.array([[0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    T_inv = np.linalg.inv(T)
    new_traj = []

    for pose in traj:
        tt = np.eye(4)
        tt[:3, :] = pose.reshape(3, 4)
        ttt = T.dot(tt).dot(T_inv)
        new_traj.append(SE2pos_quat(ttt))

    return np.array(new_traj)


def tartan2kitti(traj):
    T = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    T_inv = np.linalg.inv(T)
    new_traj = []

    for pose in traj:
        tt = np.eye(4)
        tt[:3, :] = pos_quat2SE(pose).reshape(3, 4)
        ttt = T.dot(tt).dot(T_inv)
        new_traj.append(ttt[:3, :].reshape(12))

    return np.array(new_traj)


