# import numpy as np
# from se2pose import ses2poses_quat, pos_quats2SEs, motion2pose
# from scipy.spatial.transform import Rotation as R
# def load_relative_poses(file_path):
#     # 从文件中读取每一行的相对位姿矩阵
#     relative_poses = np.load(file_path)
#     # with open(file_path, 'r') as file:
#     #     for line in file:
#     #         # 读取每一行，并将其转换为一个3x4的矩阵
#     #         p = list(map(float, line.strip().split()))
#     #         t = np.array([p[3], p[7], p[11]]).reshape(3, 1)  # 提取平移向量
#     #         R = np.array([[p[0], p[1], p[2]],                # 提取旋转矩阵
#     #                       [p[4], p[5], p[6]],
#     #                       [p[8], p[9], p[10]]])
#     #         relative_poses.append((R, t))
#     return relative_poses

# def compute_absolute_poses(relative_poses, initial_R, initial_t):
#     # 存储每一帧的绝对位姿
#     absolute_poses = [(initial_R, initial_t)]
    
#     # 遍历所有相对位姿
#     for rel_R, rel_t in relative_poses:
#         # 获取前一帧的绝对位姿
#         prev_R, prev_t = absolute_poses[-1]
        
#         # 计算当前帧的绝对旋转矩阵和平移向量
#         current_R = prev_R @ rel_R
#         current_t =(prev_R @ rel_t + prev_t)/10
        
#         # 保存结果
#         absolute_poses.append((current_R, current_t))
    
#     return absolute_poses

# def save_absolute_poses(file_path, absolute_poses):
#     with open(file_path, 'w') as f:
#         for pose in absolute_poses:
#             pose = pose.flatten()[:12]
#             line = " ".join([str(x) for x in pose]) + "\n"
#             f.write(line)


# def convert_pose_12_to_7(pose_12):
#     # 确保输入是一个包含12个元素的列表或数组
#     if len(pose_12) != 12:
#         raise ValueError("Input pose must have 12 elements.")
    
#     # 提取平移向量（位置）
#     position = np.array([pose_12[3], pose_12[7], pose_12[11]])

    
#     # 提取旋转矩阵（3x3）
#     rotation_matrix = np.array([[pose_12[0], pose_12[1], pose_12[2]],               
#                                 [pose_12[4], pose_12[5], pose_12[6]],
#                                 [pose_12[8], pose_12[9], pose_12[10]]])
    
#     # 将旋转矩阵转换为四元数
#     rotation = R.from_matrix(rotation_matrix)
#     quaternion = rotation.as_quat()  # 返回 (x, y, z, w) 格式的四元数
    
#     # 将位置和平移组合成最终的7个数的姿态
#     pose_7 = np.concatenate((position, quaternion))
    
#     return pose_7

# def main():
#     # 文件路径
#     relative_poses_file = '/home/wangdongzhihan/codes/BrightVO/checkpoints/Exp7/checkpoint_best/pred_poses_01.npy'
#     absolute_poses_file = '/home/wangdongzhihan/codes/BrightVO/checkpoints/Exp7/checkpoint_best/pred_poses/absolute_pose.txt'

#     # 读取第一帧的绝对位姿（3x4矩阵的旋转矩阵和平移向量）
#     # initial_R = np.array([
#     #     [0.03512450951214958, 0.9993800836270742, -0.0023910836607952905],
#     #     [-0.9992533666700714, 0.035158381590332764, 0.016018657948669975],
#     #     [0.01609279435209595, 0.0018266508945495637, 0.9998688340559728]
#     # ])
#     # initial_t = np.array([[0.0], [0.0], [0.0]])
#     initial_se = np.array([0.0004538001955091553, 0.9999998970257035, 3.7369722219640572e-06, 0.0, -0.9999936217856628, 0.00045381058662922574, -0.0035426605742461515, 0.0, -0.003542661905320727, -2.129288325538773e-06, 0.999993724751356, 0.0
# ])
    
#     initial_pos_quat = convert_pose_12_to_7(initial_se)
#     # 读取相对位姿
#     relative_poses = load_relative_poses(relative_poses_file)
#     relative_poses = relative_poses.squeeze(1)
#     # 计算绝对位姿
#     absolute_poses =  ses2poses_quat(relative_poses, initial_pos_quat)
#     abs_ses = pos_quats2SEs(absolute_poses)
#     all_poses = motion2pose(abs_ses, initial_se)
#     # 保存绝对位姿
#     save_absolute_poses(absolute_poses_file, all_poses)
    
# if __name__ == '__main__':
#     main()

import numpy as np

# 初始位姿的旋转矩阵和平移向量
R1 = np.array([
    [0.0004538001955091553, 0.9999998970257035, 3.7369722219640572e-06],
    [-0.9999936217856628, 0.00045381058662922574, -0.0035426605742461515],
    [-0.003542661905320727, -2.129288325538773e-06, 0.999993724751356]
])
t1 = np.array([0.0, 0.0, 0.0])

# 相对位姿的旋转矩阵和平移向量
R12 = np.array([
    [0.9999999633178923, 9.593327391557042e-05, -0.00025330025836903356],
    [-9.588287949234979e-05, 0.9999999756114958, 0.00019895597841933094],
    [0.000253319338689794, -0.00019893168396305778, 0.9999999481277476]
])
t12 = np.array([-0.16168736921685792, 0.00681505293560028, -0.03315748657226569])

# 计算第二帧的旋转矩阵和平移向量
R2 = np.dot(R1, R12)
t2 = np.dot(R1, t12) + t1

# 打印结果
print("第二帧的旋转矩阵 R2:")
print(R2)

print("\n第二帧的平移向量 t2:")
print(t2)
