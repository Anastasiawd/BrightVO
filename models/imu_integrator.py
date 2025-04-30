import torch
import pykitti
import numpy as np
import pypose as pp
from torch import nn
import tqdm, argparse
from datetime import datetime
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

class KITTI_IMU(Data.Dataset):
    def __init__(self, root, dataname, drive, duration=10, step_size=1, mode='train'):
        super().__init__()
        self.duration = duration
        self.data = pykitti.raw(root, dataname, drive)
        self.seq_len = len(self.data.timestamps) - 1
        assert mode in ['evaluate', 'train',
                        'test'], "{} mode is not supported.".format(mode)

        self.dt = torch.tensor([datetime.timestamp(self.data.timestamps[i+1]) -
                               datetime.timestamp(self.data.timestamps[i])
                               for i in range(self.seq_len)])
        self.gyro = torch.tensor([[self.data.oxts[i].packet.wx,
                                   self.data.oxts[i].packet.wy,
                                   self.data.oxts[i].packet.wz]
                                   for i in range(self.seq_len)])
        self.acc = torch.tensor([[self.data.oxts[i].packet.ax,
                                  self.data.oxts[i].packet.ay,
                                  self.data.oxts[i].packet.az]
                                  for i in range(self.seq_len)])
        self.gt_rot = pp.euler2SO3(torch.tensor([[self.data.oxts[i].packet.roll,
                                                  self.data.oxts[i].packet.pitch,
                                                  self.data.oxts[i].packet.yaw]
                                                  for i in range(self.seq_len)]))
        self.gt_vel = self.gt_rot @ torch.tensor([[self.data.oxts[i].packet.vf,
                                                   self.data.oxts[i].packet.vl,
                                                   self.data.oxts[i].packet.vu]
                                                   for i in range(self.seq_len)])
        self.gt_pos = torch.tensor(
            np.array([self.data.oxts[i].T_w_imu[0:3, 3] for i in range(self.seq_len)]))

        start_frame = 0
        end_frame = self.seq_len
        if mode == 'train':
            end_frame = np.floor(self.seq_len * 0.5).astype(int)
        elif mode == 'test':
            start_frame = np.floor(self.seq_len * 0.5).astype(int)

        self.index_map = [i for i in range(
            0, end_frame - start_frame - self.duration, step_size)]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id = self.index_map[i]
        end_frame_id = frame_id + self.duration
        return {
            'dt': self.dt[frame_id: end_frame_id],
            'acc': self.acc[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gt_pos': self.gt_pos[frame_id+1: end_frame_id+1],
            'gt_rot': self.gt_rot[frame_id+1: end_frame_id+1],
            'gt_vel': self.gt_vel[frame_id+1: end_frame_id+1],
            'init_pos': self.gt_pos[frame_id][None, ...],
            # TODO: the init rotation might be used in gravity compensation
            'init_rot': self.gt_rot[frame_id: end_frame_id],
            'init_vel': self.gt_vel[frame_id][None, ...],
        }

    def get_init_value(self):
        return {'pos': self.gt_pos[:1],
                'rot': self.gt_rot[:1],
                'vel': self.gt_vel[:1]}
    
class IMUCorrector(nn.Module):
    def __init__(self, size_list= [6, 64, 128, 128, 128, 6]):
        super().__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i], size_list[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)
        self.imu = pp.module.IMUPreintegrator(reset=True, prop_cov=False)

    def forward(self, data, init_state):
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        B, F = feature.shape[:2]

        output = self.net(feature.reshape(B*F,6)).reshape(B, F, 6)
        corrected_acc = output[...,:3] + data["acc"]
        corrected_gyro = output[...,3:] + data["gyro"]

        return self.imu(init_state = init_state,
                        dt = data['dt'],
                        gyro = corrected_gyro,
                        acc = corrected_acc,
                        rot = data['gt_rot'].contiguous())
integrator = pp.module.IMUPreintegrator()