import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from datasets.utils import rotation_to_euler
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import pykitti
import pypose as pp
def sync_data(ts_src, ts_tar):
    res = []
    j = 0
    for t in ts_tar:
        while j+1 < len(ts_src) and abs(ts_src[j+1]-t) <= abs(ts_src[j]-t):
            j += 1
        res.append(j)
    # for i in range(len(res)-1):
    #     if res[i+1] - res[i] <= 0:
    #         print('sync_data error', i, ts_tar[i:i+2], ts_src[max(0,res[i]-5):min(len(ts_src), res[i]+5)])
    return np.array(res)

class KITTI(torch.utils.data.Dataset):
    """
    Dataloader for KITTI Visual Odometry Dataset
        http://www.cvlibs.net/datasets/kitti/eval_odometry.php

    Arguments:
        data_path {str}: path to data sequences
        gt_path {str}: path to poses
    """

    def __init__(self,
                 data_path=r"/home/wangdongzhihan/datasets/KITTI",
                 gt_path=r"/home/wangdongzhihan/datasets/KITTI/poses",
                 pose_path=r"",
                 camera_id="2",
                 sequences=["01", "02", "04", "05", "06", "07", "08", "09", "10"],
                 window_size=2,
                 overlap=1,
                 read_poses=True,
                 transform=transforms.Compose([transforms.ToTensor()])
                 ):
        self.data_path = data_path
        self.gt_path = gt_path
        self.pose_path = pose_path
        self.camera_id = camera_id
        self.frame_id = 0
        self.read_poses = read_poses
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform

        # KITTI normalization
        self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        # define sequence for training, test and val
        self.sequences = sequences

        # read frames list and ground truths
        frames, seqs = self.read_frames()
        accels, gyros, vels, imu_poses= self.read_imu()
        ts_rgbs, ts_imus, rgb2imu_syncs = self.read_timestamps()
        self.rgb2imu_syncs = rgb2imu_syncs
        gt = self.read_gt()

        # create dataframe with frames and ground truths
        data = pd.DataFrame({"gt": gt})
        data = data["gt"].apply(pd.Series)
        data["frames"] = frames
        data["sequence"] = seqs

        data["accels"] = accels.tolist()
        data["gyros"] = gyros.tolist()
        data["vels"] = vels.tolist()
        data["imu_poses"] = imu_poses.tolist()
        self.data = data
        self.windowed_data = self.create_windowed_dataframe(data)

    def __len__(self):
        return len(self.windowed_data["w_idx"].unique())

    def __getitem__(self, idx):
        """
        Returns:
            frame {ndarray}: image frame at index self.frame_id
            pose {list}: list containing the ground truth pose [x, y, z]
            frame_id {int}: integer representing the frame index
        """
        # get data of corresponding window index
        data = self.windowed_data.loc[self.windowed_data["w_idx"] == idx, :]

        # Read IMU data
        accels = data["accels"].values
        accels = np.array([list(accel) for accel in accels])
        accels = torch.tensor(accels, dtype=torch.float32)
        gyros = data["gyros"].values
        gyros = np.array([list(gyro) for gyro in gyros])
        gyros = torch.tensor(gyros, dtype=torch.float32)
        vel_locs = data["vels"].values
        vel_locs = np.array([list(vel_loc) for vel_loc in vel_locs])
        vel_locs = torch.tensor(vel_locs, dtype=torch.float32)
        imu_poses = data["imu_poses"].values
        imu_poses = np.array([list(imu_pose) for imu_pose in imu_poses])
        imu_poses = torch.tensor(imu_poses, dtype=torch.float32)
        # y_imu = []
        # for i in range(len(imu_poses)):
        #     squad_tensor = imu_poses[i, :,:]
        #     R = squad_tensor[:3, :3]
        #     t = squad_tensor[:3, 3]
        #     # Euler parameterization (rotations as Euler angles)
        #     angles = rotation_to_euler(R, seq='zyx')
        #     # normalization
        #     angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
        #     t = (np.asarray(t) - self.mean_t) / self.std_t
        #     y_imu.append(list(angles) + list(t))
        # y_imu = torch.tensor(y_imu,dtype=torch.float32)
        imu_data = torch.cat((accels, gyros, vel_locs, imu_poses), dim=1)
        # Read frames as grayscale
        frames = data["frames"].values
        imgs = []
        
        for fname in frames:
            img = Image.open(fname).convert('RGB')
            # pre processing
            img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=0)
        imgs = np.asarray(imgs)
        # T C H W -> C T H W.
        imgs = imgs.transpose(1, 0, 2, 3)

        # Read ground truth [window_size-1 x 6]
        gt_poses = data.loc[:, [i for i in range(12)]].values
        y = []
        for gt_idx, gt in enumerate(gt_poses):

            # homogeneous pose matrix [4 x 4]
            pose = np.vstack([np.reshape(gt, (3, 4)), [[0., 0., 0., 1.]]])
            tstmp = gt_idx
            # compute relative pose from frame1 to frame2
            if gt_idx > 0:
                pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose)
                R = pose_wrt_prev[:3, :3]
                t = pose_wrt_prev[:3, 3]

                # Euler parameterization (rotations as Euler angles)
                angles = rotation_to_euler(R, seq='zyx')

                # normalization
                angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
                t = (np.asarray(t) - self.mean_t) / self.std_t

                # concatenate angles and translation
                y.append(list(angles) + list(t))

            pose_prev = pose

        y = np.asarray(y)  # discard first value
        y = y.flatten()

        return imgs, y, tstmp, imu_data
 
    def read_frames(self):
        # Get frames list
        frames = []
        seqs = []
        for sequence in self.sequences:
            frames_dir = os.path.join(self.data_path,"sequences", sequence, "image_{}".format(self.camera_id), "*.png")
            frames_seq = sorted(glob.glob(frames_dir))
            frames = frames + frames_seq
            seqs = seqs + [sequence] * len(frames_seq)
        return frames, seqs

    def read_gt(self):
        # Read ground truth
        if self.read_gt:
            gt = []
            for sequence in self.sequences:
                with open(os.path.join(self.gt_path, sequence + ".txt")) as f:
                    lines = f.readlines()

                # convert poses to float
                for line_idx, line in enumerate(lines):
                    line = line.strip().split()
                    line = [float(x) for x in line]
                    gt.append(line)

        else:  # test data (sequences 11-21)
            gt = None

        return gt
    
    def read_imu(self):
        vels_locals = []
        accels = []
        gyros = []
        imu_poses = []
        _, _, rgb2imu_syncs = self.read_timestamps()
        for idx, sequence in enumerate(self.sequences):

            dataset = pykitti.odometry(self.data_path, sequence)

            T_w_imu = np.array([oxts_frame.T_w_imu for oxts_frame in dataset.oxts])
            T_w_imu = T_w_imu[0:len(rgb2imu_syncs[idx])]
            imu_pose = pp.from_matrix(torch.tensor(T_w_imu).to(dtype=torch.float32).numpy(), ltype=pp.SE3_type)
            vels_local = torch.tensor([[oxts_frame.packet.vf, oxts_frame.packet.vl, oxts_frame.packet.vu] for oxts_frame in dataset.oxts], dtype=torch.float32)
            vels_local = vels_local[0:len(rgb2imu_syncs[idx])]
            accel = np.array([[oxts_frame.packet.ax, oxts_frame.packet.ay, oxts_frame.packet.az] for oxts_frame in dataset.oxts]).astype(np.float32)
            accel = accel[0:len(rgb2imu_syncs[idx])]
            gyro = np.array([[oxts_frame.packet.wx, oxts_frame.packet.wy, oxts_frame.packet.wz] for oxts_frame in dataset.oxts]).astype(np.float32)
            gyro = gyro[0:len(rgb2imu_syncs[idx])]
            imu_poses.append(imu_pose)
            accels.append(accel)
            gyros.append(gyro) 
            vels_locals.append(vels_local)

        vels_locals = np.concatenate(vels_locals,axis=0)
        accels = np.concatenate(accels,axis=0)
        gyros = np.concatenate(gyros,axis=0)
        imu_poses = np.concatenate(imu_poses,axis=0)
        return accels, gyros, vels_locals, imu_poses
    
    def read_timestamps(self):
        ts_rgbs = []
        ts_imus = []
        rgb2imu_syncs = []
        for sequence in self.sequences:
            ts_rgb = self.load_timestamps(sequence)
            ts_imu = self.load_oxt_timestamps(self.data_path + "/sequences" + "/" + sequence, 'oxts')
            rgb2imu_sync = sync_data(ts_imu, ts_rgb)
            rgb2imu_syncs.append(rgb2imu_sync)
            ts_rgbs.append(ts_rgb)
            ts_imus.append(ts_imu)
        return ts_rgbs, ts_imus, rgb2imu_syncs

    def create_windowed_dataframe(self, df):
        window_size = self.window_size
        overlap = self.overlap
        windowed_df = pd.DataFrame()
        w_idx = 0

        for sequence in df["sequence"].unique():
            seq_df = df.loc[df["sequence"] == sequence, :].reset_index(drop=True)
            row_idx = 0
            while row_idx + window_size <= len(seq_df):
                rows = seq_df.iloc[row_idx:(row_idx + window_size)].copy()
                rows["w_idx"] = len(rows) * [w_idx]  # add window index column
                row_idx = row_idx + window_size - overlap
                w_idx = w_idx + 1
                windowed_df = pd.concat([windowed_df, rows], ignore_index=True)
        windowed_df.reset_index(drop=True)
        return windowed_df
    
    def load_timestamps(self,sequence):
        import datetime as dt
        """Load timestamps from file."""
        sequence_path = os.path.join(self.data_path+"/sequences", sequence)
        timestamp_file = os.path.join(sequence_path, 'times.txt')
        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                t = dt.timedelta(seconds=float(line))
                timestamps.append(t.total_seconds())
        return timestamps

    def load_oxt_timestamps(self, datapath, subfolder):
        import datetime as dt

        """Load timestamps from file."""
        timestamp_file = os.path.join(
            datapath, subfolder, 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits.
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t.timestamp())
        timestamps.sort()
        return timestamps