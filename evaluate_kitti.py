import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torchvision import transforms
from datasets.kitti import KITTI
import pickle
import torch
import yaml
from pathlib import Path
from models.ViOT import VisualOdometryTransformer as ViOT
from easydict import EasyDict
from functools import partial
import torch.nn as nn
from models.RefinementModule import RefinementBlock
import numpy as np
from tqdm import tqdm
import pypose as pp
def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config  
def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


checkpoint_path = "/Path/to/checkpoint"
checkpoint_name = "brightvo"
sequences = ["01"] 

device = "cuda" if torch.cuda.is_available() else "cpu"



# preprocessing operation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
])

# build and load model
cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
cfg_from_yaml_file("config/cfg.yaml", cfg)

# build and load model
model = ViOT(config=cfg.MODEL)

checkpoint = torch.load(os.path.join(checkpoint_path, "{}.pth".format(checkpoint_name)),
                        map_location='cpu')
model.load_state_dict(checkpoint, strict=True)
if torch.cuda.is_available():
    model.cuda()


for sequence in sequences:
    # test dataloader
    dataset = KITTI(transform=preprocess, sequences=[sequence],
                    window_size=2, overlap=1)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,
                                             )

    with tqdm(test_loader, unit="batch") as batchs:
        pred_poses = torch.zeros((1, 1, 6), device=device)
        batchs.set_description(f"Sequence {sequence}")
        poses_list=[]       
        for idx, (images, imu_motion, tstp, imu_data) in enumerate(batchs):
            if torch.cuda.is_available():
                images, imu_motion, imu_data = images.cuda(), imu_motion.cuda(), imu_data.cuda()
                if idx == 0: 
                    init_pose = imu_data[0,0,-7:].cpu().numpy()
                    poses_list.extend([init_pose])
                with torch.no_grad():
                    model.eval()
                    model.training = False
                    batch_dict = {
                        'images': images.float(),
                        'head_mask': None,
                        'output_attentions': False,
                        'output_hidden_states': True,
                    }
                    # predict pose
                    pred_pose = model(batch_dict)

                    backend = True
                    if not backend:
                        pred_pose = torch.reshape(pred_pose, (1, 6)).to(device)
                        pred_pose = pred_pose.unsqueeze(dim=0)
                        pred_poses = torch.concat((pred_poses, pred_pose), dim=0)
                    if backend:
                        # backend optimization
                        links = torch.tensor([[0,1]],dtype=torch.long).to(device)
                        init_nodes = imu_data[0,:,-7:]
                        init_nodes = pp.LieTensor(init_nodes, ltype=pp.SE3_type).to(device)
                        refined_poses = RefinementBlock(init_nodes, pred_pose, links, imu_motion).numpy()
                        poses_list.extend(refined_poses[1:])



    save_dir = os.path.join(checkpoint_path, checkpoint_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "pred_poses_{}.npy".format(sequence)), np.array(poses_list))
    # np.save(os.path.join(save_dir, "pred_poses_{}_no_backend.npy".format(sequence)), pred_poses.cpu().numpy())
