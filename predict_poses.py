from torchvision import transforms
from datasets.kitti import KITTI
import pickle
import torch
from timesformer.models.vit import VisionTransformer

from timesformer.geom import losses
from timesformer.geom.losses import geodesic_loss, residual_loss, flow_loss
from timesformer.geom.graph_utils import graph_to_edge_list
import timesformer.geom.projective_ops as pops
from timesformer.geom.ba import BA

from functools import partial
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from timesformer.lietorch import SO3, SE3, Sim3
checkpoint_path = "/home/wangdongzhihan/codes/TSformer-VO/checkpoints/Exp7"
checkpoint_name = "checkpoint_best"
sequences = ["04"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# read hyperparameters and configuration
with open(os.path.join(checkpoint_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
f.close()
model_params = args["model_params"]
args["checkpoint_path"] = checkpoint_path
print(args)

# preprocessing operation
preprocess = transforms.Compose([
    transforms.Resize((model_params["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.0900, 0.0756, 0.0559],
        std=[0.1123, 0.1047, 0.0828]),
])

intrinsics = [584.0133433, 584.0133433, 696, 256]

# build and load model
model = VisionTransformer(img_size=model_params["image_size"],
                          num_classes=model_params["num_classes"],
                          patch_size=model_params["patch_size"],
                          embed_dim=model_params["dim"],
                          depth=model_params["depth"],
                          num_heads=model_params["heads"],
                          mlp_ratio=4,
                          qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          drop_rate=0.,
                          attn_drop_rate=0.,
                          drop_path_rate=0.1,
                          num_frames=model_params["num_frames"],
                          attention_type=model_params["attention_type"])

checkpoint = torch.load(os.path.join(args["checkpoint_path"], "{}.pth".format(checkpoint_name)),
                        map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
if torch.cuda.is_available():
    model.cuda()


for sequence in sequences:
    # test dataloader
    dataset = KITTI(transform=preprocess, sequences=[sequence],
                    window_size=args["window_size"], overlap=args["overlap"])
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,
                                             )

    with tqdm(test_loader, unit="batch") as batchs:
      pred_poses = torch.zeros((1, args["window_size"] - 1, 6), device=device)
      batchs.set_description(f"Sequence {sequence}")
      for images, gt, _ in batchs:
        if torch.cuda.is_available():
            images, gt = images.cuda(), gt.cuda()

            with torch.no_grad():
                model.eval()
                model.training = False

                # predict pose
                pred_pose, _ = model(images.float())
                # backend optimization

                pred_pose = torch.reshape(pred_pose, (args["window_size"] - 1, 6)).to(device)
                pred_pose = pred_pose.unsqueeze(dim=0)
                pred_poses = torch.concat((pred_poses, pred_pose), dim=0)
    
    # save as numpy array
    pred_poses = pred_poses[1:, :, :].cpu().detach().numpy()
    # backend optimizer
    # P, B, _  = pred_poses.shape
    # ii = torch.arange(P - 1, device=device)
    # jj = ii + 1


    save_dir = os.path.join(args["checkpoint_path"], checkpoint_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "pred_poses_{}.npy".format(sequence)), pred_poses)
