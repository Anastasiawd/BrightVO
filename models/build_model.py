from collections import OrderedDict
import torch
import yaml
from pathlib import Path
from models.ViOT import VisualOdometryTransformer as ViOT
from easydict import EasyDict
import os
import numpy as np

default_cfgs = {
    "vit_patch16_edim768":
        {
            'url': 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth', #'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
            'first_conv': 'patch_embed.proj',
            'classifier': 'head',
        },
    "vit_patch16_edim192":
        {
            'url': "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            'first_conv': 'patch_embed.proj',
            'classifier': 'head',
        },
    "vit_patch32_edim1024":
        {
            'url': "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
            'first_conv': 'patch_embed.proj',
            'classifier': 'head',
        },
    "vit_patch16_edim384":
        {
            'url': "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            'first_conv': 'patch_embed.proj',
            'classifier': 'head',
        },
}
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

def load_pretrained(model, ckpt):
    ckpt = ckpt['model']
    ckpt_keys = list(ckpt.keys())
    model_keys = list(model.encoder.state_dict().keys())
    new_dict = OrderedDict()
    j = 0
    for i,key in enumerate(model_keys):
        if "brightness" in key:
            continue
        if "illu" in key:
            continue
        if model_keys[i] == ckpt_keys[j]:
            new_dict[key] = ckpt[key]
            j += 1
        elif "query.weight" in key:
            new_dict[key] = ckpt[ckpt_keys[j]][:768]
        elif "query.bias" in key:
            new_dict[key] = ckpt[ckpt_keys[j+1]][:768]
        elif "key.weight" in key:
            new_dict[key] = ckpt[ckpt_keys[j]][768:768*2]
        elif "key.bias" in key:
            new_dict[key] = ckpt[ckpt_keys[j+1]][768:768*2]
        elif "value.weight" in key:
            new_dict[key] = ckpt[ckpt_keys[j]][768*2:]
        elif "value.bias" in key:
            new_dict[key] = ckpt[ckpt_keys[j+1]][768*2:]
            j += 2
        else:
            new_dict[key] = ckpt[ckpt_keys[j]]
            j += 1
    
    u,v = model.encoder.load_state_dict(new_dict, strict=False)
    print("Unsuccessful keys: ", u)
    print("Missing keys: ", v)
    print(" --- loaded pretrained weights ---")

    return model
    
def create_model(args):
    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.LOCAL_RANK = 0
    cfg_from_yaml_file(args.cfg_file, cfg)

    # build and load model
    model = ViOT(config=cfg.MODEL)
    args = vars(args)

    # load checkpoint
    args["epoch_init"] = 1
    args["best_val"] = np.inf
    if len(args["resume_path"]) > 0:
        checkpoint = torch.load(args["resume_path"])
        args["epoch_init"] = checkpoint["epoch"] + 1 
        args["best_val"] = checkpoint["best_val"]
        model.load_state_dict(checkpoint['model_state_dict'])

    elif args["pretrained_path"]:  # load ImageNet weights
        img_size = cfg.MODEL.PR_ENCODER.IMAGE_SIZE
        num_patches = (img_size[0] // cfg.MODEL.PR_ENCODER.PATCH_SIZE) * (img_size[1] // cfg.MODEL.PR_ENCODER.PATCH_SIZE)
        model_name = "vit_patch{}_edim{}".format(cfg.MODEL.PR_ENCODER.PATCH_SIZE, cfg.MODEL.PR_ENCODER.HIDDEN_SIZE)
        model.default_cfg = default_cfgs[model_name]
    
        print(" --- loading pretrained to start training ---")
        print(model.default_cfg["url"] + "\n")
        pretrained_model_path = args["pretrained_path"] + model_name + ".pth"
        ckpt = torch.load(pretrained_model_path)
        model = load_pretrained(model, ckpt)
        # ckpt_rwkv = torch.load('/home/wangdongzhihan/codes/BrightVO-v1/checkpoint/pretrained_model/vrwkv_l_22kto1k_384.pth',weights_only=True)
        # model.load_state_dict(ckpt_rwkv, strict=False)


    if torch.cuda.is_available():
        model.cuda()
    
    return model, args