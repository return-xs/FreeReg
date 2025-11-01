import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'FreeReg'))

from FreeReg.tools.controlnet.utils.share import *
import torch
import einops
import cv2
import numpy as np
from PIL import Image
from copy import deepcopy
from FreeReg.tools.controlnet.utils.util import resize_image, HWC3
from FreeReg.tools.controlnet.diff_feature.basic import capture
from FreeReg.tools.controlnet.diff_feature.sd_img import img_processor


from omegaconf import OmegaConf

def gen_config(cfg_path):
    return OmegaConf.load(cfg_path)

if __name__ == '__main__':
    load_model = True
    config_path = '/home/data/return/paper_exam/FreeReg/config/3dmatch.yaml'

    cfg = gen_config(config_path)

    print(f'config: {cfg.feat}')

    capturer = capture(seed = cfg.feat.cn.seed, basic = './FreeReg/' + cfg.feat.cn.ckpts.basic, 
                                    yaml = cfg.feat.cn.ckpts.yaml, sd_ckpt = cfg.feat.cn.ckpts.sd_ckpt,
                                    cn_ckpt = cfg.feat.cn.ckpts.cn_ckpt, t = cfg.feat.cn.step)
    
    img_processing = img_processor(capturer)

    image_path = '/home/data/return/paper_exam/FreeReg/data/3dmatch/7-scenes-redkitchen/query/0.color.png'

    img, feat_list = img_processing.sd_single_img(image_path, prompt = cfg.feat.cn.prompt)

    # 特征可视化
    for i, feat in enumerate(feat_list):
        print(f'Visualizing feature map at layer {i}, shape: {feat.shape}')
        capturer.visualize_chw_feat(feat, i)