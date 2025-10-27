import os
import numpy as np
from glob import glob
from PIL import Image
from utils.utils import trans_gt_for_kitti
from tools.dense.depth_map_utils import name2densefunc


class gen_meta(): 
    def __init__(self,cfg):
        self.cfg = cfg
        # basedir with :
        # scene/0.depth.png\0.color.png\0_pose.txt\0.ply
        # scene/eval/
        # scene/query_database_overlap.txt
        self.base = cfg.meta.base
        self.save_base = cfg.meta.feat_base
        # check pairs > gt overlap
        self.pair_type = cfg.meta.pair_type
        self.overlap_check = cfg.meta.overlap_pair
        self.n_seq_pair = cfg.meta.seq_n_pair
        # others
        self.rgb_intrinsic = np.array(cfg.meta.rgb_intrinsic)
        self.dpt_intrinsic = np.array(cfg.meta.dpt_intrinsic)
        self.dpt_scale = cfg.meta.dpt_scale

    def makedirs(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def run(self):
        scenes = glob(f'{self.base}/*')
        scenes.sort()
        metas = {}
        for scene in scenes:
            name = str.split(scene,'/')[-1]
            # init
            metas[name] = {}
            metas[name]['frames'] = {} # 存储所有帧的详细信息
            metas[name]['pairs'] = []
            # frames
            query_fns = glob(f'{scene}/query/*.color.png') # 获取帧路径（每个彩色图对应一帧）
            for q in query_fns:
                q_id = str.split(q,'/')[-1] # 截取帧 id
                q_id = str.split(q_id,'.')[0]
                
                # densify the input sparse depth map
                if self.cfg.meta.densify:
                    dpt_fn = f'{self.base}/{name}/query/{q_id}.dense.depth.png'
                    if not os.path.exists(dpt_fn):
                        dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.depth.png')) / self.dpt_scale
                        # 通过name2densefunc中的函数进行稠密化，再保存为稠密深度图
                        dpt = name2densefunc[self.cfg.meta.densefunc](dpt.astype(np.float32))
                        dpt = dpt * self.dpt_scale
                        dpt = dpt.astype(np.uint16)
                        dpt = Image.fromarray(dpt)
                        dpt.save(dpt_fn)
                else:
                    # 直接使用原始的稀疏深度图
                    dpt_fn = f'{self.base}/{name}/query/{q_id}.depth.png' # 深度图路径  
                
                # gt depth for evaluaiton
                if os.path.exists(f'{self.base}/{name}/query/{q_id}.color.gtd.png'):
                    # 若该场景有专用的 GT（真值）RGB 图，则用作 GT
                    rgb_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.color.gtd.png'))/self.dpt_scale
                else:
                    # 若没有专用的 GT RGB 图，则用 稠密/原始 的 RGB 图作为 GT
                    if self.cfg.meta.densify:
                        rgb_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.dense.depth.png'))/self.dpt_scale
                    else:
                        rgb_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.depth.png'))/self.dpt_scale
                        # 校验图片尺寸
                        assert (rgb_gt_dpt.shape[1] == self.cfg.meta.rgb_size[0]) and (rgb_gt_dpt.shape[0] == self.cfg.meta.rgb_size[1])

                # 同上，作为 GT 的深度图
                if os.path.exists(f'{self.base}/{name}/query/{q_id}.depth.gtd.png'):
                    dpt_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.depth.gtd.png'))/self.dpt_scale
                else:
                    if self.cfg.meta.densify:
                        dpt_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.dense.depth.png'))/self.dpt_scale
                    else:
                        dpt_gt_dpt = np.array(Image.open(f'{self.base}/{name}/query/{q_id}.depth.png'))/self.dpt_scale
                
                item = {
                    'q_id': q_id,
                    # conform to rgb intrinsic and rgb image size
                    'rgb_fn': f'{self.base}/{name}/query/{q_id}.color.png', # RGB 图路径
                    'zoe_fn': f'{self.base}/{name}/eval/query/{q_id}.depth.gen.npy', # RGB 输入到 zoe 生成的深度图路径
                    # conform to dpt intrinsic and dpt image size
                    'dpt_fn': dpt_fn, # 深度图路径
                    'dpt_scale': self.dpt_scale, # 深度缩放因子
                    # for evaluation
                    'rgb_gtd': rgb_gt_dpt,
                    'dpt_gtd': dpt_gt_dpt,
                    # extrinsic
                    'ext':np.loadtxt(f'{self.base}/{name}/query/{q_id}_pose.txt'), # 相机外参
                    # feature fn
                    'to_fn': f'{self.save_base}/{name}/feat/{q_id}.feat.pth', # 特征保存的路径
                }
                metas[name]['frames'][q_id] = item
            # pair 基于 pair_type 生成帧对，是指任意两个不同视角下，对同一场景的拍摄帧
            if self.pair_type in ['overlap']:
                over = np.loadtxt(f'{scene}/query/overlap.txt')
                over = over[over[:,-1]>self.overlap_check] # 读取场景下的overlap.txt，筛选出重叠度大于overlap_check（0.3）的帧对
                for p in over:
                    i, j, o = int(p[0]), int(p[1]), p[2]
                    sext = metas[name]['frames'][f'{i}']['ext'] # 帧 i 的外参
                    text = metas[name]['frames'][f'{j}']['ext'] # 帧 j 的外参
                    gt = np.linalg.inv(text)@sext # 变换矩阵 gt，表示 j 到 i 的位姿变换
                    item = {
                        'q_id': f'{i}',
                        'd_id': f'{j}',
                        'overlap': o,
                        'to_fn': f'{self.save_base}/{name}/match/{i}-{j}.trans.npz',
                        'gt': gt
                    }
                    metas[name]['pairs'].append(item)

            elif self.pair_type in ['seq']:
                for i in range(len(query_fns)-self.n_seq_pair):
                    sext = metas[name]['frames'][f'{i}']['ext']
                    for j in range(1,self.n_seq_pair):
                        text = metas[name]['frames'][f'{i+j}']['ext']
                        gt = np.linalg.inv(text)@sext
                        if self.cfg.name in ['kitti']:
                            gt = trans_gt_for_kitti(gt)
                        item = {
                            'q_id': f'{i}',
                            'd_id': f'{i+j}',
                            'overlap': 0.8, # no use
                            'to_fn': f'{self.save_base}/{name}/match/{i}-{i+j}.trans.npz',
                            'gt':gt
                        }
                        metas[name]['pairs'].append(item)
            else:
                raise TypeError('Wrong pairing type!: cfg.meta.pair_type:overlap/seq')
            # makedirs
            self.makedirs(f'{self.save_base}/{name}/feat')
            self.makedirs(f'{self.save_base}/{name}/match')
        return metas
