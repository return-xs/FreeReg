

from pipeline.gen_feat import pipeline_feat
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf

def gen_config(cfg_path):
    return OmegaConf.load(cfg_path)

def visualize_pointcloud_to_image(pcd, save_path='pointcloud.png'):
    """
    将点云可视化并保存为图片
    :param pcd: Open3D点云对象
    :param save_path: 保存路径
    """
    # 创建可视化器对象
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 设置为不可见
    vis.add_geometry(pcd)
    
    # 更新渲染器
    vis.poll_events()
    vis.update_renderer()
    
    # 捕获图像
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    
    # 转换并保存图像
    image_array = np.asarray(image) * 255
    image_pil = Image.fromarray(image_array.astype(np.uint8))
    image_pil.save(save_path)
    print(f"Point cloud visualization saved to {save_path}")

if __name__ == "__main__":

    config_path = '/home/data/return/paper_exam/FreeReg/config/3dmatch.yaml'

    cfg = gen_config(config_path)

    print(f'config: {cfg.feat}')

    feat = pipeline_feat(cfg, 
                      update_df_feat=True,
                      update_gf_feat=True
                      )
    
    np.set_printoptions(suppress=True)  # 取消默认科学计数法，open3d无法读取科学计数法表示
    
    # 设置路径
    npy_path = '/home/data/return/paper_exam/FreeReg/data/3dmatch/7-scenes-redkitchen/eval/query/0.depth.gen.npy'
    rgb_path = '/home/data/return/paper_exam/FreeReg/data/3dmatch/7-scenes-redkitchen/query/0.color.png'
    
    # 加载深度数据
    depth_data = np.load(npy_path)
    zoe_spconv_f = feat.spconv_feature_extract(depth_data, 
                                                    dpt_scale=1,
                                                    intrinsic=cfg.meta.rgb_intrinsic, 
                                                    extrinsic=np.eye(4),
                                                    zoe=True) # {pc:,feat:}
    
    pc = zoe_spconv_f['pc']

    idx = zoe_spconv_f['idx']
    feat = zoe_spconv_f['feat']
    
    # 保存为点云
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # # 保存点云为PLY文件（可用于后续查看）
    # o3d.io.write_point_cloud("zoe_pointcloud.ply", pcd)
    # print("Point cloud saved to zoe_pointcloud.ply")



    