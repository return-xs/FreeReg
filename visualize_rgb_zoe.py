import open3d as o3d
import numpy as np
from tools.zoe.gen_dpt import img2dpt
import matplotlib.pyplot as plt
from PIL import Image
import os

def depth_to_pointcloud(depth, intrinsic, rgb_image=None):
    """
    将深度图转换为点云
    :param depth: 深度图 (H, W)
    :param intrinsic: 相机内参矩阵 (3, 3)
    :param rgb_image: 彩色图像 (H, W, 3)，可选
    :return: 点云对象
    """
    h, w = depth.shape
    
    # 创建像素坐标网格
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    
    # 获取有效深度值的掩码
    z = depth.flatten()
    valid = z > 0
    
    # 仅保留有效点
    x = x[valid]
    y = y[valid]
    z = z[valid]
    
    # 将像素坐标转换为相机坐标
    x_cam = (x - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y_cam = (y - intrinsic[1, 2]) * z / intrinsic[1, 1]
    
    # 创建点云
    points = np.stack([x_cam, y_cam, z], axis=1)
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 如果提供了彩色图像，则添加颜色信息
    if rgb_image is not None:
        colors = rgb_image[y, x] / 255.0  # 归一化到[0,1]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # 如果没有彩色图像，给点云添加统一的灰色
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    return pcd

def visualize_depth_map(depth, save_path='depth_map.png'):
    """
    可视化深度图并保存为PNG文件
    :param depth: 深度图 (H, W)
    :param save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(depth, cmap='viridis')
    plt.colorbar(label='Depth (m)')
    plt.title('Zoe Depth Estimation')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    
    # 保存图像而不是显示
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Depth map saved to {save_path}")

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

if __name__ == '__main__':
    np.set_printoptions(suppress=True)  # 取消默认科学计数法，open3d无法读取科学计数法表示
    
    # 设置路径
    npy_path = '/home/data/return/paper_exam/FreeReg/data/3dmatch/7-scenes-redkitchen/eval/query/0.depth.gen.npy'
    rgb_path = '/home/data/return/paper_exam/FreeReg/data/3dmatch/7-scenes-redkitchen/query/0.color.png'
    
    # 加载深度数据
    depth_data = np.load(npy_path)
    print(type(depth_data))

    # import tools.zoe.zoedepth.utils.misc as misc
    # misc.save_raw_16bit(depth_data)

    print(f"Depth data shape: {depth_data.shape}")
    print(f"Depth range: {np.min(depth_data)} - {np.max(depth_data)}")
    
    # 默认相机内参矩阵（根据3DMatch数据集常用参数）
    intrinsic = np.array([[585, 0, 320],
                         [0, 585, 240],
                         [0, 0, 1]])
    
    # 加载对应的RGB图像（如果存在）
    rgb_image = None
    try:
        rgb_pil = Image.open(rgb_path).convert('RGB')
        rgb_image = np.array(rgb_pil)
        print(f"RGB image shape: {rgb_image.shape}")
    except Exception as e:
        print(f"Could not load RGB image: {e}")
    
    # 可视化深度图并保存
    visualize_depth_map(depth_data, 'zoe_depth_map.png')
    
    # # 转换为点云
    # pcd = depth_to_pointcloud(depth_data, intrinsic, rgb_image)
    
    # # 显示点云信息
    # print(f"Point cloud has {len(pcd.points)} points")
    
    # # 保存点云为PLY文件（可用于后续查看）
    # o3d.io.write_point_cloud("zoe_pointcloud.ply", pcd)
    # print("Point cloud saved to zoe_pointcloud.ply")
    
    # # 将点云可视化保存为图片
    # visualize_pointcloud_to_image(pcd, 'zoe_pointcloud_view.png')
    
    # print("Visualization complete. Files saved:")
    # print("- zoe_depth_map.png: Depth map visualization")
    # print("- zoe_pointcloud.ply: Point cloud data (PLY format)")
    # print("- zoe_pointcloud_view.png: Point cloud visualization as image")