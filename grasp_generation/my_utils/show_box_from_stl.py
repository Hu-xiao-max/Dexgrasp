import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_stl_and_compute_bbox(file_path):
    """
    加载 STL 文件，计算最小包围盒并返回数据
    """
    # 使用 trimesh 加载 STL 文件
    mesh = trimesh.load_mesh(file_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("无法加载 STL 文件或文件格式不正确。")
    
    # 获取顶点坐标
    vertices = mesh.vertices

    # 计算最小包围盒
    min_point = vertices.min(axis=0)  # 最小点 (x_min, y_min, z_min)
    max_point = vertices.max(axis=0)  # 最大点 (x_max, y_max, z_max)

    # 计算包围盒的中心点和长宽高
    center = (min_point + max_point) / 2
    dimensions = max_point - min_point  # [长, 宽, 高]

    return mesh, min_point, max_point, center, dimensions

def visualize_mesh_and_bbox(mesh, min_point, max_point):
    """
    可视化网格模型和其最小包围盒
    """
    # 创建 Matplotlib 的 3D 图形对象
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 STL 文件的三角面片
    ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.3, edgecolor='k'))

    # 绘制最小包围盒
    # 计算包围盒的 8 个顶点
    bbox_vertices = np.array([
        [min_point[0], min_point[1], min_point[2]],
        [max_point[0], min_point[1], min_point[2]],
        [max_point[0], max_point[1], min_point[2]],
        [min_point[0], max_point[1], min_point[2]],
        [min_point[0], min_point[1], max_point[2]],
        [max_point[0], min_point[1], max_point[2]],
        [max_point[0], max_point[1], max_point[2]],
        [min_point[0], max_point[1], max_point[2]],
    ])

    # 定义包围盒的 12 条边（由顶点索引构成）
    bbox_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面四条边
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面四条边
        [0, 4], [1, 5], [2, 6], [3, 7],  # 竖向四条边
    ]

    # 绘制包围盒的线条
    for edge in bbox_edges:
        edge_points = bbox_vertices[edge]
        ax.plot3D(*edge_points.T, color='r')

    # 设置坐标轴范围
    ax.set_xlim(min_point[0], max_point[0])
    ax.set_ylim(min_point[1], max_point[1])
    ax.set_zlim(min_point[2], max_point[2])

    # 设置标题和标签
    ax.set_title('STL Model and Bounding Box', fontsize=16)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()

# 主程序
if __name__ == "__main__":
    # STL 文件路径（请替换为你的文件路径）
    stl_file_path = "Kable_Hand_mjcf/meshes/base_link.STL"
    # stl_file_path = "Kable_Hand_mjcf/meshes/Finger1_Link2.STL"

    # 加载 STL 文件并计算包围盒
    try:
        mesh, min_pt, max_pt, center, dimensions = load_stl_and_compute_bbox(stl_file_path)
        print("Bounding Box Min Point:", min_pt)
        print("Bounding Box Max Point:", max_pt)
        print("Bounding Box Center Point:", center)
        print("Bounding Box Dimensions (Length, Width, Height):", dimensions)
        
        # 可视化模型和包围盒
        visualize_mesh_and_bbox(mesh, min_pt, max_pt)
    except Exception as e:
        print(f"错误: {e}")
