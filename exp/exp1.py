import open3d as o3d
import numpy as np
import trimesh

obj_path =...
mesh = trimesh.load(obj_path)

vertices = np.array(mesh.vertices)
if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
    colors = mesh.visual.vertex_colors
else:
    colors = np.ones((len(vertices), 4)) * 255
    colors[:, 3] = 255

pcd = trimesh.PointCloud(vertices, colors=colors)
save_path = ...
pcd.export(save_path)

print(f"点云数据已保存到: {save_path}")
print(f"点云数量: {len(vertices)}")
print(f"点云维度: {vertices.shape}")
print(f"是否包含颜色信息: {hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors')}")
