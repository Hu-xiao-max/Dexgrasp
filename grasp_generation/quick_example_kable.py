import os
import random
from utils.hand_model_lite_kable import HandModelMJCFLite
import numpy as np
import transforms3d
import torch
import trimesh

mesh_path = "../data/meshdata"
data_path = "../data/kabledata"


use_visual_mesh = False

hand_file = "Kable_Hand_mjcf/Kable_Hand.xml"

joint_names = ['Thumb_Side_Gear_Joint', 'Thumb_Joint1', 'Thumb_Joint2',
                   'Finger1_Joint1', 'Finger1_Joint2', 
                   'Finger2_Joint1', 'Finger2_Joint2', 
                   'Finger3_Joint1', 'Finger3_Joint2', 
                   'Finger4_Joint1', 'Finger4_Joint2', 
                   ]

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

hand_model = HandModelMJCFLite(
    hand_file,
    "Kable_Hand_mjcf/meshes")

# print(hand_model.joints_lower)
# print(hand_model.joints_upper)

grasp_code_list = []
for code in os.listdir(data_path):
    grasp_code_list.append(code[:-4])

grasp_code = random.choice(grasp_code_list)
print(grasp_code)
grasp_data = np.load(
    os.path.join(data_path, grasp_code+".npy"), allow_pickle=True)
object_mesh_origin = trimesh.load(os.path.join(
    mesh_path, grasp_code, "coacd/decomposed.obj"))
# print(len(grasp_data))
object_mesh_origin.visual.vertex_colors = [255, 0, 0, 255]

index = random.randint(0, len(grasp_data) - 1)


qpos = grasp_data[index]['qpos']
# print(qpos)
rot = np.array(transforms3d.euler.euler2mat(
    *[qpos[name] for name in rot_names]))
rot = rot[:, :2].T.ravel().tolist()
hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name]
                         for name in joint_names], dtype=torch.float, device="cpu").unsqueeze(0)
print(hand_pose)
hand_model.set_parameters(hand_pose)
hand_mesh = hand_model.get_trimesh_data(0)
object_mesh = object_mesh_origin.copy().apply_scale(grasp_data[index]["scale"])

(hand_mesh+object_mesh).show()