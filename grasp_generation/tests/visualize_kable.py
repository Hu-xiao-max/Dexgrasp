"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize hand model using plotly.graph_objects
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath('.'))

import numpy as np
import torch
import trimesh as tm
import transforms3d
import plotly.graph_objects as go
from utils.hand_model_kable import HandModel


torch.manual_seed(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    device = torch.device('cpu')

    # hand model

    hand_model = HandModel(
        mjcf_path='Kable_Hand_mjcf/Kable_Hand.xml',
        mesh_path='Kable_Hand_mjcf/meshes',
        contact_points_path='Kable_Hand_mjcf/contact_points.json',
        penetration_points_path='Kable_Hand_mjcf/penetration_points.json',
        n_surface_points=2000,
        device=device
    )
    # joint_angles = torch.tensor([0.2, 0.5, 0.7, 0.2, 0.3, 0.9, 0.8, 0.4, 0.9, 0.1, 0.4], dtype=torch.float, device=device)
    # rotation = torch.tensor(transforms3d.euler.euler2mat(0, 2 * np.pi / 3, np.pi / 2, axes='rzxz'), dtype=torch.float, device=device)

    # joint_angles = torch.tensor([ 1.8898e-02,  8.0037e-02, -3.4177e-02, -8.1038e-01,  4.4351e-01,
    #       3.8285e-01, -9.6301e-02, -7.4538e-01,  6.5965e-01,  1.4410e+00,
    #      -7.3986e-02], dtype=torch.float, device=device)
    # rotation = torch.tensor(transforms3d.euler.euler2mat(0, 0, 0, axes='rzxz'), dtype=torch.float, device=device)
    # joint_angles = torch.tensor([0.9, -0.9, -0.9, -0.5, -0.5, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=device)
    # rotation = torch.tensor(transforms3d.euler.euler2mat(0, 0, 0, axes='rzxz'), dtype=torch.float, device=device)
    joint_angles = torch.tensor([1.57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=device)
    rotation = torch.tensor(transforms3d.euler.euler2mat(0, 2 * np.pi / 3, np.pi, axes='rzxz'), dtype=torch.float, device=device)
    # hand_pose = torch.cat([torch.tensor([20, 0.6, -0.7], dtype=torch.float, device=device), rotation.T.ravel()[:6], joint_angles])
    hand_pose = torch.cat([torch.tensor([0, 0, 0], dtype=torch.float, device=device), rotation.T.ravel()[:6], joint_angles])
    hand_model.set_parameters(hand_pose.unsqueeze(0))
    # print(hand_model.current_status)

    # info
    surface_points = hand_model.get_surface_points()[0].detach().cpu().numpy()
    contact_candidates = hand_model.get_contact_candidates()[0].detach().cpu().numpy()
    penetration_keypoints = hand_model.get_penetraion_keypoints()[0].detach().cpu().numpy()
    
    print('n_surface_points', surface_points.shape[0])
    print('n_contact_candidates', contact_candidates.shape[0])

    # visualize

    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue', with_contact_points=False)
    surface_points_plotly = [go.Scatter3d(x=surface_points[:, 0], y=surface_points[:, 1], z=surface_points[:, 2], mode='markers', marker=dict(color='lightblue', size=2))]
    contact_candidates_plotly = [go.Scatter3d(x=contact_candidates[:, 0], y=contact_candidates[:, 1], z=contact_candidates[:, 2], mode='markers', marker=dict(color='white', size=2))]
    penetration_keypoints_plotly = [go.Scatter3d(x=penetration_keypoints[:, 0], y=penetration_keypoints[:, 1], z=penetration_keypoints[:, 2], mode='markers', marker=dict(color='red', size=3))]
    for penetration_keypoint in penetration_keypoints:
        mesh = tm.primitives.Capsule(radius=0.011, height=0)
        v = mesh.vertices + penetration_keypoint
        f = mesh.faces
        penetration_keypoints_plotly += [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='burlywood', opacity=0.5)]

    fig = go.Figure(hand_plotly + surface_points_plotly + contact_candidates_plotly + penetration_keypoints_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.show()
