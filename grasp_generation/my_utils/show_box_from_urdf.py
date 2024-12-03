import os
import trimesh
import matplotlib.pyplot as plt
from urdfpy import URDF
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import kinpy as kp

def load_urdf_and_visualize_with_kinpy(urdf_path):
    # Step 1: Load the URDF file
    robot = URDF.load(urdf_path)
    print(f"Loaded URDF: {robot.name}")
    
    # Step 2: Generate kinematic chain using kinpy
    chain = kp.build_chain_from_urdf(open(urdf_path).read())
    print(f"Kinematic Chain: {chain}")

    # Step 3: Traverse links and get STL files
    meshes = []
    bboxes = []
    for link in robot.links:
        if link.visuals:
            for visual in link.visuals:
                if visual.geometry.mesh:
                    stl_path = visual.geometry.mesh.filename
                    if not os.path.isabs(stl_path):
                        # Convert relative paths to absolute paths (assumes URDF's directory as base)
                        stl_path = os.path.join(os.path.dirname(urdf_path), stl_path)
                    
                    # Load STL file
                    mesh = trimesh.load_mesh(stl_path)
                    meshes.append((link.name, mesh))
                    
                    # Calculate bounding box
                    bbox = mesh.bounds  # (min, max)
                    bboxes.append((link.name, bbox))
    
    # Step 4: Visualize STL models and bounding boxes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for (link_name, mesh), (bbox_name, bbox) in zip(meshes, bboxes):
        # Plot the STL mesh
        ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.3, edgecolor='k'))
        
        # Plot the bounding box
        bbox_min, bbox_max = bbox
        x = [bbox_min[0], bbox_max[0], bbox_max[0], bbox_min[0], bbox_min[0], bbox_max[0], bbox_max[0], bbox_min[0]]
        y = [bbox_min[1], bbox_min[1], bbox_max[1], bbox_max[1], bbox_min[1], bbox_min[1], bbox_max[1], bbox_max[1]]
        z = [bbox_min[2], bbox_min[2], bbox_min[2], bbox_min[2], bbox_max[2], bbox_max[2], bbox_max[2], bbox_max[2]]
        
        # Draw bbox edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom square
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top square
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical lines
        ]
        for edge in edges:
            ax.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], [z[edge[0]], z[edge[1]]], 'r')
        
        # Label the link
        center = (bbox_min + bbox_max) / 2
        ax.text(center[0], center[1], center[2], link_name, color='blue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('STL Models with Bounding Boxes')
    plt.show()

# Example usage
urdf_path = "Kable_Hand_mjcf/urdf/Kable_Hand.urdf"  # Replace with the actual URDF file path
load_urdf_and_visualize_with_kinpy(urdf_path)
