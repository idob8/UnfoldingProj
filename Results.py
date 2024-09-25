import numpy as np
import matplotlib.pyplot as plt
from Mesh import Mesh
from Unfolder import Unfolder
import os
# Results Notebook with Customizable Save Path

import numpy as np
import matplotlib.pyplot as plt
from Mesh import Mesh
from Unfolder import Unfolder
import os

def save_3d_mesh(mesh, filename):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh.poly3d)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set equal aspect ratio
    x = mesh.vertices[:,0]
    y = mesh.vertices[:,1]
    z = mesh.vertices[:,2]
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.title('3D Mesh Visualization')
    plt.savefig(filename)
    plt.close(fig)

def save_2d_unfolding(mesh_2d, filename):
    fig, ax = plt.subplots(figsize=(10, 10))
    for _, vertices_2d in mesh_2d.polygons.items():
        polygon = plt.Polygon(list(vertices_2d.values()), fill=None, edgecolor='black')
        ax.add_patch(polygon)
    ax.autoscale()
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Unfolding Visualization')
    plt.savefig(filename)
    plt.close(fig)

def run_experiment(mesh_path, n_iterations, transformation_method, transformation_params, results_path):
    # Create the results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)
    
    # Load the mesh
    mesh = Mesh()
    mesh.read_off(mesh_path)
    
    # Initialize the unfolder
    unfolder = Unfolder(mesh)
    
    # Initialize lists to store results
    collision_counts = []
    total_squared_angle_deficits = []
    
    # Initial unfolding
    tree = unfolder.steepest_edge_unfolder()
    unfolder.unfold_mesh_along_tree(tree)
    initial_collisions = unfolder.mesh_2d.count_collisions()
    collision_counts.append(initial_collisions)
    total_squared_angle_deficits.append(mesh.calculate_total_squared_angle_deficit())
    
    print(f"Initial collisions: {initial_collisions}")
    
    # Save initial state visualizations
    save_3d_mesh(mesh, os.path.join(results_path, f'mesh_3d_{transformation_method}_iteration_0.png'))
    save_2d_unfolding(unfolder.mesh_2d, os.path.join(results_path, f'mesh_2d_{transformation_method}_iteration_0.png'))
    
    # Main iteration loop
    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}")
        
        # Apply transformation
        if transformation_method == "MCF":
            mesh.mean_curvature_flow(n_iterations=transformation_params['n_iterations'], 
                                     step_factor=transformation_params['step_factor'])
        elif transformation_method == "ENAF":
            mesh.edge_normal_alignment_flow(n_iterations=transformation_params['n_iterations'], 
                                            step_size=transformation_params['step_size'])
        else:
            raise ValueError("Unknown transformation method")
        
        # Update unfolder with transformed mesh
        unfolder = Unfolder(mesh)
        
        # Re-run unfolding
        tree = unfolder.steepest_edge_unfolder()
        unfolder.unfold_mesh_along_tree(tree)
        
        # Count collisions and calculate angle deficit
        collisions = unfolder.mesh_2d.count_collisions()
        collision_counts.append(collisions)
        total_squared_angle_deficits.append(mesh.calculate_total_squared_angle_deficit())
        
        print(f"Collisions after iteration {i+1}: {collisions}")
        
        # Save current state visualizations
        save_3d_mesh(mesh, os.path.join(results_path, f'mesh_3d_{transformation_method}_iteration_{i+1}.png'))
        save_2d_unfolding(unfolder.mesh_2d, os.path.join(results_path, f'mesh_2d_{transformation_method}_iteration_{i+1}.png'))
    
    return collision_counts, total_squared_angle_deficits

def plot_results(collision_counts, total_squared_angle_deficits, transformation_method, results_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(collision_counts, marker='o')
    ax1.set_title(f'Collisions vs Iterations ({transformation_method})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Collisions')
    
    ax2.plot(total_squared_angle_deficits, marker='o')
    ax2.set_title(f'Total Squared Angle Deficit vs Iterations ({transformation_method})')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Total Squared Angle Deficit')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'results_{transformation_method}.png'))
    plt.close(fig)



mesh_path = "/Users/meravkeidar/OneDrive/Technion/semester4/DGP/DigitalGeometryProcessing/HW2/hw2_data/phands.off"
results_path = "/Users/meravkeidar/OneDrive/Technion/semester4/DGP/project/Results/phands"
n_iterations = 4

# Run experiment with Mean Curvature Flow
mcf_params = {'n_iterations': 4, 'step_factor': 0.01}
mcf_collisions, mcf_angle_deficits = run_experiment(mesh_path, n_iterations, "MCF", mcf_params, results_path)
plot_results(mcf_collisions, mcf_angle_deficits, "Mean_Curvature_Flow", results_path)

# Run experiment with Edge Normal Alignment Flow
enaf_params = {'n_iterations': 4, 'step_size': 0.01}
enaf_collisions, enaf_angle_deficits = run_experiment(mesh_path, n_iterations, "ENAF", enaf_params, results_path)
plot_results(enaf_collisions, enaf_angle_deficits, "Edge_Normal_Alignment_Flow", results_path)

# Compare the two methods
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(mcf_collisions, marker='o', label='MCF')
ax1.plot(enaf_collisions, marker='s', label='ENAF')
ax1.set_title('Collisions vs Iterations (MCF vs ENAF)')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Number of Collisions')
ax1.legend()

ax2.plot(mcf_angle_deficits, marker='o', label='MCF')
ax2.plot(enaf_angle_deficits, marker='s', label='ENAF')
ax2.set_title('Total Squared Angle Deficit vs Iterations (MCF vs ENAF)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Total Squared Angle Deficit')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_path, 'comparison_MCF_vs_ENAF.png'))
plt.close(fig)

