import numpy as np
import matplotlib.pyplot as plt
from Mesh import Mesh
from Unfolder import Unfolder
import os

def save_3d_mesh(mesh, filename, method, iteration, mesh_name):
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
    plt.title(f'3D Mesh Visualization - {mesh_name}\n{method} - Iteration {iteration}')
    plt.savefig(filename)
    plt.close(fig)

def save_2d_unfolding(mesh_2d, filename, method, iteration, mesh_name, collisions):
    fig, ax = plt.subplots(figsize=(10, 10))
    for _, vertices_2d in mesh_2d.polygons.items():
        polygon = plt.Polygon(list(vertices_2d.values()), fill=None, edgecolor='black')
        ax.add_patch(polygon)
    ax.autoscale()
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'2D Unfolding Visualization - {mesh_name}\n{method} - Iteration {iteration}\nCollisions: {collisions}')
    plt.savefig(filename)
    plt.close(fig)

def run_experiment(mesh_path, mesh_name, n_iterations, transformation_method, transformation_params, results_path):
    os.makedirs(results_path, exist_ok=True)
    mesh = Mesh()
    mesh.read_off(mesh_path)
    unfolder = Unfolder(mesh)
    collision_counts = []
    # Initial unfolding
    tree = unfolder.steepest_edge_unfolder()
    unfolder.unfold_mesh_along_tree(tree)
    initial_collisions = unfolder.mesh_2d.count_collisions()
    collision_counts.append(initial_collisions)
    print(f"Initial collisions: {initial_collisions}")
    # Save initial state visualizations
    save_3d_mesh(mesh, os.path.join(results_path, f'mesh_3d_{transformation_method}_iteration_0.png'), transformation_method, 0, mesh_name)
    save_2d_unfolding(unfolder.mesh_2d, os.path.join(results_path, f'mesh_2d_{transformation_method}_iteration_0.png'), transformation_method, 0, mesh_name, initial_collisions)
    
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
        # Count collisions
        collisions = unfolder.mesh_2d.count_collisions()
        collision_counts.append(collisions)
        print(f"Collisions after iteration {i+1}: {collisions}")
        # Save current state visualizations
        save_3d_mesh(mesh, os.path.join(results_path, f'mesh_3d_{transformation_method}_iteration_{i+1}.png'), transformation_method, i+1, mesh_name)
        save_2d_unfolding(unfolder.mesh_2d, os.path.join(results_path, f'mesh_2d_{transformation_method}_iteration_{i+1}.png'), transformation_method, i+1, mesh_name, collisions)
        
    return collision_counts

def plot_results(collision_counts, transformation_method, results_path, mesh_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(collision_counts, marker='o')
    ax.set_title(f'Collisions vs Iterations - {mesh_name}\n({transformation_method})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Collisions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'results_{transformation_method}_{mesh_name}.png'))
    plt.close(fig)



def save_angle_deficit_plot(angle_deficits, mesh_name, results_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(angle_deficits)), angle_deficits, marker='o')
    ax.set_title(f'Total Squared Angle Deficit - {mesh_name})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Squared Angle Deficit')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{mesh_name}_angle_deficit.png'))
    plt.close(fig)

def visualize_mesh_transformations(mesh_path, mesh_name, n_iterations, transformation_method, transformation_params, results_path):
    os.makedirs(results_path, exist_ok=True)
    mesh = Mesh()
    mesh.read_off(mesh_path)
    save_3d_mesh(mesh, os.path.join(results_path, f'{mesh_name}_{transformation_method}_iteration_0.png'), 
                 transformation_method, 0, mesh_name)

    angle_deficits = [mesh.calculate_total_squared_angle_deficit()]
    
    # Apply transformation and save visualizations
    for i in range(1, n_iterations + 1):
        if transformation_method == "MCF":
            mesh.mean_curvature_flow(n_iterations=transformation_params['n_iterations'], 
                                     step_factor=transformation_params['step_size'])
        elif transformation_method == "ENAF":
            mesh.edge_normal_alignment_flow(n_iterations=transformation_params['n_iterations'], 
                                            step_size=transformation_params['step_size'])
        else:
            raise ValueError("Unknown transformation method. Use 'MCF' or 'ENAF'.")
        
        angle_deficits.append(mesh.calculate_total_squared_angle_deficit())
        save_3d_mesh(mesh, os.path.join(results_path, f'{mesh_name}_{transformation_method}_iteration_{i}.png'), 
                     transformation_method, i, mesh_name)
    
    # Save angle deficit plot
    if transformation_method == "ENAF":
        save_angle_deficit_plot(angle_deficits, mesh_name, results_path)
    
    # Create a figure combining all iterations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{transformation_method} Transformation of {mesh_name}', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i <= n_iterations:
            img = plt.imread(os.path.join(results_path, f'{mesh_name}_{transformation_method}_iteration_{i}.png'))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Iteration {i}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'{mesh_name}_{transformation_method}_transformations.png'))
    plt.close(fig)
    
# '''Preform MCF transformation'''
# mesh_path = "/Users/meravkeidar/OneDrive/Technion/semester4/DGP/DigitalGeometryProcessing/HW2/hw2_data/frog_s3.off"
# mesh_name = "frog"
# results_path = "/Users/meravkeidar/OneDrive/Technion/semester4/DGP/project/Results/MCF/frog"
# n_iterations = 5
# mcf_params = {'n_iterations': 1, 'step_size': 0.001}
# visualize_mesh_transformations(mesh_path, mesh_name, n_iterations, "MCF", mcf_params, results_path)

# '''Preform ENAF transformation'''
# results_path = "/Users/meravkeidar/OneDrive/Technion/semester4/DGP/project/Results/ENAF/frog"
# enaf_params = {'n_iterations': 1, 'step_size': 0.001}
# visualize_mesh_transformations(mesh_path, mesh_name, n_iterations, "ENAF", enaf_params, results_path)

'''Iteratevly transform mesh, find unfolding and collisions '''
mesh_path = "/Users/meravkeidar/OneDrive/Technion/semester4/DGP/DigitalGeometryProcessing/HW2/hw2_data/phands.off"
mesh_name = "phands"  
results_path = "/Users/meravkeidar/OneDrive/Technion/semester4/DGP/project/Results/phands"
n_iterations = 5

# Run experiment with Mean Curvature Flow
mcf_params = {'n_iterations': 1, 'step_factor': 0.001}
mcf_collisions = run_experiment(mesh_path, mesh_name, n_iterations, "MCF", mcf_params, results_path)
plot_results(mcf_collisions, "Mean_Curvature_Flow", results_path, mesh_name)

# Run experiment with Edge Normal Alignment Flow
enaf_params = {'n_iterations': 1, 'step_size': 0.001}
enaf_collisions = run_experiment(mesh_path, mesh_name, n_iterations, "ENAF", enaf_params, results_path)
plot_results(enaf_collisions, "Edge_Normal_Alignment_Flow", results_path, mesh_name)

# Compare the two methods
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(mcf_collisions, marker='o', label='MCF')
ax.plot(enaf_collisions, marker='s', label='ENAF')
ax.set_title(f'Collisions vs Iterations (MCF vs ENAF) - {mesh_name}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Number of Collisions')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_path, f'comparison_MCF_vs_ENAF_{mesh_name}.png'))
plt.close(fig)