import pyvista as pv
import os
import numpy as np
from scipy.spatial import KDTree


# Function to calculate the radius of a sphere with an equal volume as a cube with a given side
def calculate_sphere_radius(voxel_size = 0.01):
    voxel_volume = voxel_size ** 3
    radius = ((3*voxel_volume)/(4*np.pi))**(1/3)
    return radius

# Function to generate density for each voxel and add it as a field
def calculate_neighbours(mesh, voxel_size = 0.01):
    # voxelize the given mesh with a specified size voxels
    voxels = pv.voxelize(mesh, density=voxel_size, check_surface=False)
    # Get the voxel center points
    voxel_centers = voxels.cell_centers().points
    # Get the mesh vertices
    mesh_vertices = mesh.points
    # Calculate the KDTree of the mesh vertices from Scipy
    kd_tree_vertices = KDTree(mesh_vertices)
    # Call the sphere radius function and calculate the new radius
    radius = calculate_sphere_radius(voxel_size)
    # Use the calculated KDTree and radius to get the neighbors for each voxel center
    neighbours = kd_tree_vertices.query_ball_point(voxel_centers,radius)
    # Count the number of points for each voxel center
    neighbour_count = [len(curr_neighbours) for curr_neighbours in neighbours]
    # Cast to array and normalize between 0 and 1 
    neighbour_count = np.array(neighbour_count, dtype=np.float32)
    neighbour_density =  neighbour_count/neighbour_count.max()
    # Add the density as a field to the voxels
    voxels['density'] = neighbour_density

    return voxels

# Function to visualize and threshold the voxel representation based on the calculated density
def visualize_thresh(voxels):
    p = pv.Plotter()
    p.add_mesh_threshold(voxels,show_edges=True)
    p.show()

if __name__ == '__main__':

    mesh_path = os.path.join('mesh','rooster.obj')
    mesh = pv.read(mesh_path)

    voxels = calculate_neighbours(mesh)
    visualize_thresh(voxels)