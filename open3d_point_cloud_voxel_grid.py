import os
import open3d as o3d
import glob


def create_voxel_grid(points):
    voxel_points = []
    # Initialize a point cloud object
    pcd = o3d.geometry.PointCloud()
    # Add the points, colors and normals as Vectors
    pcd.points = o3d.utility.Vector3dVector(points.points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.normals = o3d.utility.Vector3dVector(normals)

    # Create a voxel grid from the point cloud with a voxel_size of 0.01
    voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.01)
    voxels = voxel_grid.get_voxels()
    
    for voxel in voxels:
        voxel_points.append(list(voxel.grid_index))

    return voxel_grid, voxel_points

def main():
    # Read the bunny statue point cloud using numpy's loadtxt
    folder =  os.getcwd() + '/2011_09_26_drive_0084_sync'
    point_files = sorted(glob.glob(folder + "/velodyne_points/data/*.pcd"))
    index = 0
    pcd_file = point_files[index]
    points = o3d.io.read_point_cloud(pcd_file)
    # Separate the into points, colors and normals array
    # points = point_cloud[:,:3]
    # colors = point_cloud[:,3:6]
    # normals = point_cloud[:,6:]

    voxel_grid, voxel_points = create_voxel_grid(points)

    # Initialize a visualizer object
    vis = o3d.visualization.Visualizer()
    # Create a window, name it and scale it
    vis.create_window(window_name='Bunny Visualize', width=800, height=600)

    # Add the voxel grid to the visualizer
    vis.add_geometry(voxel_grid)
    print(voxel_points)
    # We run the visualizater
    vis.run()
    # Once the visualizer is closed destroy the window and clean up
    vis.destroy_window()

if __name__ == '__main__':
    main()