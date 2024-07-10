import numpy as np
import glob
import open3d as o3d
import struct

folder = '2011_09_26_drive_0084_sync'
point_files = sorted(glob.glob(folder + "/velodyne_points/data/*.bin"))

index = 0

size_float = 4
list_pcd = []

for file_to_open in point_files:
    #file_to_open = point_files[index]
    file_to_save = str(point_files[index])[:-3]+"pcd"
    with open (file_to_open, "rb") as f:
        byte = f.read(size_float*4)
        while byte:
            x,y,z,intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float*4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)

    o3d.io.write_point_cloud(file_to_save, pcd)
    index += 1

print("Convertion finished")