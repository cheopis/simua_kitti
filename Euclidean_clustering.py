"""
Processing of Point Cloud data
"""
from ProcessPointcloud import ProcessPointCloud
import os
import glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt 

def planar_detection(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 1.0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([outlier_cloud],
    #                                 zoom=0.8,
    #                                 front=[-0.4999, -0.1659, -0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])
    return outlier_cloud

def clustering(pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.455,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])

if __name__ == "__main__":
    folder =  os.getcwd() + '/2011_09_26_drive_0084_sync'
    point_files = sorted(glob.glob(folder + "/velodyne_points/data/*.pcd"))
    index = 0
    pcd_file = point_files[index]

    pcd = o3d.io.read_point_cloud(pcd_file)
    # o3d.visualization.draw_geometries([pcd])

    # clustering(pcd)
    outlier_cloud = planar_detection(pcd)
    clustering(outlier_cloud)
    

    '''
    # APPLICATION = ProcessPointCloud(pcd_file="point_cloud_data_sample.xyz", nrows_value=100, display_output_flag=True)
    # APPLICATION = ProcessPointCloud(pcd_file="point_cloud_data_sample_2.pcd" ,nrows_value=6000, display_output_flag=False)
    APPLICATION = ProcessPointCloud(pcd_file=pcd_file ,nrows_value=6000, display_output_flag=False)
    clusters = APPLICATION.euclidean_clustering(distance_threshold=5, cluster_parameters={"min_size":2})
    # print(clusters)
    APPLICATION.visualize_clusters(clusters)
    '''