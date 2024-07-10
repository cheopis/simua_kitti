import os
import cv2
import glob

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from LiDAR2Camera import LiDAR2Camera


# LOAD THE FILES FROM KITTI DATASET
#   
folder =  os.getcwd() + '/2011_09_26_drive_0084_sync'
image_files = sorted(glob.glob(folder + "/image_02/data/*.png"))
point_files = sorted(glob.glob(folder + "/velodyne_points/data/*.pcd"))
calib_files = sorted(glob.glob(folder + "/calib/calib_*.txt"))

# Show only for the first image image [0]:
index = 0
pcd_file = point_files[index]
image = cv2.cvtColor(cv2.imread(image_files[index]), cv2.COLOR_BGR2RGB)
cloud = o3d.io.read_point_cloud(pcd_file)
points= np.asarray(cloud.points)

lidar2cam = LiDAR2Camera(calib_files)

img_3 = image.copy()
img_3 = lidar2cam.show_lidar_on_image(points[:,:3], img_3)
plt.figure(figsize=(14,7))
plt.imshow(img_3)
plt.show()

