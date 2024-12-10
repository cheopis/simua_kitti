
import os
import cv2
import glob
import time
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import tensorflow as tf
import csv

from ultralytics import YOLO
from ultralytics.utils import ASSETS, yaml_load

from LiDAR2Camera import LiDAR2Camera

# LOAD THE FILES FROM KITTI DATASET
#   
folder =  os.getcwd() + '/datas/'
image_files = sorted(glob.glob(folder + "cam2/*.jpg"))
point_files = sorted(glob.glob(folder + "velodyne_points/CSV/FilesPCD/*.pcd"))
calib_files = sorted(glob.glob(folder + "calib/calib_*.txt"))

# Show only for the first image image [0]:
index = 266
pcd_file = point_files[index]
image = cv2.cvtColor(cv2.imread(image_files[index]), cv2.COLOR_BGR2RGB)
#cloud = list(csv.reader(pcd_file, delimiter=' '))
cloud = o3d.io.read_point_cloud(pcd_file)
pcl= np.asarray(cloud)
print(pcl)

# Lidar-Camera Fusion
lidar2cam = LiDAR2Camera(calib_files)

img_3 = image.copy()
img_3 = lidar2cam.show_lidar_on_image(pcl, img_3)
image_pcl = lidar2cam

# Build model
model = YOLO("yolov8m-seg.pt")
#model.export(format="onnx", imgsz=640, opset=12)

conf = 0.25
iou = 0.45

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
colors = [random.choices(range(256), k=3) for _ in classes_ids]

results = model.predict(image, conf=conf, iou=iou, show=True)

def bounding_box(points, min_x=-np.inf, max_x=np.inf, 
                 min_y=-np.inf, max_y=np.inf):
    box = []
    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)

    bb_filter = np.logical_and(bound_x, bound_y)
    for i, bb in enumerate(bb_filter):
        if bb:
            box.append(points[i])

    return box

def run_obstacle_detection(image, results, points):

    imgfov_pc_velo, pts_2d, fov_inds = lidar2cam.get_lidar_in_image_fov(
            points, 0, 0, image.shape[1], image.shape[0], True)
    depths = []

    for result in results:
        for box in result.boxes:
            box = box.cpu() # COmentar essa linha caso usar tensoflow na cpu
            x_min = np.int32(box.xyxy)[0][0]
            x_max = np.int32(box.xyxy)[0][2]
            y_min = np.int32(box.xyxy)[0][1]
            y_max = np.int32(box.xyxy)[0][3]

            # Só tem uma imagem, então só uma cor
            cv2.rectangle(image, 
                          (x_min, y_min), 
                          (x_max, y_max), 
                          (0, 0, 255), 2)
            
            bb = bounding_box(pts_2d, 
                         x_min,
                         x_max,
                         y_min,
                         y_max)

            bb_center = np.array((x_max + x_min)/2, (y_max + y_min)/2)
            distances = np.abs(bb - bb_center)
            min_index = np.argmin(distances)
            print(bb[min_index])

            for point in bb:
                cv2.circle(
                    image,(int(np.round(point[0])), int(np.round(point[1]))),2,
                    (255, 0, 0),
                    thickness=-1,
                )
            cv2.circle(
                    image,(int(np.round(bb[min_index][0])), int(np.round(bb[min_index][1]))),2,
                    (0, 255, 0),
                    thickness=-1,
                )
            

    return depths



depth = run_obstacle_detection(image, results, pcl)


fig_camera = plt.figure(figsize=(14, 7))
ax_lidar = fig_camera.subplots()
ax_lidar.imshow(image)
plt.show()
