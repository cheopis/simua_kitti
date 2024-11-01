
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

from ultralytics import YOLO
from ultralytics.utils import ASSETS, yaml_load

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
pcl= np.asarray(cloud.points)

# Lidar-Camera Fusion
lidar2cam = LiDAR2Camera(calib_files)

img_3 = image.copy()
img_3 = lidar2cam.show_lidar_on_image(pcl[:,:3], img_3)
image_pcl = lidar2cam

# Build model
model = YOLO("yolov8m-seg.pt")
model.export(format="onnx", imgsz=640, opset=12)

conf = 0.25
iou = 0.45

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
colors = [random.choices(range(256), k=3) for _ in classes_ids]


def run_obstacle_detection(image):
    results = model.predict(image, conf=conf, iou=iou, show=True)

    for result in results:
        for mask, box in zip(result.masks, result.boxes):
            points = np.int32([mask.xy])
            print(mask.data)
            # cv2.polylines(img, pcl, True, (255, 0, 0), 1)
            # color_number = classes_ids.index(int(box.cls[0]))
            # cv2.fillPoly(image, points, colors[color_number])
            lidar2cam.get_lidar_in_image_fov(pcl[:,:3],
                                             np.int32(box.xyxy)[0][0],
                                             np.int32(box.xyxy)[0][1],
                                             np.int32(box.xyxy)[0][2],
                                             np.int32(box.xyxy)[0][3],
                                             mask=points)

    

#   return pcl

# result, pred_bboxes = run_obstacle_detection(image)

run_obstacle_detection(image)


fig_camera = plt.figure(figsize=(14, 7))
ax_lidar = fig_camera.subplots()
ax_lidar.imshow(image)
plt.show()
