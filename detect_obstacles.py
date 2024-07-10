
import os
import cv2
import glob
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import tensorflow as tf

from yolov4.tf import YOLOv4



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

yolo = YOLOv4()
yolo.config.parse_names("data/coco.names")
yolo.config.parse_cfg("data/yolov4-tiny.cfg")
yolo.make_model()
yolo.load_weights("data/yolov4-tiny.weights", weights_type="yolo")

def run_obstacle_detection(img):
    start_time=time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = yolo.resize_image(img)
    # 0 ~ 255 to 0.0 ~ 1.0
    resized_image = resized_image / 255.
    #input_data == Dim(1, input_size, input_size, channels)
    input_data = resized_image[np.newaxis, ...].astype(np.float32)

    candidates = yolo.model.predict(input_data)

    _candidates = []
    result = img.copy()
    for candidate in candidates:
        batch_size = candidate.shape[0]
        grid_size = candidate.shape[1]
        _candidates.append(tf.reshape(candidate, shape=(1, grid_size * grid_size * 3, -1)))
        #candidates == Dim(batch, candidates, (bbox))
        candidates = np.concatenate(_candidates, axis=1)
        print(candidates.shape)
        #pred_bboxes == Dim(candidates, (x, y, w, h, class_id, prob))
        pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0], iou_threshold=0.35, score_threshold=0.40)
        pred_bboxes = pred_bboxes[~(pred_bboxes==0).all(1)] #https://stackoverflow.com/questions/35673095/python-how-to-eliminate-all-the-zero-rows-from-a-matrix-in-numpy?lq=1
        pred_bboxes = yolo.fit_pred_bboxes_to_original(pred_bboxes, img.shape)
        exec_time = time.time() - start_time
        #print("time: {:.2f} ms".format(exec_time * 1000))
        result = yolo.draw_bboxes(img, pred_bboxes)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result, pred_bboxes

result, pred_bboxes = run_obstacle_detection(image)



fig_camera = plt.figure(figsize=(14, 7))
ax_lidar = fig_camera.subplots()
ax_lidar.imshow(result)
plt.show()