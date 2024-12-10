import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import statistics

class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = []
        for file in calib_file:
            calibs.append(self.read_calib_file(file))
            # ['calib/calib_cam_to_cam.txt', 'calib/calib_imu_to_velo.txt', 'calib/calib_velo_to_cam.txt']
        
        'CALIB CAM TO CAM'
        P = calibs[0]["P_rect_00"]
        self.P = np.reshape(P, [3, 4])
        # Rotation from reference camera coord to rect camera coord
        R0 = calibs[0]["R_rect_00"]
        self.R0 = np.reshape(R0, [3, 3])

        'CALIB VELO TO CAM'
        # Rotation transform from Velodyne coord to reference camera coord
        R = calibs[2]["R"]
        self.R = np.reshape(R, [3, 3])
        # Translation transform from Velodyne coord to reference camera coord
        T = calibs[2]["T"]
        self.T = np.reshape(T, [3, 1])

        delta_f = calibs[2]["delta_f"]
        self.delta_f = np.reshape(delta_f, [2, 1])
        delta_c = calibs[2]["delta_c"]
        self.delta_c = np.reshape(delta_c, [2, 1])
        
        

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
     
    def project_velo_to_image(self, pts_3d_velo):
        '''
        Input: 3D points in Velodyne coord [nx3]
        Output: 3D points in Camera coord [nx3]
        '''
        print(pts_3d_velo)
        self.V2C = np.column_stack((self.R, self.T)) # RT

        # NORMAL TECHNIQUE
        R0_homo = np.vstack([self.R0, [0, 0, 0]])
        R0_homo_2 = np.column_stack([R0_homo, [0, 0, 0, 1]])
        p_r0 = np.dot(self.P, R0_homo_2) #PxR0
        p_r0_rt =  np.dot(p_r0, np.vstack((self.V2C, [0, 0, 0, 1]))) #PxROxRT
        pts_3d_homo = np.column_stack([pts_3d_velo, np.ones((1,1))])
        p_r0_rt_x = np.dot(p_r0_rt, np.transpose(pts_3d_homo))#PxROxRTxX
        pts_2d = np.transpose(p_r0_rt_x)
        
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]

        return pts_2d[:, 0:2]

    def get_lidar_in_image_fov(self,pc_velo, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0, mask=False):
        """ Filter lidar points, keep those in image FOV """
        pts_2d = self.project_velo_to_image(pc_velo)
        fov_inds = (
            (pts_2d[:, 0] < xmax)
            & (pts_2d[:, 0] >= xmin)
            & (pts_2d[:, 1] < ymax)
            & (pts_2d[:, 1] >= ymin)
        )
        fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance) # We don't want things that are closer to the clip distance (2m)

        imgfov_pc_velo = pc_velo[fov_inds, :]
        pts_2d = np.int32(pts_2d)
        
        try:
            for item in pts_2d[fov_inds, :]:
                if item not in mask[0]:
                    pts_2d.remove(item)
                    print('a')
        except:
            pass
        if return_more:
            return imgfov_pc_velo, pts_2d, fov_inds
        else:
            return imgfov_pc_velo
        

    def show_lidar_on_image(self, pc_velo, img, debug="False"):
        """ Project LiDAR points to image """
        imgfov_pc_velo, pts_2d, fov_inds = self.get_lidar_in_image_fov(
            pc_velo, 0, 0, img.shape[1], img.shape[0], True
        )
        if (debug==True):
            print("3D PC Velo "+ str(imgfov_pc_velo)) # The 3D point Cloud Coordinates
            print("2D PIXEL: " + str(pts_2d)) # The 2D Pixels
            print("FOV : "+str(fov_inds)) # Whether the Pixel is in the image or not
        self.imgfov_pts_2d = pts_2d[fov_inds, :]
        '''
        #homogeneous = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
        homogeneous = self.cart2hom(imgfov_pc_velo)
        transposed_RT = np.dot(homogeneous, np.transpose(self.V2C))
        dotted_RO = np.transpose(np.dot(self.R0, np.transpose(transposed_RT)))
        self.imgfov_pc_rect = dotted_RO
        
        if debug==True:
            print("FOV PC Rect "+ str(self.imgfov_pc_rect))
        '''
        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        self.imgfov_pc_velo = imgfov_pc_velo
        
        for i in range(self.imgfov_pts_2d.shape[0]):
            #depth = self.imgfov_pc_rect[i,2]
            #print(depth)
            depth = imgfov_pc_velo[i,0]
            #print(depth)
            color = cmap[int(510.0 / depth), :]
            cv2.circle(
                img,(int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))),2,
                color=tuple(color),
                thickness=-1,
            )

        return img
    

    def rectContains(rect,pt, w, h, shrink_factor = 0):       
        x1 = int(rect[0]*w - rect[2]*w*0.5*(1-shrink_factor)) # center_x - width /2 * shrink_factor
        y1 = int(rect[1]*h-rect[3]*h*0.5*(1-shrink_factor)) # center_y - height /2 * shrink_factor
        x2 = int(rect[0]*w + rect[2]*w*0.5*(1-shrink_factor)) # center_x + width/2 * shrink_factor
        y2 = int(rect[1]*h+rect[3]*h*0.5*(1-shrink_factor)) # center_y + height/2 * shrink_factor
        
        return x1 < pt[0]


    def filter_outliers(distances):
        inliers = []
        mu  = statistics.mean(distances)
        std = statistics.stdev(distances)
        for x in distances:
            if abs(x-mu) < std:
                # This is an INLIER
                inliers.append(x)
        return inliers

    def get_best_distance(distances, technique="closest"):
        if technique == "closest":
            return min(distances)
        elif technique =="average":
            return statistics.mean(distances)
        elif technique == "random":
            return random.choice(distances)
        else:
            return statistics.median(sorted(distances))
    
    def lidar_camera_fusion(self, pred_bboxes, image):
        img_bis = image.copy()

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        distances = []
        for box in pred_bboxes:
            distances = []
            for i in range(self.imgfov_pts_2d.shape[0]):
                #depth = self.imgfov_pc_rect[i, 2]
                depth = self.imgfov_pc_velo[i,0]
                if (self.rectContains(box, self.imgfov_pts_2d[i], image.shape[1], image.shape[0], shrink_factor=0)==True):
                    distances.append(depth)

                    color = cmap[int(510.0 / depth), :]
                    cv2.circle(img_bis,(int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))),2,color=tuple(color),thickness=-1,)
            h, w, _ = img_bis.shape
            if (len(distances)>2):
                distances = self.filter_outliers(distances)
                best_distance = self.get_best_distance(distances, technique="average")
                cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0]*w),int(box[1]*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3, cv2.LINE_AA)    
            distances_to_keep = []
        
        return img_bis, distances
