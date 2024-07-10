import numpy as np


class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = []
        for file in calib_file:
            calibs.append(self.read_calib_file(file))
            # ['calib/calib_cam_to_cam.txt', 'calib/calib_imu_to_velo.txt', 'calib/calib_velo_to_cam.txt']
        
        'CALIB CAM TO CAM'
        P = calibs[0]["P_rect_02"]
        self.P = np.reshape(P, [3, 4])
        # Rotation from reference camera coord to rect camera coord
        R0 = calibs[0]["R_rect_02"]
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
     
    def velo_to_camera(self, pts_3d_vel):
        '''
        Input: 3D points in Velodyne coord [nx3]
        Output: 3D points in Camera coord [nx3]
        '''
        ## Camera parameters
        focal = 4e-3    #focal distance of the camera
        dx = 1          #
        dy = 1          #
        u0 = 0          # origin pixel coordinate
        v0 = 0          # origin pixel coordinate

        # FASE 1
        matrix = np.column_stack((self.R, self.T)) # RT
        rt_r0_1 = np.vstack((matrix, [[0., 0., 0., 1.]])) # [[RT][R01]]

        pts_3d_homo = np.column_stack((pts_3d_vel,np.ones(pts_3d_vel.shape[0]))) #

        result = np.dot(rt_r0_1, np.transpose(pts_3d_homo))
        pts_2d = np.transpose(result)

        # FASE 2
        focal_matrix = [[focal, 0, 0, 0],
                        [0, focal, 0, 0],
                        [0, 0, 1, 0]]
        tr_proj = np.dot(focal_matrix, result)

        Zc = result[2]
        x_y_1 = np.zeros([tr_proj.shape[1], 3])

        for lineIndex, line in enumerate(x_y_1):
            x_y_1[lineIndex][:] = 1/Zc[lineIndex] * np.transpose(tr_proj)[lineIndex][:]

        # FASE 3
        img_2_pixel = [[1/dx, 0, u0],
                       [0, 1/dy, v0],
                       [0, 0, 1]]
        R0_homo = np.vstack([self.R0, [0, 0, 0]])
        R0_homo_2 = np.column_stack([R0_homo, [0, 0, 0, 1]])
        img_2_pixel = np.dot(self.P, R0_homo_2)

        u_v_1 = np.dot(img_2_pixel, np.transpose(pts_2d))

        u_v_1[:, 0] /= u_v_1[:, 2]
        u_v_1[:, 1] /= u_v_1[:, 2]
        print(u_v_1)


    def project_velo_to_image(self, pts_3d_vel):
        pass

    def transf_image_to_pixel(self, pts_3d_vel):
        pass