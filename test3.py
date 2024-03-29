#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""


# Fixing the OpenBLAS threading issue
import os
# For OpenBLAS
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
# For Intel MKL
# PLEASE REMOVE NAN VALUE FOR INTEL MKL!
# os.environ["MKL_NUM_THREADS"] = "1" 

#!/usr/bin/env python
import numpy as np
import cv2
import scipy
from scipy import ndimage
from scipy.signal import savgol_filter

import rospy
import ros_numpy # Convert ROS messages into Numpy array
from sensor_msgs.msg import PointCloud2

from sklearn.linear_model import LinearRegression, Ridge
from scipy.spatial.transform import Rotation as R

# import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# ROS operations

def pts_np_to_pclmsg(pts, frame="laser", intensity=255):
    d = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), 
                  ('intensity', np.float32)])
    struc_arr = np.zeros(pts.shape[0:-1], d)
    try:
        struc_arr['x'] = pts[:,:,0]
        struc_arr['y'] = pts[:,:,1]
        struc_arr['z'] = pts[:,:,2]
        struc_arr['intensity'] = intensity
    except IndexError: # For unorganized array
        struc_arr['x'] = pts[:,0]
        struc_arr['y'] = pts[:,1]
        struc_arr['z'] = pts[:,2]
        struc_arr['intensity'] = intensity
    
    msg = ros_numpy.msgify(PointCloud2, struc_arr, frame_id=frame)

    return msg

#------------------------------------------------------------------------------
    
def callback_ptcloud(ptcloud_data):
    data = ros_numpy.numpify(ptcloud_data)
    pts  = np.array([data['x'], data['y'], data['z']])
    # intensity = np.array(data['intensity'])
    pts  = np.moveaxis(pts, 0, 2)
    
    # !!! We just leave numpy and OpenBLAS to handle nan themselves
    # Handle nan. Since tan=(Opposite Side/Adjacent side), and small angle would be deprecated anyway
    # pts[:, :, 2] = np.nan_to_num(pts[:, :, 2],       0) # z, or opposite side
    # pts[:, :, 1] = np.nan_to_num(pts[:, :, 1], 9999999) # y, a component of adjacent side
    # pts[:, :, 0] = np.nan_to_num(pts[:, :, 0], 9999999) # x, a component of adjacent side
    # mask = np.isnan(pts[:, :, 2])
    # pts[mask, 2] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pts[~mask, 2])

    # Remove NaN value
    # NOTE: MUST REMOVE NAN VALUE FOR INTEL MKL LIBRARY OR DIVERGENCE ERROR!
    # NOT RECOMMEND FOR THE LOST OF FAR RANGE DETAILS
    # pts = pts[~np.isnan(pts).any(axis=(2,1))]

    # Filter (smoothing) z value by column (or multiple ring in the same azimuth)
    pts[:, :, 2] = savgol_filter(pts[:, :, 2], window_length=5, polyorder=3, axis=1)
    
    z0 = pts[:, 0:31, 2]
    z1 = pts[:, 1:32, 2]

    r  = np.linalg.norm(pts[:, :, 0:2], axis=2)
    r0 = r[:, 0:31]
    r1 = r[:, 1:32]

    angle = np.abs(np.arctan2((z1-z0),(r1-r0)))
    angle[angle>1.57079633] = 3.141592654 - angle[angle>1.57079633]
    angle = np.nan_to_num(angle, nan=0)

    # FOR TESTING ONLY! MAY CONFLICT WITH ROS CALLBACK
    # plt.hist(degrees_angle)
    # plt.show()

    mask_2d = angle>min_angle
    mask_3d = np.repeat(mask_2d.reshape(-1, 31, 1), 3, axis=2)

    # Ground removed
    # ground_removed = np.ma.where(mask_3d, pts[:, 1:32, :], np.nan)
    # intensity = data['intensity'][:, 1:32]
    intensity = angle
    # msg = pts_np_to_pclmsg(ground_removed, intensity=intensity)
    # ground_removed_publisher.publish(msg)

    # Ground seperated
    ground = np.ma.where(np.bitwise_not(mask_3d), pts[:, 1:32, :], np.nan)
    msg = pts_np_to_pclmsg(ground, intensity=intensity)
    ground_seperated_publisher.publish(msg)

    # Assumed ground takes account mostly of nearest ring. Perform Plane regression
    first_ring = pts[:, 0, :]
    first_ring = first_ring[~np.isnan(first_ring).any(axis=1)]    

    X = np.vstack((first_ring[:,0], first_ring[:,1])).reshape(-1, 2)
    y = first_ring[:,2]

    reg = Ridge().fit(X, y)
    vec = np.array([reg.coef_[0], reg.coef_[1], 0])
    r = R.from_rotvec(vec)
    r_mat = r.as_matrix()
    r_mat = np.linalg.inv(r_mat)

    pts_flatten = pts.reshape(-1, 3) 
    pts_transformed = np.matmul(pts, r_mat)
    
    h = reg.intercept_
    HEIGHT_MIN = h-0.5
    HEIGHT_MAX = h+0.3
    print(HEIGHT_MIN, HEIGHT_MAX)
    mask_2d_min = (pts_transformed[:, :, 2]) > HEIGHT_MIN
    mask_2d_max = (pts_transformed[:, :, 2]) < HEIGHT_MAX
    mask_2d = np.bitwise_and(mask_2d_min, mask_2d_min)
    mask_3d = np.repeat(mask_2d.reshape(-1, 32, 1), 3, axis=2)
    ground = pts_transformed[mask_2d]

    msg = pts_np_to_pclmsg(ground)
    ground_seperated2_publisher.publish(msg)
    

#------------------------------------------------------------------------------
OBSTACLE_SLOPE_ANGLE_RANGE = 1.22173048  # rad
min_angle = 1.57079633 - OBSTACLE_SLOPE_ANGLE_RANGE

if (__name__ == "__main__"):
    try:
        rospy.init_node("test_lidar", anonymous=True)
        rospy.Subscriber("/velodyne_points", PointCloud2, callback_ptcloud, \
                         queue_size=1)
        ground_seperated_publisher = rospy.Publisher('/pointcloud/ground_seperated_publisher', PointCloud2,\
                                        queue_size=1)
        ground_seperated2_publisher = rospy.Publisher('/pointcloud/ground_seperated2_publisher', PointCloud2, \
                                           queue_size=1)
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()