#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 00:56:26 2022

@author: tranquockhue
"""

# Fixing the OpenBLAS threading issue
import os
from cv2 import THRESH_OTSU
os.environ["OPENBLAS_NUM_THREADS"] = "1" 

#!/usr/bin/env python
import numpy as np
import cv2
import scipy
from scipy import ndimage
from scipy.signal import savgol_filter

import rospy
import ros_numpy # Convert ROS messages into Numpy array
from sensor_msgs.msg import PointCloud2

# import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# ROS operations

def pts_np_to_pclmsg(pts, frame="laser", intensity=255):
    d = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), 
                  ('intensity', np.float32)])
    struc_arr = np.zeros(pts.shape[0:-1], d)
    struc_arr['x'] = pts[:,:,0]
    struc_arr['y'] = pts[:,:,1]
    struc_arr['z'] = pts[:,:,2]
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

    # Filter (smoothing) z value by column (or multiple ring in the same azimuth)
    pts[:, :, 2] = savgol_filter(pts[:, :, 2], window_length=5, polyorder=3, axis=1)
    
    z0 = pts[:, 0:31, 2]
    z1 = pts[:, 1:32, 2]

    r  = np.linalg.norm(pts[:, :, 0:2], axis=2)
    r0 = r[:, 0:31]
    r1 = r[:, 1:32]

    angle = np.arctan2((z1-z0),(r1-r0))

    # FOR TESTING ONLY! MAY CONFLICT WITH ROS CALLBACK
    # plt.hist(degrees_angle)
    # plt.show()

    mask_2d = np.bitwise_and(angle>min_angle, angle<max_angle)
    mask_3d = np.repeat(mask_2d.reshape(-1, 31, 1), 3, axis=2)

    # Ground removed
    ground_removed = np.ma.where(mask_3d, pts[:, 1:32, :], np.nan)
    # intensity = data['intensity'][:, 1:32]
    intensity = angle
    msg = pts_np_to_pclmsg(ground_removed, intensity=intensity)
    ground_removed_publisher.publish(msg)

    # Ground seperated
    ground = np.ma.where(np.bitwise_not(mask_3d), pts[:, 1:32, :], np.nan)
    r = np.ma.where(np.bitwise_not(mask_3d), r[:, 1:32, :], np.nan)
    
    # --------------------------------
    # tEsTiNG z0nE

    # test_intensity = np.gradient(ground[:,:,2], axis=0) # Intra-ring
    pts[:, :, 2] = savgol_filter(pts[:, :, 2], window_length=7, polyorder=5, axis=0)
    
    z0 = pts[1:, :, 2]
    z1 = pts[:-1, :, 2]

    r  = np.linalg.norm(pts[:, :, 0:2], axis=1)
    r0 = r[1:, :]
    r1 = r[:,-1, :]

    angle = np.arctan2((z1-z0),(r1-r0))
    np.clip(angle, a_min=0.01, a_max=0.5)
    test_intensity = angle
    # --------------------------------

    msg = pts_np_to_pclmsg(ground, intensity=test_intensity)
    ground_seperated_publisher.publish(msg)

#------------------------------------------------------------------------------
OBSTACLE_SLOPE_ANGLE_RANGE = 1.91986218 # rad
min_angle = 1.57079633 - OBSTACLE_SLOPE_ANGLE_RANGE/2
max_angle = 1.57079633 + OBSTACLE_SLOPE_ANGLE_RANGE/2

if (__name__ == "__main__"):
    try:
        rospy.init_node("test_lidar", anonymous=True)
        rospy.Subscriber("/velodyne_points", PointCloud2, callback_ptcloud, \
                         queue_size=1)
        ground_removed_publisher = rospy.Publisher('/pointcloud/ground_removed', PointCloud2,\
                                        queue_size=1)
        ground_seperated_publisher = rospy.Publisher('/pointcloud/ground_seperated', PointCloud2, \
                                           queue_size=1)
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()