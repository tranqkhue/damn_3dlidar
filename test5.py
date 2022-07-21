#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""


# Fixing the OpenBLAS threading issue
import os
from re import A
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
last_height_z_ground = 0
def callback_ptcloud(ptcloud_data):
    data = ros_numpy.numpify(ptcloud_data)
    pts  = np.array([data['x'], data['y'], data['z']])
    # intensity = np.array(data['intensity'])
    pts  = np.moveaxis(pts, 0, 2)

    #------------------------------------------------------------------------------
    # Filter not ceiling points by absolute height
    
    mask_2d = (pts[:,:,2] > -0.5)
    mask_3d = np.repeat(mask_2d.reshape(-1, 32, 1), 3, axis=2)
    not_ceiling = np.where(mask_3d, pts[:, :, :], np.nan)
 
    #------------------------------------------------------------------------------
    # Find the lower ground (not ceiling)
    # By finding angle inter-rings

    # Output: Ground removed for future steps (as flat ground should not be an obstacle)

    not_ceiling[:, :, 2] = savgol_filter(not_ceiling[:, :, 2], window_length=5, polyorder=3, axis=1)
    z0 = not_ceiling[:, 0:31, 2]
    z1 = not_ceiling[:, 1:32, 2]

    r  = np.linalg.norm(not_ceiling[:, :, 0:2], axis=2)
    r0 = r[:, 0:31]
    r1 = r[:, 1:32]
    angle = np.abs(np.arctan2((z1-z0),(r1-r0)))
    angle[angle>1.57079633] = 3.141592654 - angle[angle>1.57079633]
    angle = np.nan_to_num(angle, nan=0)

    mask_2d = angle>min_angle
    mask_3d = np.repeat(mask_2d.reshape(-1, 31, 1), 3, axis=2)
    ground_filtered_angle = np.ma.where(np.bitwise_not(mask_3d), not_ceiling[:, 1:32, :], -1)

    # After finding the ground level rings, use histogram to find the mode of distribution
    ground_z = np.reshape(ground_filtered_angle[:,:,2], (-1, 1)) 
    ground_z = ground_z[~np.isnan(ground_z)]
    hist_z = np.histogram(ground_z, bins=30)
    # Filter value smaller than zero (assume lidar is up-side down)
    hist_z[0][np.where(hist_z[1]<0)[0]-1] = 9999
    # Find the most dominant height 
    height_z = hist_z[1][np.argmax(hist_z[0])]
    global last_height_z_ground
    if (height_z) > 0.5:
        last_height_z_ground = height_z
    else: 
        height_z = last_height_z_ground
    ground_removed_mask_2d = (pts[:,:,2] < height_z-0.3)
    mask_3d = np.repeat(ground_removed_mask_2d.reshape(-1, 32, 1), 3, axis=2)
    ground_removed = np.where(mask_3d, pts[:, :, :], np.nan)

    msg = pts_np_to_pclmsg(ground_removed)
    ground_removed_pub.publish(msg)

    #------------------------------------------------------------------------------
    # Get low level obstacle
    mask_2d = (pts[:,:,2] > -1.5)
    low_obstacle_mask = mask_2d
    mask_3d = np.repeat(mask_2d.reshape(-1, 32, 1), 3, axis=2)
    low_obstacle = np.where(mask_3d, ground_removed[:, :, :], np.nan)

    msg = pts_np_to_pclmsg(low_obstacle)
    low_obstacle_layer.publish(msg)

    #------------------------------------------------------------------------------
    # Find the area without covering roof

    mask_2d = (pts[:,:,2] < -1.5)
    mask_3d = np.repeat(mask_2d.reshape(-1, 32, 1), 3, axis=2)
    ceiling_height_filtered = np.where(mask_3d, ground_removed[:, :, :], np.nan)

    no_nan = ceiling_height_filtered
    no_nan[:,:,0] = np.nan_to_num(no_nan[:,:,0], nan=9999)
    no_nan[:,:,1] = np.nan_to_num(no_nan[:,:,1], nan=9999)
    # no_nan[:,:,2] = np.nan_to_num(no_nan[:,:,2], nan=9999)
    # z_grad_intraring = np.abs(np.gradient(no_nan[:,:,2], axis=0))
 
    r_no_nan = np.linalg.norm(no_nan[:,:,0:2], axis=2)
    grad_intraring = np.abs(np.gradient(r_no_nan, axis=0))
    mask_2d = (grad_intraring>999)
    ceiling_mask = mask_2d
    mask_3d = np.repeat(mask_2d.reshape(-1, 32, 1), 3, axis=2)
    ceiling_intersect = np.where(mask_3d, no_nan[:, :, :], np.nan)

    msg = pts_np_to_pclmsg(ceiling_intersect)
    high_obstacle_layer.publish(msg)

    #------------------------------------------------------------------------------
    combined_mask = np.bitwise_or(low_obstacle_mask, ceiling_mask)
    mask_3d = np.repeat(combined_mask.reshape(-1, 32, 1), 3, axis=2)
    combined = np.where(mask_3d, ground_removed, np.nan)

    msg = pts_np_to_pclmsg(combined)
    combined_obstacle_layer.publish(msg)
    
#------------------------------------------------------------------------------
OBSTACLE_SLOPE_ANGLE_RANGE = 1.45  # rad
min_angle = 1.57079633 - OBSTACLE_SLOPE_ANGLE_RANGE

if (__name__ == "__main__"):
    try:
        rospy.init_node("test_lidar", anonymous=True)
        rospy.Subscriber("/velodyne_points", PointCloud2, callback_ptcloud, \
                         queue_size=1)
        ground_removed_pub = rospy.Publisher('/ground_removed', PointCloud2,\
                                             queue_size=1)
        low_obstacle_layer = rospy.Publisher('/low_obstacle', PointCloud2, \
                                             queue_size=1)
        high_obstacle_layer = rospy.Publisher('/ceiling_obstacle', PointCloud2, \
                                              queue_size=1)
        combined_obstacle_layer = rospy.Publisher('/combined_obstacle', PointCloud2, \
                                                  queue_size=1)
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()