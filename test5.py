#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
from std_msgs.msg import Header

from sklearn.linear_model import LinearRegression, Ridge
from scipy.spatial.transform import Rotation as R

# import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def equirectangular_viz(organized_xyz):
    # Visualize withvalue is the radius (distance) from Lidar to obstacles!
    nonan = np.nan_to_num(organized_xyz, nan=9999)
    
    # Equirectangular with value distance d = sqrt(x**2 + y**2 + z**2)
    distanced = np.linalg.norm(nonan[:,:,0:2], axis=2)
    
    # Normalize to 0->255 for visualization
    MAX_RANGE = 15
    clipped = np.clip(distanced, 0, MAX_RANGE)
    normed = (clipped - 0)/(MAX_RANGE) * 256.0
    #normed = (distanced - 0)/(30 - 0) * 255.0
    normed = normed.astype(np.uint8)
    normed = normed.transpose()
    flipped = np.flip(normed, 0)

    viz = cv2.resize(flipped, (0,0), fx=0.4, fy=3.5)
    vizC = cv2.applyColorMap(viz, cv2.COLORMAP_JET)
    vizC = cv2.rotate(vizC, rotateCode=cv2.ROTATE_180)
    cv2.imshow('viz', vizC)
    cv2.waitKey(1)


#------------------------------------------------------------------------------
# ROS operations

def pts_np_to_pclmsg(pts, header=Header(), frame="laser", intensity=255):
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
    
    msg = ros_numpy.msgify(PointCloud2, struc_arr)
    msg.header = header
    msg.header.frame_id = frame

    return msg

#------------------------------------------------------------------------------
# Main runtime callback

def callback_ptcloud(ptcloud_data):
    
    #------------------------------------------------------------------------------
    # Receive data from ROS 

    data = ros_numpy.numpify(ptcloud_data)
    header = ptcloud_data.header
    pts  = np.array([data['x'], data['y'], data['z']])
    pts  = np.moveaxis(pts, 0, 2)
    # intensity = np.array(data['intensity'])
    
    # equirectangular_viz(pts)

    #------------------------------------------------------------------------------
    # Filter top_ring

    # top_ring_only = pts[:, (pts.shape[1]-3):(pts.shape[1]), :]
    top_ring_only = pts[:, 0:2, :]

    mask_2d = top_ring_only[:, :, 2] > -3
    mask_3d = np.repeat(mask_2d.reshape(-1, 2, 1), 3, axis=2)
    filtered_top_by_height = np.ma.where(mask_3d, top_ring_only, np.nan)

    #------------------------------------------------------------------------------
    # Filter top_ring with angle

    z0 = filtered_top_by_height[:, 0, 2]
    z1 = filtered_top_by_height[:, 1, 2]

    r  = np.linalg.norm(filtered_top_by_height[:, :, 0:2], axis=2)
    r0 = r[:, 0]
    r1 = r[:, 1]

    angle = np.abs(np.arctan2((z1-z0),(r1-r0)))
    angle[angle>1.57079633] = 3.141592654 - angle[angle>1.57079633]
    angle = np.nan_to_num(angle, nan=0.69)

    mask_2d = np.bitwise_or(angle > 1.3962634, angle < 0.174532925)
    mask_3d = np.repeat(mask_2d.reshape(-1, 1, 1), 3, axis=2)
    filtered_top_by_angle = np.ma.where(mask_3d, filtered_top_by_height[:, :-1, :], np.nan)
    
    top_ring_msg = pts_np_to_pclmsg(filtered_top_by_angle)
    top_ring_pub.publish(top_ring_msg)

    global a
    a = filtered_top_by_angle

    #------------------------------------------------------------------------------
    # Filter blind spot

    not_blind_azi = np.isnan(filtered_top_by_angle[:,:,1])
    blind_azi = np.bitwise_not(not_blind_azi)
    mask_3d = np.repeat(blind_azi.reshape(-1, 1, 1), 32, axis=1)
    mask_3d = np.repeat(mask_3d.reshape(-1, 32, 1),   3, axis=2)
    filtered_blind_spot = np.ma.where(mask_3d, pts, np.nan)

    #------------------------------------------------------------------------------
    # Publish to ROS

    blind_spot_filtered_msg = pts_np_to_pclmsg(filtered_blind_spot)
    blind_spot_filtered_pub.publish(blind_spot_filtered_msg)
    
#------------------------------------------------------------------------------

if (__name__ == "__main__"):
    try:
        rospy.init_node("test_lidar", anonymous=True)
        rospy.Subscriber("/velodyne_points", PointCloud2, callback_ptcloud, \
                         queue_size=1)
        top_ring_pub = rospy.Publisher('/top_ring', PointCloud2,\
                                       queue_size=1)
        blind_spot_filtered_pub = rospy.Publisher('/blind_spot_filtered_pub', PointCloud2,\
                                                   queue_size=1)
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()