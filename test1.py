#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 00:56:26 2022

@author: tranquockhue
"""
#!/usr/bin/env python
import numpy as np
import cv2
import scipy
from scipy import ndimage

import rospy
import ros_numpy # Convert ROS messages into Numpy array

from sensor_msgs.msg import PointCloud2

# Dimensions: first channel is lidar firing, second channel is physical axes!

#------------------------------------------------------------------------------
# ROS operations

def pts_np_to_pclmsg(pts, intensity, frame):
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
# Processing operations

def convert_cartesian2spherical(cartesian_pts):
    r = np.linalg.norm(a, axis=2)
    alt = np.round(np.arcsin((a[:,:,2]/r)), 5) # z/r
    azi = np.round(np.arctan2(a[:,:,1], a[:,:,0]), 5) # y/x
    spherical = np.dstack((r, alt, azi))

    return spherical
   
def organizing(pts):
    return None

#------------------------------------------------------------------------------
    
def callback_ptcloud(ptcloud_data):
    data = ros_numpy.numpify(ptcloud_data)
    pts  = np.array([data['x'], data['y'], data['z']])
    pts  = np.moveaxis(pts, 0, 2)
    
    global a
    a = pts

    global spherical
    spherical = convert_cartesian2spherical(pts)

    LIDAR_HORIZONTAL_RESOLUTION = 0.0035
    global horizontal_bins
    horizontal_bins = np.unique(np.divmod(spherical[:,:,2], LIDAR_HORIZONTAL_RESOLUTION)[0])
    horizontal_bins = horizontal_bins[~np.isnan(horizontal_bins)] * LIDAR_HORIZONTAL_RESOLUTION 
    horizontal_bins = horizontal_bins - np.min(horizontal_bins)
    print(horizontal_bins)

    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

if (__name__ == "__main__"):
    try:
        rospy.init_node("test_lidar", anonymous=True)
        rospy.Subscriber("/velodyne_points", PointCloud2, callback_ptcloud, \
                         queue_size=1)
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()