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

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Dimensions: first channel is lidar firing, second channel is physical axes!

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

    a = savgol_filter(a, window_length=5, polyorder=4, axis=1)

    b = a[:, 0:31, :]
    c = a[:, 1:32, :]

    test_angle = np.arctan2((c[:,:,2]-b[:,:,2]),(c[:,:,0]-b[:,:,0]))
    global degrees_angle 
    degrees_angle = np.abs(np.degrees(test_angle))

    mask_2d = np.bitwise_and(150>degrees_angle, 30<degrees_angle)
    #mask_2d = 130>degrees_angle>50

    global mask_3d
    mask_3d = np.repeat(mask_2d.reshape(-1, 31, 1), 3, axis=2)
    # field3d_mask = np.broadcast_to(degrees_angle > 45, b.shape)
    global out
    out = np.where(mask_3d, pts[:, 1:32, :], np.nan)

    intensity = data['intensity'][:, 1:32]
    msg = pts_np_to_pclmsg(out, intensity=intensity)
    gnd_publisher.publish(msg)

    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

if (__name__ == "__main__"):
    try:
        rospy.init_node("test_lidar", anonymous=True)
        rospy.Subscriber("/velodyne_points", PointCloud2, callback_ptcloud, \
                         queue_size=1)
        gnd_publisher = rospy.Publisher('/gnd_removed', PointCloud2,\
                                        queue_size=1)
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()