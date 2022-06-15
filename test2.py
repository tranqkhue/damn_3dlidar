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
from scipy.signal import savgol_filter

import rospy
import ros_numpy # Convert ROS messages into Numpy array
from sensor_msgs.msg import PointCloud2

import matplotlib.pyplot as plt

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
    
def callback_ptcloud(ptcloud_data):
    data = ros_numpy.numpify(ptcloud_data)
    global pts
    pts  = np.array([data['x'], data['y'], data['z']])
    # intensity = np.array(data['intensity'])
    pts  = np.moveaxis(pts, 0, 2)
    
    # Filter (smoothing) z value by column (or multiple ring in the same azimuth)
    pts[:, :, 2] = savgol_filter(pts[:, :, 2], window_length=5, polyorder=3, axis=1)
    
    z0 = pts[:, 0:31, 2]
    z1 = pts[:, 1:32, 2]

    r0 = np.linalg.norm(pts[:, 0:31, 0:2], axis=2)
    r1 = np.linalg.norm(pts[:, 1:32, 0:2], axis=2)

    angle = np.arctan2((z1-z0),(r1-r0))
    degrees_angle = np.abs(np.degrees(angle))

    # FOR TESTING ONLY! MAY CONFLICT WITH ROS CALLBACK
    # plt.hist(degrees_angle)
    # plt.show()

    ANGLE_RANGE = 120
    mask_2d = np.bitwise_and(degrees_angle>(90-ANGLE_RANGE/2), \
                             degrees_angle<(90+ANGLE_RANGE/2))

    mask_3d = np.repeat(mask_2d.reshape(-1, 31, 1), 3, axis=2)
    ground_removed = np.ma.where(mask_3d, pts[:, 1:32, :], np.nan)

    # intensity = data['intensity'][:, 1:32]
    intensity = degrees_angle
    msg = pts_np_to_pclmsg(ground_removed, intensity=intensity)
    gnd_publisher.publish(msg)

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