#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 00:56:26 2022

@author: tranquockhue
"""
#!/usr/bin/env python
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import PointField as PointField
from std_msgs.msg import Header as Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

# Dimensions: first channel is lidar firing, second channel is physical axes!

#------------------------------------------------------------------------------

def pts_np_to_pclmsg(pts, frame):
    header = Header()
    header.frame_id=frame
    msg = pc2.create_cloud_xyz32(header, pts)
    return msg

#------------------------------------------------------------------------------

def convert_cartesian2spherical(cartesian_pts):
    x = cartesian_pts[:, 0]
    y = cartesian_pts[:, 1]
    z = cartesian_pts[:, 2]
    
    r = np.sqrt(x**2 + y**2+ z**2)
    altitude = np.round(np.arcsin(z/r), 5)
    azimuth  = np.round(np.arctan2(y,x), 5) 
    spherical = np.vstack((r, altitude, azimuth)).transpose()
    
    return spherical

def equirectangular_viz(organized_xyz):
    nonan = np.nan_to_num(organized_xyz)
    
    # Equirectangular with value distance d = sqrt(x**2 + y**2 + z**2)
    distanced = np.linalg.norm(nonan, axis=2)
    
    # Normalize to 0->255 for visualization
    MAX_RANGE = 40
    clipped = np.clip(distanced, 0, MAX_RANGE)
    normed = (clipped - 0)/(MAX_RANGE) * 256.0
    #normed = (distanced - 0)/(30 - 0) * 255.0
    normed = normed.astype(np.uint8)
    normed = normed.transpose()
    flipped = np.flip(normed, 0)
    
    viz = cv2.resize(flipped, (0,0), fx=0.3, fy=3.5)
    vizC = cv2.applyColorMap(viz, cv2.COLORMAP_JET)
    cv2.imshow('viz', vizC)
    cv2.waitKey(1)
    
def sorting(pts):
    # Assume the points are already organized by velodyne package
    LIDAR_RINGS_NUM=32 # For Velodyne HDL-32E Lidar
    
    # Equirectangular with value = [x,y,z]
    equirectangular = np.reshape(pts, (-1, LIDAR_RINGS_NUM, 3))
    equirectangular_viz(equirectangular)
    
    return equirectangular

#------------------------------------------------------------------------------
    
def callback_ptcloud(ptcloud_data):
    xd = list(pc2.read_points(ptcloud_data, skip_nans=False))
    pc = np.asarray(xd)
    # [x,y,z]
    pts = pc[:,:3]
    sorting(pts)

#------------------------------------------------------------------------------

if (__name__ == "__main__"):
    rospy.init_node("test_lidar", anonymous=True)
    pts_publisher = rospy.Publisher('/test_points', PointCloud2,\
                                    queue_size=1)
    rospy.Subscriber("/velodyne_points", PointCloud2, callback_ptcloud, \
                     queue_size=1)
    rospy.spin()
cv2.destroyAllWindows()