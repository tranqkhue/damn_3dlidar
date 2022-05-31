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
# Visualization operations

def equirectangular_viz(organized_xyz):
    # Visualize withvalue is the radius (distance) from Lidar to obstacles!
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

#------------------------------------------------------------------------------
# Processing operations

def convert_cartesian2spherical(cartesian_pts):
    x = cartesian_pts[:, 0]
    y = cartesian_pts[:, 1]
    z = cartesian_pts[:, 2]
    
    r = np.sqrt(x**2 + y**2+ z**2)
    altitude = np.round(np.arcsin(z/r), 5)
    azimuth  = np.round(np.arctan2(y,x), 5) 
    spherical = np.vstack((r, altitude, azimuth)).transpose()
    
    return spherical
   
def organizing(pts):
    # Assume the points are already organized by velodyne package
    LIDAR_RINGS_NUM=32 # For Velodyne HDL-32E Lidar
    
    # Equirectangular with value = [x,y,z]
    equirectangular = np.reshape(pts, (-1, LIDAR_RINGS_NUM, 3))
    
    return equirectangular

def cal_z_grad(organized_xyz):
    nonan_organized_xyz = np.nan_to_num(organized_xyz)
    z_matrix = nonan_organized_xyz[:,:,2]
    # NOTE: Image frame is tranposed and flipped to the numpy pointcloud array!
    # Calculation of Sobely (0,1)
    sobelx = cv2.Sobel(z_matrix,cv2.CV_64F,dx=0,dy=1,ksize=5)
    # Calculation of Sobely (1,0)
    sobely = cv2.Sobel(z_matrix,cv2.CV_64F,dx=1,dy=0,ksize=5)
  
    # Return absolute gradient value
    sobelx = np.abs(sobelx)
    sobely = np.abs(sobely)
    
    return sobelx, sobely

#------------------------------------------------------------------------------
    
def callback_ptcloud(ptcloud_data):
    data = ros_numpy.numpify(ptcloud_data)
    pts  = np.array([data['x'], data['y'], data['z']])
    pts  = np.moveaxis(pts, 0, 2)
    
    pts_organized = organizing(pts)
    # equirectangular_viz(pts_organized)

    #--------------------------------------------------------------------------
    # Calculate gradient by delta_z

    sobel_intra_ring, sobel_inter_ring = cal_z_grad(pts_organized)
    
    # PLEASE MODIFY CLIP RANGE IF NEEDED!
    sobel_intra_ring = np.clip(sobel_intra_ring, a_min=0.0, a_max=0.1)
    msg = pts_np_to_pclmsg(pts_organized, sobel_intra_ring, "laser")
    pts_x_publisher.publish(msg)
    
    # PLEASE MODIFY CLIP RANGE IF NEEDED!
    sobel_inter_ring = np.clip(sobel_inter_ring, a_min=0.0, a_max=0.1)
    msg = pts_np_to_pclmsg(pts_organized, sobel_inter_ring, "laser")
    pts_y_publisher.publish(msg)
    #--------------------------------------------------------------------------


#------------------------------------------------------------------------------

if (__name__ == "__main__"):
    try:
        rospy.init_node("test_lidar", anonymous=True)
        pts_x_publisher = rospy.Publisher('/sobel_z_intra_ring', PointCloud2,\
                                        queue_size=1)
        pts_y_publisher = rospy.Publisher('/sobel_z_inter_ring', PointCloud2,\
                                        queue_size=1)
        rospy.Subscriber("/velodyne_points", PointCloud2, callback_ptcloud, \
                         queue_size=1)
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()