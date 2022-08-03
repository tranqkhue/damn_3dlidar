import cv2
import numpy as np
import ros_numpy # Convert ROS messages into Numpy array

import tf
import rospy
from nav_msgs.msg import OccupancyGrid


def occ_grid_viz(occ_grid):
    occ_grid = occ_grid.astype('int16')
    # Unknown:= -1, Free:= 0, Occupied:=100
    occ_grid[occ_grid == 0]   = 255
    occ_grid[occ_grid == 100] = 0
    occ_grid[occ_grid == -1]  = 191
    occ_grid = occ_grid.astype('uint8')

    cv2.imshow('OccupancyGridViz', cv2.resize(occ_grid, (0,0), fx=0.4, fy=0.4))
    #cv2.imshow('OccupancyGridViz', occ_grid)
    cv2.waitKey(1)

# Modify: Get completed gmapping map once, then go to processing loop

def callback_map(map_data):
    global trans, rot
    print(trans, rot)

    global occ_grid
    # Unknown:= -1, Free:= 0, Occupied:=100
    occ_grid = ros_numpy.numpify(map_data).data
    map_x = map_data.info.origin.position.x
    map_y = map_data.info.origin.position.y
    map_res = map_data.info.resolution

    occ_grid_viz(occ_grid)
    print('thang gay')


if (__name__ == "__main__"):
    rospy.init_node("test_lidar", anonymous=True)
    tf_listener = tf.TransformListener()
    rospy.Subscriber("/map", OccupancyGrid, callback_map, \
                     queue_size=1)
    while not rospy.is_shutdown(): 
        try:
            (trans,rot) = tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    cv2.destroyAllWindows()