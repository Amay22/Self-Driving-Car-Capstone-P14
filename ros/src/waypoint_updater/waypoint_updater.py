#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight
from std_msgs.msg import Int32
import numpy as np

import math
from enum import Enum

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 80 # Number of waypoints we will publish. You can change this number
KMPH2MPS = 1000. / (60. * 60.)   # 0.277778

DIST_STOP_TL = 6.0

CAR_STATE_STOP = 0
CAR_STATE_SLOWDOWN = 1
CAR_STATE_MAX_SPEED = 2
CAR_STATE_FULL_THROTTLE = 3

class CarState(Enum):
    STOP = 0
    SLOW_DOWN = 1
    MAX_SPEED = 2

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.current_velocity = None
        # subscribe to the velocity as well
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb, queue_size=1)

        # Add other member variables you need below
        # flag to check if waypoints were retrieved
        self.flag_waypoints_retrieved = False
        # Store waypoints
        self.base_waypoints = None
        # allows us to do a quick check to see if there are any waypoints
        self.num_waypoints = -1
        # Car related info
        self.pose = None
        self.pose_stamp = None
        self.car_x = None
        self.car_y = None
        self.car_theta = None
        # keep track of the next waypoint index
        self.next_wp_index = None
        # variables for traffic lights
        self.tl_waypoint = None
        self.last_tl_waypoint = None
        # define a car state as a small state machine to react on traffic lights
        self.car_state_tl = CAR_STATE_STOP

        self.loop()

    def loop(self):
        rate = rospy.Rate(10) # 10Hz
        while not rospy.is_shutdown():
            if self.flag_waypoints_retrieved and self.pose is not None:
                # unwrapping the vehicle pose
                self.car_x = self.pose.position.x
                self.car_y = self.pose.position.y
                # now obtaining orientation of the car (assuming rotation about z: [0;0;1])
                self.car_theta = 2 * np.arccos(self.pose.orientation.w)
                # Constraining the angle in [-pi, pi)
                if self.car_theta > np.pi:
                    self.car_theta = -(2 * np.pi - self.car_theta)
                self.next_wp_index = self.calc_next_waypoint()
                # publish the nodes
                self.publish_waypoints(self.next_wp_index)
            rate.sleep()

    # cross product of position-minus-neareset wp and direction vector from nearest to next.
    def calc_next_waypoint(self):
        # use brute force minimum distance
        nn_index = 0
        min_dist = float('inf')
        min_dist_map_x = 0
        min_dist_map_y = 0
        for idx, waypoint  in enumerate(self.base_waypoints):
            map_x = waypoint.pose.pose.position.x
            map_y = waypoint.pose.pose.position.y
            dist = (self.car_x - map_x) ** 2 + (self.car_y - map_y) ** 2
            if dist < min_dist:
                min_dist = dist
                min_dist_map_x = map_x
                min_dist_map_y = map_y
                nn_index = idx
        # now this node maybe 'behind ' or 'ahead' of the car
        # with repsect to its ****current heading*****
        # So we need to take cases
        next_wp_index = ( nn_index + 1 ) % len(self.base_waypoints)
        # now the difference vector of the car's position and the direction
        # vector v from the nearest waypoint to the next
        vx = self.base_waypoints[next_wp_index].pose.pose.position.x - min_dist_map_x
        vy = self.base_waypoints[next_wp_index].pose.pose.position.y - min_dist_map_y

        norm_v = np.sqrt( vx*vx + vy*vy )

        vx /= norm_v
        vy /= norm_v
        # now the difference : car position - nearest wp
        dx = self.car_x - min_dist_map_x
        dy = self.car_y - min_dist_map_y
        # Get the dot product of d and v
        d_dot_v = vx * dx + vy * dy
        if d_dot_v >= 0:
            return next_wp_index
        return nn_index

    # This function publishes the next waypoints
    # For now it sets velocity to the same value...
    def publish_waypoints(self, next_wp_index):

        msg = Lane()
        msg.waypoints = []
        index = next_wp_index

        current_velocity = self.current_velocity.linear.x if self.current_velocity is not None else 0.0

        for i in range(LOOKAHEAD_WPS):
            wp = Waypoint()
            wp.pose.pose.position.x = self.base_waypoints[index].pose.pose.position.x
            wp.pose.pose.position.y = self.base_waypoints[index].pose.pose.position.y
            # get maximum allowed speed from base waypoint
            max_speed = self.get_waypoint_velocity(self.base_waypoints[index])

            if self.car_state_tl == CAR_STATE_FULL_THROTTLE:
                wp.twist.twist.linear.x = max_speed * 2
            elif self.car_state_tl == CAR_STATE_MAX_SPEED:
                wp.twist.twist.linear.x = max_speed
            elif self.tl_waypoint is not None:
                dist_tl = self.distance(self.base_waypoints, index, self.tl_waypoint)
                if self.car_state_tl == CAR_STATE_STOP:
                    wp.twist.twist.linear.x = min(max(0.0, 0.2*(dist_tl-DIST_STOP_TL)), max_speed)
                    # stop vehicle if vehicle is almost standing and target speed is very low for waypoint
                    if current_velocity < 0.25 and wp.twist.twist.linear.x < 0.25:
                        wp.twist.twist.linear.x = 0.0
                elif self.car_state_tl == CAR_STATE_SLOWDOWN:
                    # slow down to max_speed/2
                    wp.twist.twist.linear.x = min(max(max_speed/2, max_speed/2+0.2*(dist_tl-DIST_STOP_TL)), max_speed)
            else:
                wp.twist.twist.linear.x = 0.0
            # add the waypoint to the list
            msg.waypoints.append(wp)
            # increase/decrease index
            index = (index + 1) % self.num_waypoints

        # publish the message
        self.final_waypoints_pub.publish(msg)

    def pose_cb(self, msg):
        # get the pose from the message
        self.pose = msg.pose

    def waypoints_cb(self, waypoints):
        # Check if we have previously retrieved the flag waypoints
        if not self.flag_waypoints_retrieved:
            self.base_waypoints = waypoints.waypoints
            self.num_waypoints = len(self.base_waypoints)
            self.flag_waypoints_retrieved = True

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist

    def traffic_cb(self, msg):
        if msg.data >= 0:
            # Red traffic light detected
            self.tl_waypoint = msg.data
        else:
            self.tl_waypoint = None

        if self.last_tl_waypoint:
            self.car_state_tl = CAR_STATE_FULL_THROTTLE
            self.last_tl_waypoint = self.tl_waypoint
            return

        self.last_tl_waypoint = self.tl_waypoint

        if self.base_waypoints is None or self.next_wp_index is None:
            # not all data received yet
            self.car_state_tl = CAR_STATE_STOP
            # return
        elif self.tl_waypoint is None:
            # no red light so make the car go at full speed
            self.car_state_tl = CAR_STATE_MAX_SPEED
            return

        dist_tl = self.distance(self.base_waypoints, self.next_wp_index, self.tl_waypoint)

        if self.car_state_tl != CAR_STATE_MAX_SPEED and dist_tl < 0.1:
            self.car_state_tl = CAR_STATE_MAX_SPEED
        else:
            self.car_state_tl = CAR_STATE_STOP

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
