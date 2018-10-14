import rospy

from pid import PID
from yaw_controller import YawController
import numpy as np
import math

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
NUM_WHEEL_ACCELERATION = 2
MIN_SPEED = 5.0

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                                    wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.min_steer_angle = -max_steer_angle
        self.max_steer_angle = max_steer_angle

        self.min_force = vehicle_mass * accel_limit # max brake force
        self.force_brake_deadband = -1.0 * vehicle_mass * brake_deadband
        self.vehicle_accel_sensitivity = vehicle_mass*wheel_radius

        # force at 100% throttle position
        self.ref_force_throttle = vehicle_mass * 3.0

        # store all the previsous steer, speed and acceleration values
        self.last_speed = 0.0
        self.last_target_accel = 0.0
        self.last_expected_speed = 0.0
        self.last_force = 0.0
        self.last_steer = 0.0
        # brakes are not applied initially
        self.last_brake_applied = False
        # init feed forward yaw-rate control
        self.yawControl = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)
        # initialize acceleration and steering PID
        self.accel_PID = PID(kp=2.0 * self.vehicle_accel_sensitivity, ki=0.015 * self.vehicle_accel_sensitivity, kd=0.2 * self.vehicle_accel_sensitivity)
        self.steer_PID = PID(kp=0.05, ki=0.0005, kd=0.5)

    def control(self, waypoints, pose, current_speed, target_speed, target_yaw, delta_t):
        throttle, brake = self.control_speed(current_speed, target_speed, delta_t)
        steer = self.control_steer(waypoints, pose, current_speed, target_speed, target_yaw, delta_t)
        return throttle, brake, steer

    def control_speed(self, current_speed, target_speed, delta_t):
        if target_speed > 0.05 or current_speed > 1.0:
            # overall resistance (tuned in simulator)
            force_resistance = 4 * current_speed**2 + 40 * current_speed + 40
            # use PID to get better control performance
            CTE = self.speed_CTE(current_speed, target_speed, delta_t)
            force_PID = self.accel_PID.step(CTE, delta_t)
            # calculate force from target_accel using mass: F = m * a
            force_feedForward = self.last_target_accel * self.vehicle_mass + force_resistance
            # calulate overall torque
            force = force_feedForward + force_PID
        else:
            # apply minimum force if vehicle is near the target speed
            force = min(self.last_force, self.min_force / 2)
            self.accel_PID.freeze()

        # Initialiaze the throttle, brake and last_brake_applied
        throttle = 0.0
        brake = 0.0
        self.last_brake_applied = False
        # calulate throttle
        # published throttle is a percentage [0 ... 1]!!!
        if force > 0.0:
            throttle = force / self.ref_force_throttle
        elif force < self.force_brake_deadband or target_speed < 1.0:
            brake = -force * self.wheel_radius / NUM_WHEEL_ACCELERATION
            self.last_brake_applied = True

        # store last_speed and last_force for next iteration
        self.last_speed = current_speed
        self.last_force = force

        return throttle, brake

    def speed_CTE(self, current_speed, target_speed, delta_t):
        # calculate speed error
        speed_error = target_speed - current_speed
        # calulate target acceleration from speed error
        target_accel = 0.3 if target_speed > 0.05 else 1.0
        if abs(speed_error) < 1.0:
            target_accel *= speed_error
        else:
            target_accel *= speed_error**3
        # check for min and max allowed acceleration
        target_accel = max(min(target_accel, self.accel_limit), self.decel_limit)

        self.last_target_accel = target_accel
        # calculate CTE and return it
        return speed_error

    def control_steer(self, waypoints, pose, current_speed, target_speed, target_yaw, delta_t):
        if current_speed < MIN_SPEED:
            self.steer_PID.freeze()
            return self.last_steer
        # feed forward control to drive curvature of road
        steer_feedForward = self.yawControl.get_steering(target_speed, target_yaw, current_speed)
        # limit steering angle
        steer_feedForward = max(min(steer_feedForward, self.max_steer_angle), self.min_steer_angle)
        # PID control
        CTE = self.steer_CTE(waypoints, pose)
        steer_PID = self.steer_PID.step(CTE, delta_t)
        # steering command
        steer = steer_feedForward + steer_PID
        self.last_steer = steer
        return steer

    def steer_CTE(self, waypoints, pose):
        # transfrom waypoints into vehicle coordinates
        car_theta = 2 * np.arccos(pose.orientation.w)
        # Constraining the angle in [-pi, pi)
        if car_theta > np.pi:
            car_theta = -(2 * np.pi - car_theta)
        num_waypoints = len(waypoints)
        car_x = pose.position.x
        car_y = pose.position.y
        # transform waypoints in vehicle coordiantes
        wp_car_coords_x = np.zeros(num_waypoints)
        wp_car_coords_y = np.zeros(num_waypoints)

        for idx, waypoint in enumerate(waypoints):
            wp_x = waypoint.pose.pose.position.x
            wp_y = waypoint.pose.pose.position.y
            wp_car_coords_x[idx] = (wp_y-car_y)*math.sin(car_theta)-(car_x-wp_x)*math.cos(car_theta)
            wp_car_coords_y[idx] = (wp_y-car_y)*math.cos(car_theta)-(wp_x-car_x)*math.sin(car_theta)

        # get waypoint which should be used for controller input#
        # get the first waypoint in front of car. The waypoints are already in order as
        # they are coming out of /waypoint_updater topic
        viewRange = 10.0
        first_wp_idx = -1
        last_wp_idx = -1
        first_wp_found = False

        for idx, wp_car_coord_x in enumerate(wp_car_coords_x):
            if not first_wp_found and wp_car_coord_x >= 0.0:
                first_wp_idx = idx
                first_wp_found = True
            elif wp_car_coord_x >= viewRange:
                last_wp_idx = idx
                break

        # Check if we have enough waypoints to move forward We need over 5 waypoints
        if first_wp_idx < 0 or last_wp_idx - first_wp_idx < 4:
            rospy.logerr("twist_controller: Invalid Waypoints")
            return 0.0

        # Interpolate waypoints (already transformed to vehicle coordinates) to a polynomial of 3rd degree
        coeffs = np.polyfit(wp_car_coords_x[first_wp_idx:last_wp_idx+1], wp_car_coords_y[first_wp_idx:last_wp_idx+1], 3)
        # distance to track is polynomial at car's position x = 0
        return np.poly1d(coeffs)(0.0)

    def reset(self, current_speed):
        self.steer_PID.reset()
        self.accel_PID.reset()
        self.last_speed = current_speed
        self.last_target_accel = 0.0
        self.last_expected_speed = current_speed
        self.last_force = 0.0
