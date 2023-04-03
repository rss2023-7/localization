#!/usr/bin/env python2

import numpy as np
import rospy
from sensor_model import SensorModel
from motion_model import MotionModel
from collections import defaultdict

import matplotlib.pyplot as plt

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

import tf.transformations as trans

class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")
        self.num_particles  = rospy.get_param("~num_particles")

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")

        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.laser_callback, # TODO: Fill this in
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odom_callback, # TODO: Fill this in
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.generate_particles,
                                          queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        self.rate = 20

        rospy.Rate(self.rate)

        # Initialize the particles
        # self.particles = self.generate_particles()

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def generate_particles(self, initial_pose_msg):
        """
        Generates a matrix of particles for the filter at initialization

        modifies:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]
        """
        initial_pose = initial_pose_msg.pose.pose
        initial_x = initial_pose.position.x
        initial_y = initial_pose.position.y
        initial_theta, _, _ = trans.rotation_from_matrix(trans.quaternion_matrix([initial_pose.orientation.x, initial_pose.orientation.y, initial_pose.orientation.z, initial_pose.orientation.w]))
        rospy.loginfo(str(initial_x)+" "+str(initial_y)+" "+str(initial_theta))
        self.particles = np.hstack((np.random.normal(initial_x, .5, (self.num_particles, 1)),
                                    np.random.normal(initial_y, .5, (self.num_particles, 1)),
                                    np.random.normal(initial_theta, .5, (self.num_particles, 1))))
        self.compute_particle_avg()

    def laser_callback(self, laser_msg):
        """
        Updates the probabilities of the filter particles

        args:
            laser_msg: A LaserScan msg object

        """
        probs = self.sensor_model.evaluate(self.particles, laser_msg.ranges)
        self.particles = (np.random.choice(self.particles, self.num_particles, probs) + 
                                       np.hstack((np.random.normal(0, .5, (self.num_particles, 1)),
                                       np.random.normal(0, .5, (self.num_particles, 1)),
                                       np.random.normal(0, .5, (self.num_particles, 1)))))


    def odom_callback(self, odom_msg):
        """
        Update the particle positions of the filter

        args:
            odom_msg: A Odometry msg
        """

        # create the [dx, dy, dtheta] from the odom msg
        odom_list = self.rate * np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.angular.z])

        self.particles = self.motion_model.evaluate(self.particles, odom_list)

    def compute_particle_avg(self):
        """
        Returns an [x, y, theta] from taking the average of self.particles
        """

        bucket_size = 0.5
        discretization_factor_x = (np.max(self.particles[:,0]) - np.min(self.particles[:,0])) / bucket_size
        discretization_factor_y = (np.max(self.particles[:,1]) - np.min(self.particles[:,1])) / bucket_size


        x_y_indices = np.array([self.particles[:,0] // discretization_factor_x, 
                               self.particles[:,1] // discretization_factor_y])

        coord_freq = defaultdict(int)
        for i in range(x_y_indices.shape[1]):
            coord_freq[(x_y_indices[0][i], x_y_indices[1][i])] += 1
    

        most_freq_key = max(coord_freq, key=coord_freq.get)

        avg_x = 0
        avg_y = 0
        thetas = []
        count = 0
        for i in range(self.particles.shape[1]):
            if (self.particles[i,0] // discretization_factor_x == most_freq_key[0] and 
                self.particles[i,1] // discretization_factor_y  == most_freq_key[1]):
                avg_x += self.particles[i,0]
                avg_y += self.particles[i,1]
                thetas.append(self.particles[i,2])

                count += 1

        avg_theta = np.angle(np.sum(np.exp(np.array(thetas) * 1j)))

        odom_msg = Odometry()
        odom_msg.pose.position.x = avg_x / count
        odom_msg.pose.position.y = avg_y / count
        odom_msg.pose.position.z = 0

        quaternion = trans.quaternion_about_axis(avg_theta, (0,0,1))
        odom_msg.pose.quaternion.x = quaternion[0]
        odom_msg.pose.quaternion.y = quaternion[1]
        odom_msg.pose.quaternion.z = quaternion[2]
        odom_msg.pose.quaternion.w = quaternion[3]

        self.odom_pub.publish(odom_msg)
    
    
if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
