#!/usr/bin/env python2

import numpy as np
import rospy
from sensor_model import SensorModel
from motion_model import MotionModel
from collections import defaultdict

import tf2_ros
import tf.transformations as trans
from geometry_msgs.msg import TransformStamped, PoseArray, Pose

import matplotlib.pyplot as plt

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32

class ParticleFilter:

    def __init__(self):


        # init stuff
        self.rate = 50
        self.particles = None
        rospy.Rate(self.rate)

        self.prev_time = None
        self.count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

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

        self.points_pub = rospy.Publisher("/pf/points", PoseArray, queue_size = 1)

        self.error_pub = rospy.Publisher("/linear_error", Float32, queue_size = 1)


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
        # rospy.loginfo(str(initial_x)+" "+str(initial_y)+" "+str(initial_theta))
        self.particles = np.hstack((np.random.normal(initial_x, .25, (self.num_particles, 1)),
                                    np.random.normal(initial_y, .25, (self.num_particles, 1)),
                                    np.random.normal(initial_theta, .5, (self.num_particles, 1))))
        self.compute_particle_avg()

    def laser_callback(self, laser_msg):
        """
        Updates the probabilities of the filter particles

        args:
            laser_msg: A LaserScan msg object

        """

        if self.particles is None:
            return

        self.count += 1
        if self.count % 10 != 0:
            return

        probs = self.sensor_model.evaluate(self.particles, np.array(laser_msg.ranges))
        probs = probs / np.sum(probs)
        self.particles = (self.particles[np.random.choice(self.num_particles, size=self.num_particles, p=probs)] + 
                                       np.hstack((np.random.normal(0, .1, (self.num_particles, 1)),
                                       np.random.normal(0, .1, (self.num_particles, 1)),
                                       np.random.normal(0, .5, (self.num_particles, 1)))))
        
        self.compute_particle_avg()


    def odom_callback(self, odom_msg):
        """
        Update the particle positions of the filter

        args:
            odom_msg: A Odometry msg
        """

        if self.particles is None:
            return


        # create the [dx, dy, dtheta] from the odom msg
        # rospy.loginfo([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.angular.z])

        time_dif = 0
        if self.prev_time is None:
            time_dif = 1 / self.rate
        else:
            time_dif = odom_msg.header.stamp.to_time() - self.prev_time
        self.prev_time = odom_msg.header.stamp.to_time()

        odom_list = time_dif * np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.angular.z])
        # rospy.loginfo(odom_list)

        self.particles = self.motion_model.evaluate(self.particles, odom_list)

        self.compute_particle_avg()


    def compute_particle_avg(self):
        """
        Returns an [x, y, theta] from taking the average of self.particles
        """

        pose_arr_msg = PoseArray()
        pose_arr_msg.header.stamp = rospy.Time.now()
        pose_arr_msg.header.frame_id = "map"
        pose_arr = []
        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0

            quaternion = trans.quaternion_about_axis(particle[2], (0,0,1))
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]
            pose_arr.append(pose)
        pose_arr_msg.poses = pose_arr
        self.points_pub.publish(pose_arr_msg)

        bucket_size = 0.5
        discretization_factor_x = (np.max(self.particles[:,0]) - np.min(self.particles[:,0])) / bucket_size
        discretization_factor_y = (np.max(self.particles[:,1]) - np.min(self.particles[:,1])) / bucket_size


        x_y_indices = np.array([self.particles[:,0] // discretization_factor_x, 
                               self.particles[:,1] // discretization_factor_y])
        x_y_indices = np.transpose(x_y_indices)
        # rospy.loginfo(str(x_y_indices))

        coord_freq = defaultdict(int)
        for i in range(self.num_particles):
            coord_freq[(x_y_indices[i,0], x_y_indices[i,1])] += 1
    
        most_freq_key = max(coord_freq, key=coord_freq.get)
        # rospy.loginfo(most_freq_key)

        avg_x = 0
        avg_y = 0
        thetas = []
        count = coord_freq[most_freq_key]
        # count = self.num_particles
        for i in range(self.num_particles):
            if (self.particles[i,0] // discretization_factor_x == most_freq_key[0] and 
                self.particles[i,1] // discretization_factor_y == most_freq_key[1]):
                avg_x += self.particles[i,0]
                avg_y += self.particles[i,1]
                thetas.append(self.particles[i,2])


        avg_theta = np.angle(np.sum(np.exp(np.array(thetas) * 1j)))

        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "map"
        odom_msg.pose.pose.position.x = avg_x / count
        odom_msg.pose.pose.position.y = avg_y / count
        odom_msg.pose.pose.position.z = 0

        quaternion = trans.quaternion_about_axis(avg_theta, (0,0,1))
        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
        odom_msg.pose.pose.orientation.w = quaternion[3]

        self.odom_pub.publish(odom_msg)
        # rospy.loginfo("most populated: "+str(most_freq_key[0]*discretization_factor_x)+","+str(most_freq_key[1]*discretization_factor_y))
        # rospy.loginfo("avg: "+str(odom_msg.pose.pose.position.x)+", "+str(odom_msg.pose.pose.position.y)+", "+str(avg_theta))

        map_to_base_link_trans = TransformStamped()
        map_to_base_link_trans.header.stamp = rospy.Time.now()
        map_to_base_link_trans.header.frame_id = "map"
        map_to_base_link_trans.child_frame_id = self.particle_filter_frame
        t = map_to_base_link_trans.header.stamp.to_sec()
        map_to_base_link_trans.transform.translation.x = odom_msg.pose.pose.position.x
        map_to_base_link_trans.transform.translation.y = odom_msg.pose.pose.position.y
        map_to_base_link_trans.transform.translation.z = odom_msg.pose.pose.position.z
        map_to_base_link_trans.transform.rotation.x = quaternion[0]
        map_to_base_link_trans.transform.rotation.y = quaternion[1]
        map_to_base_link_trans.transform.rotation.z = quaternion[2]
        map_to_base_link_trans.transform.rotation.w = quaternion[3]
        self.tf_broadcaster.sendTransform(map_to_base_link_trans)

        # get the ground truth transformation (for sim)
        # rospy.loginfo("trying to publish data")
        try:
            ground_truth_pose = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time())
            ground_truth_x = ground_truth_pose.transform.translation.x
            ground_truth_y = ground_truth_pose.transform.translation.y

            euclidean_dist_error = np.sqrt((ground_truth_x-odom_msg.pose.pose.position.x) ** 2+(ground_truth_y-odom_msg.pose.pose.position.y) ** 2)
            self.error_pub.publish(euclidean_dist_error)

        except Exception,e:
            rospy.loginfo(str(e))
            pass
    
    
if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
