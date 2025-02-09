import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

import math

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale", 1.0)


        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = None
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization) 

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)

        # Map resolution is updated in the map_topic's callback function
        self.map_resolution = None

    def p_hit(self, z, d):
        return 1./np.sqrt(2.*np.pi*self.sigma_hit**2.) * np.exp(-((z-d)**2.) / (2.*self.sigma_hit**2.))

    def p_short(self, z, d):
        res = np.zeros(shape=(self.table_width,self.table_width))
        for (i,j) in np.ndindex(res.shape):
            if not (d[i,j] == 0. or z[i,j] < 0. or d[i,j] < z[i,j]):
                res[i,j] = 2./d[i,j] * (1.-z[i,j]/d[i,j])
        return res

    def p_max(self, z):
        res = np.zeros(shape=(self.table_width,self.table_width))
        res[-1] = np.ones(shape=res[-1].shape)
        return res

    def p_rand(self, z):
        return np.ones(shape=(self.table_width,self.table_width))*1./(self.table_width-1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        p_hit_table = np.fromfunction(lambda z,d: self.p_hit(z,d), (self.table_width, self.table_width))
        norm_p_hit_table = p_hit_table/p_hit_table.sum(axis=0,keepdims=1)
        p_short_table = np.fromfunction(lambda z,d: self.p_short(z,d), (self.table_width, self.table_width))
        p_max_table = np.fromfunction(lambda z,d: self.p_max(z), (self.table_width, self.table_width))
        p_rand_table = np.fromfunction(lambda z,d: self.p_rand(z), (self.table_width, self.table_width))
        raw_sensor_model_table = self.alpha_hit*norm_p_hit_table + self.alpha_short*p_short_table + self.alpha_max*p_max_table + self.alpha_rand*p_rand_table
        self.sensor_model_table = raw_sensor_model_table/raw_sensor_model_table.sum(axis=0,keepdims=1)

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return


        # This produces a matrix of size N x num_beams_per_particle 
        scans = self.scan_sim.scan(particles)

        # downsample data (do this first to prevent preprocessing data that will be removed!)
        step = float(len(observation))/self.num_beams_per_particle
        ind = 0
        new_obs = []
	while(ind < len(observation)):
	    new_obs.append(observation[int(np.rint(ind))])
            ind += step
        observation = np.array(new_obs)

        #observation = observation[::len(observation)/self.num_beams_per_particle]
        #scans = scans[:,::scans.shape[1]/self.num_beams_per_particle]

        # scale meters to pixels
        scans = scans / (self.lidar_scale_to_map_scale * self.map_resolution)
        observation = observation / (self.lidar_scale_to_map_scale * self.map_resolution)

        # clip values to the range [0, self.table_width-1]
        scans = np.clip(scans, 0, self.table_width-1)
        observation = np.clip(observation, 0, self.table_width-1)

        # round all values to the nearest int
        scans = np.rint(scans).astype(int)
        observation = np.rint(observation).astype(int)

        return self.sensor_model_table[observation, scans[0:scans.shape[0]]].prod(axis=1) ** (1/2.2)
    

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        self.map_resolution = map_msg.info.resolution

        print("Map initialized")
