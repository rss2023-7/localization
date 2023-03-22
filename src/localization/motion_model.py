import numpy as np
from math import sin, cos, atan2

class MotionModel:

    def __init__(self, sigma_x=0.05, sigma_y=0.05, sigma_theta=0.05):
        """
        Initialize the motion model with noise coefficients.
        args:
            sigma_x: Standard deviation of noise in x direction
            sigma_y: Standard deviation of noise in y direction
            sigma_theta: Standard deviation of noise in theta direction
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_theta = sigma_theta

    def T_wc(self, x, y, theta):
        return np.array([[cos(theta), -sin(theta), x],
                         [sin(theta), cos(theta), y],
                         [0, 0, 1]])

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.
        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]
            odometry: A 3-vector [dx dy dtheta]
        returns:
            particles: An updated matrix of the
                same size
        """
        num_particles = particles.shape[0]
        updated_particles = np.zeros_like(particles)

        # Add noise to odometry
        delta_x_noisy = odometry[0] + np.random.normal(0, self.sigma_x, num_particles)
        delta_y_noisy = odometry[1] + np.random.normal(0, self.sigma_y, num_particles)
        delta_theta_noisy = odometry[2] + np.random.normal(0, self.sigma_theta, num_particles)

        for i in range(num_particles):
            # Get the transformation matrix for the current particle
            T_wi = self.T_wc(particles[i, 0], particles[i, 1], particles[i, 2])

            # Apply the noisy odometry to the current particle
            T_delta = self.T_wc(delta_x_noisy[i], delta_y_noisy[i], delta_theta_noisy[i])

            # Update the particle pose
            T_wi_new = np.dot(T_wi, T_delta)

            # Extract the updated pose from the transformation matrix
            updated_particles[i, 0] = T_wi_new[0, 2]
            updated_particles[i, 1] = T_wi_new[1, 2]
            updated_particles[i, 2] = particles[i, 2] + delta_theta_noisy[i]

        return updated_particles
