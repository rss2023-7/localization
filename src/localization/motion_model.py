import numpy as np

class MotionModel:

    def __init__(self):
        # Initialize noise parameters for motion model
        self.translation_noise_std = 0.0005  # Standard deviation for translation noise
        self.rotation_noise_std = 0.0005    # Standard deviation for rotation noise


    def evaluate(self, particles, odometry):
        # Update the particles to reflect probable future states given the odometry data
        num_particles = len(particles)

        for i in range(num_particles):
            # Add noise to the odometry data
            noisy_odometry = np.copy(odometry)
            noisy_odometry[0] += np.random.normal(0, self.translation_noise_std)
            noisy_odometry[1] += np.random.normal(0, self.translation_noise_std)
            noisy_odometry[2] += np.random.normal(0, self.rotation_noise_std)
            print("git test")
            # Apply the noisy odometry data to the particle
            x, y, theta = particles[i]
            dx, dy, dtheta = noisy_odometry

            # Compute the new particle position based on the noisy odometry
            x_new = x + dx * np.cos(theta) - dy * np.sin(theta)
            y_new = y + dx * np.sin(theta) + dy * np.cos(theta)
            theta_new = theta + dtheta

            # Normalize the angle to be between -pi and pi
            theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

            # Update the particle
            particles[i] = [x_new, y_new, theta_new]

        return particles
