# Source: https://gist.github.com/IranNeto/21542660d740ac02dfce2f6aeb11ebeb#file-pso_struc-py

import random
import numpy as np


# function that models the problem
def fitness_function(position):
    return position[0] ** 2 + position[1] ** 2 + 1


# Some variables to calculate the velocity
W = 0.5
c1 = 0.5
c2 = 0.9
target = 1

n_iterations = 50  # int(input("Inform the number of iterations: "))
target_error = 1e-6  #float(input("Inform the target error: "))
n_particles = 30  # int(input("Inform the number of particles: "))

particle_position_vector = np.array([np.array([(-1) ** (bool(random.getrandbits(1))) * random.random() * 50,
                                               (-1) ** (bool(random.getrandbits(1))) * random.random() * 50]) for _ in
                                     range(n_particles)])
pbest_position = particle_position_vector
pbest_fitness_value = np.array([float('inf') for _ in range(n_particles)])
gbest_fitness_value = float('inf')
gbest_position = np.array([float('inf'), float('inf')])

velocity_vector = ([np.array([0, 0]) for _ in range(n_particles)])
iteration = 0
while iteration < n_iterations:
    for i in range(n_particles):
        fitness_cadidate = fitness_function(particle_position_vector[i])
        print(fitness_cadidate, ' ', particle_position_vector[i])

        if (pbest_fitness_value[i] > fitness_cadidate):
            pbest_fitness_value[i] = fitness_cadidate
            pbest_position[i] = particle_position_vector[i]

        if (gbest_fitness_value > fitness_cadidate):
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position_vector[i]

    if (abs(gbest_fitness_value - target) < target_error):
        break

    for i in range(n_particles):
        new_velocity = (W * velocity_vector[i]) + (c1 * random.random()) * (
                    pbest_position[i] - particle_position_vector[i]) + (c2 * random.random()) * (
                                   gbest_position - particle_position_vector[i])
        new_position = new_velocity + particle_position_vector[i]
        particle_position_vector[i] = new_position

    iteration = iteration + 1

print("The best position is ", gbest_position, "in iteration number ", iteration)