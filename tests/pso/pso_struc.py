# Source: https://gist.github.com/IranNeto/21542660d740ac02dfce2f6aeb11ebeb#file-pso_struc-py

import random
import numpy as np
import matplotlib.pyplot as plt


# function that models the problem
def fitness_function(position):
    return position[0] ** 2 + position[1] ** 2 + 1


# Some variables to calculate the velocity
W = 0.5
c1 = 0.5
c2 = 0.9
target = 1

n_iterations = 100  # int(input("Inform the number of iterations: "))
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
bestpos = np.empty(n_iterations)
while iteration < n_iterations:
    for i in range(n_particles):
        fitness_candidate = fitness_function(particle_position_vector[i])
        # print(fitness_candidate, ' ', particle_position_vector[i])

        if pbest_fitness_value[i] > fitness_candidate:
            pbest_fitness_value[i] = fitness_candidate
            pbest_position[i] = particle_position_vector[i]

        if gbest_fitness_value > fitness_candidate:
            gbest_fitness_value = fitness_candidate
            gbest_position = particle_position_vector[i]
    bestpos[iteration] = fitness_function(gbest_position)
    if abs(gbest_fitness_value - target) < target_error:
        break

    for i in range(n_particles):
        new_velocity = (W * velocity_vector[i]) + (c1 * random.random()) * (
                    pbest_position[i] - particle_position_vector[i]) + (c2 * random.random()) * (
                                   gbest_position - particle_position_vector[i])
        new_position = new_velocity + particle_position_vector[i]
        particle_position_vector[i] = new_position

    iteration = iteration + 1

print("The best position is ", gbest_position, "in iteration number ", iteration)


# Results
plt.plot(bestpos)
plt.semilogy(bestpos)
plt.xlim(0, n_iterations)
plt.xlabel('Iterations')
plt.ylabel('Global Best Position')
plt.title('Pattern Swarm Algorithm (PSO)')
plt.grid(True)
plt.show()
