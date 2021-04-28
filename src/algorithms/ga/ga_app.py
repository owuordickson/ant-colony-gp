# -*- coding: utf-8 -*-
"""
@author: "Marziye Derakhshannia and Dickson Owuor"
@license: "MIT"
@version: "1.0"
@email: "dm.derakhshannia@gmail.com or owuordickson@gmail.com "
@created: "07 November 2020"

Description: genetic heuristic algorithm that optimizes data lake jobs

"""


import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga


# GA for demand-jobs
def nodes_count(x):
    return x


def jobs_cost(c):
    return c


def cost_func(gene, c_matrix, u_demand):
    cost = 0
    for i in range(gene.shape[0]):
        # u = gene[i]
        total = np.sum(gene[i])
        if total == 0:
            return np.inf
        for j in range(gene.shape[1]):
            cost += (gene[i][j]/total) * c_matrix[i][j] * u_demand[i]
    return cost


# Demand: number of nodes, cost for every job (stored in matrix)
# demand = structure()
# demand.nodes = nodes_count
# demand.cost = jobs_cost

# Jobs: number of nodes
# job = structure()
# job.max_nodes = nodes_count


# Problem definition
problem = structure()
problem.costfunc = cost_func
problem.vals = [1, 0]

# GA Parameters
params = structure()
params.maxit = 100
params.npop = 20
params.pc = 1

# Run GA
out = ga.run(problem, params)

# Results
plt.plot(out.bestcost)
plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()

