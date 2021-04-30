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
from ypstruct import structure


def run(problem, params):
    c_matrix = np.array([[6, 9], [4, 7], [3, 4], [5, 3], [8, 4]])
    u_demand = np.array([80, 270, 250, 160, 180])

    # Problem Information
    costfunc = problem.costfunc

    # Parameters
    maxit = params.maxit
    npop = params.npop
    pc = params.pc
    nc = int(np.round(pc * npop/2) * 2)

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.gene = None
    empty_individual.cost = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].gene = build_gene(problem, c_matrix.shape)
        pop[i].cost = costfunc(pop[i].gene, c_matrix, u_demand)
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Best Cost of Iteration
    bestcost = np.empty(maxit)
    bestgene = [] #np.empty(maxit)

    # Main Loop
    for it in range(maxit):

        popc = []
        for _ in range(nc//2):
            # Select Parents
            q = np.random.permutation(npop)
            p1 = pop[q[0]]
            p2 = pop[q[1]]

            # Perform Crossover
            c1, c2 = crossover(p1, p2)

            # Perform Mutation
            c1 = mutate(c1)
            c2 = mutate(c2)

            # Apply Bound
            # apply_bound(c1, varmin, varmax)
            # apply_bound(c2, varmin, varmax)

            # Evaluate First Offspring
            c1.cost = costfunc(c1.gene, c_matrix, u_demand)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()

            # Evaluate Second Offspring
            c2.cost = costfunc(c2.gene, c_matrix, u_demand)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()

            # Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)

        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x:x.cost)
        pop = pop[0:npop]

        # Store Best Cost
        bestcost[it] = bestsol.cost
        bestgene.append(bestsol.gene)

        # Show Iteration Information
        # print("Iteration {}: Best Cost = {}, Best Gene = {}".format(it, bestcost[it], bestgene[it]))
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out


def crossover(p1, p2):
    c1 = p1.copy()
    c2 = p1.copy()
    choice = np.random.randint(2, size=c1.gene.size).reshape(c1.gene.shape).astype(bool)
    c1.gene = np.where(choice, p1.gene, p2.gene)
    c2.gene = np.where(choice, p2.gene, p1.gene)
    return c1, c2


def mutate(x):
    y = x.copy()
    rand_val_1 = np.random.randint(0, x.gene.shape[0])
    rand_val_2 = np.random.randint(0, x.gene.shape[1])
    if y.gene[rand_val_1, rand_val_2] == 0:
        y.gene[rand_val_1, rand_val_2] = 1
    else:
        y.gene[rand_val_1, rand_val_2] = 0
    return y


def build_gene(prob, shape):
    temp_gene = []
    for i in range(shape[0]):
        temp = np.random.choice(a=prob.vals, size=(shape[1],))
        temp_gene.append(temp)
    return np.array(temp_gene)
