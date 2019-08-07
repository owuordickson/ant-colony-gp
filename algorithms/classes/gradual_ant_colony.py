# -*- coding: utf-8 -*-

"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""


import numpy as np
import random as rand
import matplotlib.pyplot as plt


class GradualAntColony:

    def __init__(self, steps, max_combs, attributes, min_supp):
        self.steps = steps
        self.max_combs = max_combs
        self.thd_supp = min_supp
        self.attributes = attributes
        self.feature = ['+', '-', 'x']
        self.p_matrix = np.ones((len(self.attributes), len(self.feature)), dtype=float)

    def evaluate_solution(self, pattern):
        return 0

    def run_ant_colony(self):
        p = self.p_matrix
        all_sols = []
        sols_win = []
        for t in range(self.steps):
            for n in range(self.max_combs):
                sol_n = []
                for i in range(len(self.attributes)):
                    x = (rand.randint(1, self.max_combs) / self.max_combs)
                    pos = p[i][0] / (p[i][0] + p[i][1] + p[i][2])
                    neg = (p[i][0] + p[i][1]) / (p[i][0] + p[i][1] + p[i][2])
                    if x < pos:
                        temp_n = [self.attributes[i], '+']
                    elif (x >= pos) and x < neg:
                        temp_n = [self.attributes[i], '-']
                    else:
                        # temp_n = 'x'
                        continue
                    if temp_n not in sol_n:
                        sol_n.append(temp_n)
                # if sol_n not in all_sols:
                #    all_sols.append(sol_n)
                print(sol_n)
                supp = self.evaluate_solution(sol_n)
                # test_pattern
                # update p_matrix
        return sols_win, p

    def plot_pheromone_matrix(self, p_matrix, y_attrs):
        X = np.array(p_matrix)
        x_ticks = ['+', '-', 'x']
        x = [0.5, 1.5, 2.5]
        y_ticks = []
        y = []
        for i in range(len(y_attrs)):
            y.append(i + 0.5)
            y_ticks.append(y_attrs[i][1])
        # Figure size (width, height) in inches
        # plt.figure(figsize=(5, 5))
        plt.title("Attribute Gray Plot")
        plt.xlabel("+ => increasing; - => decreasing; x => irrelevant")
        plt.ylabel('Attribute')
        plt.xlim(0, 3)
        plt.ylim(0, len(p_matrix))
        plt.xticks(x, x_ticks)
        plt.yticks(y, y_ticks)
        plt.pcolor(X)
        plt.gray()
        plt.show()