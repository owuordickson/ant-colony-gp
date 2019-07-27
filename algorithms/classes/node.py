# -*- coding: utf-8 -*-

"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""


class Node:

    def __init__(self, index, value):
        self.index = index
        self.value = value
        self.pheromone = 0

    def update_pheromone(self, update, value):
        # update description
        # -1 -> decrease
        # 0 -> same
        # 1 -> increase
        if update == -1:
            self.pheromone = float(self.pheromone) - 0.1
        elif update == 0:
            self.pheromone = float(value)
        elif update == 1:
            self.pheromone = float(self.pheromone) + float(value) + 0.1
        else:
            self.pheromone = 0
