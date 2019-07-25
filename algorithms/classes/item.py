# -*- coding: utf-8 -*-

"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""


class Item:

    def __init__(self, index, value):
        self.index = index
        self.value = value
        self.pheromone = 0

    def update_pheromone(self, direction):
        # 0 -> decrease
        # 1 -> increase
        if direction == 0:
            self.pheromone = self.pheromone - 0.1
        elif direction == 1:
            self.pheromone = self.pheromone + 0.1
        else:
            self.pheromone = 0
