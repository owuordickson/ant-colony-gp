# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "3.0"
@email: "owuordickson@gmail.com"
@created: "06 July 2020"

Depth-First Search for gradual patterns (ACO-LCM)

"""

import numpy as np
from numpy import random as rand
from ..common.gp import GI, GP
from ..common.dataset_dfs import Dataset_dfs


class GradACO_dfs:

    def __init__(self, f_path, min_supp, eq):
        print("GradACO: Version 2.0")
        self.d_set = Dataset_dfs(f_path, min_supp, eq)
        self.d_set.init_gp_attributes()
        self.c_matrix = self.d_set.cost_matrix
        self.p_matrix = np.ones((self.d_set.column_size, 3), dtype=int)
        # Data set reduction and update: (p_matrix, attr_index)
        # self.bins, self.indices = self.reduce_data()
        self.reduce_data()
        self.attr_index = self.d_set.attr_cols
        # self.e_factor = 0.1  # evaporation factor
        print(self.d_set.encoded_data)

    def reduce_data(self):
        c_matrix = self.c_matrix
        # 1. remove invalid attributes
        valid_a1 = list()
        valid_a2 = [-2, -1]
        for i in range(len(self.d_set.attr_cols)):
            a = self.d_set.attr_cols[i]
            valid = (c_matrix[a][0] < c_matrix[a][2]) or \
                    (c_matrix[a][1] < c_matrix[a][2])
            if valid:
                valid_a1.append(i)
                valid_a2.append(i)
            else:
                self.p_matrix[a][0] = 0
                self.p_matrix[a][1] = 0
        self.d_set.attr_cols = self.d_set.attr_cols[valid_a1]
        valid_a2 = np.array(valid_a2) + 2
        self.d_set.encoded_data = self.d_set.encoded_data[:, valid_a2]
        # 2. merge similar patterns
        # 2a. get indices
        # vals, inverse, count = np.unique(self.d_set.encoded_data[:, 2:],
        #                                 return_inverse=True,
        #                                 return_counts=True,
        #                                 axis=0)
        # return vals, inverse

    def generate_random_gp(self):
        p = self.p_matrix
        c = self.c_matrix
        attrs = self.attr_index.copy()
        np.random.shuffle(attrs)
        n = len(attrs)
        pattern = GP()
        for i in attrs:
            max_extreme = n  # * 100
            x = float(rand.randint(1, max_extreme) / max_extreme)
            p0 = p[i][0] * (1 / c[i][0])
            p1 = p[i][1] * (1 / c[i][1])
            p2 = p[i][2] * (1 / c[i][2])
            pos = float(p0 / (p0 + p1 + p2))
            neg = float((p0 + p1) / (p0 + p1 + p2))
            if x < pos:
                temp = GI(i, '+')
            elif (x >= pos) and (x < neg):
                temp = GI(i, '-')
            else:
                # temp = GI(self.attr_index[i], 'x')
                continue
            pattern.add_gradual_item(temp)
        return pattern

    def validate_gp(self, pattern):
        pass
