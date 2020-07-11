# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "09 July 2020"

Depth-First Search for gradual patterns (ACO-LCM)

"""

import numpy as np
from numpy import random as rand
import gc
from collections import defaultdict

# from joblib import Parallel, delayed
from ..common.lcm_grad import LCM_g
from ..common.gp import GI, GP
from ..common.dataset_dfs import Dataset_dfs


class LcmACO(LCM_g):

    def __init__(self, f_path, min_supp, n_jobs=1):
        print("LcmACO: Version 1.0")
        self.min_supp = min_supp  # provided by user
        self._min_supp = LCM_g.check_min_supp(self.min_supp)
        self.n_jobs = 1  # n_jobs

        self.d_set = Dataset_dfs(f_path, min_supp, eq=False)
        self.D = self.d_set.remove_inv_attrs(self.d_set.encode_data())
        self.size = self.d_set.attr_size
        self.c_matrix = np.ones((self.size, self.size), dtype=np.float64)
        self.p_matrix = np.ones((self.size, self.size), dtype=np.float64)
        np.fill_diagonal(self.p_matrix, 0)
        self.e_factor = 0.5  # evaporation factor
        self.item_to_tids = self._fit()
        # self.large_tids = np.array([])
        # self.attr_index = self.d_set.attr_cols
        # self.e_factor = 0.1  # evaporation factor
        # print(self.d_set.cost_matrix)
        # print(self.d_set.encoded_data)
        # print(self.p_matrix)

    def _fit(self):
        item_to_tids = defaultdict(set)

        # 1. group similar items
        for t in range(len(self.D)):
            transaction = self.D[t][2:]
            for item in transaction:
                item_to_tids[item].add(tuple(self.D[t][:2]))

        # 2. reduce data set
        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.size

        low_supp_items = [k for k, v in item_to_tids.items()
                          if len(np.unique(np.array(list(v))[:, 0], axis=0))
                          < self._min_supp]
        for item in low_supp_items:
            del item_to_tids[item]

        # 3. construct cost_matrix
        tids = item_to_tids.values()
        for nodes in tids:
            idx = np.array(list(nodes))
            np.add.at(self.c_matrix, (idx[:, 0], idx[:, 1]), 1)
        self.c_matrix = 1 / self.c_matrix

        self.D = None
        gc.collect()
        return item_to_tids

    def run_ant_colony(self):
        winner_attrs = list()
        lst_gp = list()

        for i in range(self.size):
            temp_tids = None
            temp_attrs = list()
            node = self.generate_random_node(i)
            if len(node) > 1:
                for k, v in self.item_to_tids.items():
                    if node in v:
                        temp_attrs.append(k)
                        if temp_tids is None:
                            temp_tids = v
                        else:
                            temp_tids = temp_tids.intersection(v)
                # check for subset or equality (attrs and temp_attrs)
                if (tuple(temp_attrs) in winner_attrs) or \
                        set(tuple(temp_attrs)).issubset(set(winner_attrs)):
                    break
                if len(temp_tids) <= 0:
                    continue
                else:
                    supp = self.calculate_support(temp_tids)
                    if supp >= self.min_supp:
                        self.deposit_pheromone(temp_tids)
                        gp = LcmACO.construct_gp(temp_attrs, supp)
                        winner_attrs.append(tuple(temp_attrs))
                        # lst_gp.append([gp.to_string(), supp, temp_tids])
                        lst_gp.append(gp)
                    else:
                        self.evaporate_pheromone(temp_tids)
        return lst_gp

    def generate_random_node(self, i):
        C = self.c_matrix
        P = self.p_matrix
        n = self.size
        ph = P[i] * (1 / C[i])
        tot_sum = np.sum(ph)
        # x = float(rand.randint(1, n) / n)
        x = rand.uniform(0, 1)
        for j in range((i + 1), n):
            p_sum = np.sum(ph[(i+1):(j+1)])
            pr = p_sum / tot_sum
            if x < pr:
                return tuple([i, j])
        return tuple([])

    def deposit_pheromone(self, tids):
        for node in tids:
            self.p_matrix[node] += 1

    def evaporate_pheromone(self, tids):
        for node in tids:
            self.p_matrix[node] = (1 - self.e_factor) * self.p_matrix[node]

    @staticmethod
    def construct_gp(lst_attrs, supp):
        pat = GP()
        for a in lst_attrs:
            if a < 0:
                sym = '-'
            elif a > 0:
                sym = '+'
            else:
                sym = 'x'
            attr = abs(a) - 1
            pat.add_gradual_item(GI(attr, sym))
        pat.set_support(supp)
        return pat
