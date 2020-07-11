# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "09 July 2020"

Depth-First Search for gradual patterns (ACO-LCM)

"""

import numpy as np
from numpy import random as rand
import gc
import pandas as pd
from collections import defaultdict

# from joblib import Parallel, delayed
from ..common.lcm_grad import LCM_g
from ..common.gp import GI, GP
from ..common.dataset_dfs import Dataset_dfs


class LcmACO(LCM_g):

    def __init__(self, f_path, min_supp):
        print("LcmACO: Version 1.0")
        self.min_supp = min_supp  # provided by user
        self._min_supp = LCM_g.check_min_supp(self.min_supp)
        # self.item_to_tids = None
        # self.n_jobs = 1  # to be removed

        self.d_set = Dataset_dfs(f_path, min_supp, eq=False)
        self.D = self.d_set.remove_inv_attrs(self.d_set.encode_data())
        self.size = self.d_set.attr_size
        self.c_matrix = np.ones((self.size, self.size), dtype=np.float64)
        self.p_matrix = np.ones((self.size, self.size), dtype=np.int64)
        np.fill_diagonal(self.p_matrix, 0)
        self.e_factor = 0.5  # evaporation factor
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

    def deposit_pheromone(self, node):
        self.p_matrix[node[0], node[1]] += 1

    def evaporate_pheromone(self, node):
        self.p_matrix[node[0], node[1]] = \
            (1 - self.e_factor) * self.p_matrix[node[0], node[1]]

    def run_ant_colony(self, return_tids=False):
        empty_df = pd.DataFrame(columns=['pattern', 'support', 'tids'])
        dfs = list()
        item_to_tids = self._fit()

        i = 0
        lst_attrs = list()
        lst_gp = list()
        while i < self.size:
            temp_tid = None
            temp_attrs = list()
            node = self.generate_random_node(i)
            if len(node) > 1:
                for k, v in item_to_tids.items():
                    if node in v:
                        temp_attrs.append(k)
                        if temp_tid is None:
                            temp_tid = v
                        else:
                            temp_tid = temp_tid.intersection(v)
                # check for subset or equality (attrs and temp_attrs)
                if temp_attrs in lst_attrs or len(temp_tid) <= 0:
                    i += 1
                    # continue
                    break
                else:
                    supp = self.calculate_support(temp_tid)
                    if supp >= self.min_supp:
                        self.deposit_pheromone(node)
                        gp = LcmACO.construct_gp(temp_attrs, supp)
                        lst_attrs.append(temp_attrs)
                        lst_gp.append([gp.to_string(), supp, temp_tid])
                    else:
                        self.evaporate_pheromone(node)
            i += 1

        dfs.append(pd.DataFrame(data=lst_gp, columns=['pattern', 'support', 'tids']))
        dfs.append(empty_df)  # make sure we have something to concat
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if not return_tids:
            df.drop('tids', axis=1, inplace=True)
        return df

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
