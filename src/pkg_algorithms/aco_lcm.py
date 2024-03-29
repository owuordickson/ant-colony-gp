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

from .lcm_gp import LcmGP
from .shared.gp import GI, GP
from .shared.dataset_dfs import DatasetDFS
from .shared.profile import Profile
# from .shared import config as cfg


class LcmACO(LcmGP):

    def __init__(self, f_path, min_supp, evaporation_factor, n_jobs=1):
        # super().__init__(file, min_supp, n_jobs)
        print("LcmACO: Version 1.0")
        self.min_supp = min_supp  # provided by user
        self._min_supp = LcmGP.check_min_supp(self.min_supp)
        self.n_jobs = n_jobs  # n_jobs

        self.d_set = DatasetDFS(f_path, min_supp, eq=False)
        self.D = self.d_set.remove_inv_attrs(self.d_set.encode_data())
        self.size = self.d_set.attr_size
        self.c_matrix = np.ones((self.size, self.size), dtype=np.float64)
        self.p_matrix = np.ones((self.size, self.size), dtype=np.float64)
        np.fill_diagonal(self.p_matrix, 0)
        self.e_factor = evaporation_factor  # evaporation factor
        self.item_to_tids = self._fit()
        # self.large_tids = np.array([])
        # self.attr_index = self.d_set.attr_cols
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
        repeated = 0
        i = 0

        while repeated == 0 and i < self.size:
            # for i in range(self.size):
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
                            temp = temp_tids.copy()
                            temp = temp.intersection(v)
                            supp = self.calculate_support(temp)
                            if supp >= self.min_supp:
                                temp_tids = temp.copy()
                                temp = None
                # check for subset or equality (attrs and temp_attrs)
                if (tuple(temp_attrs) in winner_attrs) or \
                        set(tuple(temp_attrs)).issubset(set(winner_attrs)):
                    repeated = 1
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
            if i >= (self.size - 1):
                i = 0
                repeated += 1
            else:
                i += 1
        gc.collect()
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


def init(f_path, min_supp, e_factor, cores):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        ac = LcmACO(f_path, min_supp, e_factor, n_jobs=num_cores)
        lst_gp = ac.run_ant_colony()

        d_set = ac.d_set
        wr_line = "Algorithm: ACO-LCM (1.0)\n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.row_count) + '\n'
        wr_line += "Minimum support: " + str(ac.min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(lst_gp)) + '\n\n'

        for txt in d_set.titles:
            try:
                wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in lst_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')
        # wr_line += str(df_gp)

        wr_line += "\nPheromone Matrix\n"
        # wr_line += str(ac.p_matrix)
        # ac.plot_pheromone_matrix()
        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line
