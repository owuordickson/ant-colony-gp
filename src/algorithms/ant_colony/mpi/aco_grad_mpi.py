# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "3.0"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"
@modified: "16 June 2020"

"""

import numpy as np
from ...common.gp import GP
from ...common.hdf5.dataset_h5 import Dataset_h5
from ..aco_grad import GradACO


class GradACO_h5(GradACO):

    def __init__(self, f_path, min_supp, eq):
        self.d_set = Dataset_h5(f_path, min_supp, eq)
        self.d_set.init_attributes()
        self.attr_index = self.d_set.attr_cols
        # self.e_factor = 0.1  # evaporation factor
        # fetch previous p_matrix from memory
        grp = 'dataset/' + self.d_set.step_name + '/p_matrix'
        p_matrix = self.d_set.read_h5_dataset(grp)
        if np.sum(p_matrix) > 0:
            self.p_matrix = p_matrix
        else:
            self.p_matrix = np.ones((self.d_set.column_size, 3), dtype=float)

    def run_ant_colony(self):
        min_supp = self.d_set.thd_supp
        winner_gps = list()  # subsets
        loser_gps = list()  # supersets
        repeated = 0
        while repeated < 1:
            rand_gp = self.generate_random_gp()
            if len(rand_gp.gradual_items) > 1:
                # print(rand_gp.get_pattern())
                exits = GradACO.is_duplicate(rand_gp, winner_gps, loser_gps)
                if not exits:
                    repeated = 0
                    # check for anti-monotony
                    is_super = GradACO.check_anti_monotony(loser_gps, rand_gp, subset=False)
                    is_sub = GradACO.check_anti_monotony(winner_gps, rand_gp, subset=True)
                    if is_super or is_sub:
                        continue
                    gen_gp = self.validate_gp(rand_gp)
                    if gen_gp.support >= min_supp:
                        self.deposit_pheromone(gen_gp)
                        is_present = GradACO.is_duplicate(gen_gp, winner_gps, loser_gps)
                        is_sub = GradACO.check_anti_monotony(winner_gps, gen_gp, subset=True)
                        if is_present or is_sub:
                            repeated += 1
                        else:
                            winner_gps.append(gen_gp)
                    else:
                        loser_gps.append(gen_gp)
                        # update pheromone as irrelevant with loss_sols
                        # self.vaporize_pheromone(gen_gp, self.e_factor)
                    if set(gen_gp.get_pattern()) != set(rand_gp.get_pattern()):
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1
        grp = 'dataset/' + self.d_set.step_name + '/p_matrix'
        self.d_set.add_h5_dataset(grp, self.p_matrix)
        return winner_gps

    def validate_gp(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = self.d_set.thd_supp
        gen_pattern = GP()
        bin_data = np.array([])

        for gi in pattern.gradual_items:
            if self.d_set.invalid_bins.size > 0 and np.any(np.isin(self.d_set.invalid_bins, gi.gradual_item)):
                continue
            else:
                grp = 'dataset/' + self.d_set.step_name + '/valid_bins/' + gi.as_string()
                temp = self.d_set.read_h5_dataset(grp)
                if bin_data.size <= 0:
                    bin_data = np.array([temp, temp])
                    gen_pattern.add_gradual_item(gi)
                else:
                    bin_data[1] = temp
                    temp_bin, supp = self.bin_and(bin_data, self.d_set.attr_size)
                    if supp >= min_supp:
                        bin_data[0] = temp_bin
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.set_support(supp)
        if len(gen_pattern.gradual_items) <= 1:
            return pattern
        else:
            return gen_pattern
