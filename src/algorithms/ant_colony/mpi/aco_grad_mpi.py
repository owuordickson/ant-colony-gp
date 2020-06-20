# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "3.0"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"
@modified: "20 June 2020"

"""

import numpy as np
import gc
from ...common.gp import GP
from ...common.hdf5.dataset_h5 import Dataset_h5
from ..hdf5.aco_grad_h5 import GradACO_h5


class Dataset_mpi(Dataset_h5):

    def __init__(self, file_path=None, min_sup=None, eq=False, h5f=None):
        if h5f is not None:
            print("Fetching data from h5 file")

            self.title = h5f['dataset/title'][:]
            # self.time_cols = h5f['dataset/time_cols'][:]
            self.attr_cols = h5f['dataset/attr_cols'][:]
            size = h5f['dataset/size'][:]
            self.column_size = size[0]
            self.size = size[1]
            self.attr_size = size[2]
            self.step_name = 'step_' + str(int(self.size - self.attr_size))
            self.invalid_bins = h5f['dataset/' + self.step_name + '/invalid_bins'][:]

            self.thd_supp = min_sup
            # self.equal = eq
            self.data = None
        else:
            data = Dataset_mpi.read_csv(file_path)
            if len(data) <= 1:
                self.data = np.array([])
                data = None
                print("csv file read error")
                raise Exception("Unable to read csv file or file has no data")
            else:
                print("Data fetched from csv file")
                self.data = np.array([])
                self.title = self.get_title(data)  # optimized (numpy)
                self.time_cols = self.get_time_cols()  # optimized (numpy)
                self.attr_cols = self.get_attributes()  # optimized (numpy)
                self.column_size = self.get_attribute_no()  # optimized (numpy)
                self.size = self.get_size()  # optimized (numpy)
                self.attr_size = 0
                self.step_name = ''
                self.thd_supp = min_sup
                self.equal = eq
                self.invalid_bins = np.array([])
                data = None


class GradACO_mpi(GradACO_h5):

    def __init__(self, d_set, h5f):
        self.d_set = d_set
        self.h5f = h5f
        self.attr_index = self.d_set.attr_cols
        grp = 'dataset/' + self.d_set.step_name + '/p_matrix'
        if grp in h5f:
            p_matrix = h5f[grp][:]
        else:
            p_matrix = np.array([])
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
                exits = GradACO_mpi.is_duplicate(rand_gp, winner_gps, loser_gps)
                if not exits:
                    repeated = 0
                    # check for anti-monotony
                    is_super = GradACO_mpi.check_anti_monotony(loser_gps, rand_gp, subset=False)
                    is_sub = GradACO_mpi.check_anti_monotony(winner_gps, rand_gp, subset=True)
                    if is_super or is_sub:
                        continue
                    gen_gp = self.validate_gp(rand_gp)
                    if gen_gp.support >= min_supp:
                        self.deposit_pheromone(gen_gp)
                        is_present = GradACO_mpi.is_duplicate(gen_gp, winner_gps, loser_gps)
                        is_sub = GradACO_mpi.check_anti_monotony(winner_gps, gen_gp, subset=True)
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
        return winner_gps

    def validate_gp(self, pattern):  # needs to read from h5 file
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = self.d_set.thd_supp
        gen_pattern = GP()
        bin_data = np.array([])

        for gi in pattern.gradual_items:
            if self.d_set.invalid_bins.size > 0 and np.any(np.isin(self.d_set.invalid_bins, gi.gradual_item)):
                continue
            else:
                ds = 'dataset/' + self.d_set.step_name + '/valid_bins/' + gi.as_string()
                if ds in self.h5f:
                    temp = self.h5f[ds][:]
                else:
                    continue
                    # temp = np.array([])
                # for obj in self.d_set.valid_bins:
                #    if obj[0] == gi.gradual_item:
                #        temp = obj[1]
                #        break
                if bin_data.size <= 0:
                    bin_data = np.array([temp, np.array([])])
                    gen_pattern.add_gradual_item(gi)
                else:
                    bin_data[1] = temp
                    temp_bin, supp = self.bin_and(bin_data, self.d_set.attr_size)
                    if supp >= min_supp:
                        bin_data[0] = temp_bin
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.set_support(supp)
        gc.collect()
        if len(gen_pattern.gradual_items) <= 1:
            return pattern
        else:
            return gen_pattern
