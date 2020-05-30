# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Joseph Orero and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "19 November 2019"



Description: updated version that uses aco-graank and parallel multi-processing

"""

# from joblib import Parallel, delayed
import numpy as np
import multiprocessing as mp
from src.algorithms.ant_colony.aco_grad import GradACO
from src.algorithms.common.dataset import Dataset
#from src.algorithms.ant_colony.cython.cyt_aco_grad import GradACO
#from src.algorithms.common.cython.cyt_dataset import Dataset
from src.algorithms.common.profile_cpu import Profile


class TgradACO:

    def __init__(self, d_set, ref_item, min_sup, min_rep, cores):
        # For tgraank
        self.d_set = d_set
        cols = d_set.time_cols
        if len(cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            self.time_cols = cols
            self.min_sup = min_sup
            self.ref_item = ref_item
            self.max_step = self.get_max_step(min_rep)
            self.orig_attr_data = self.d_set.data.copy().T
            self.cores = cores
            # self.multi_data = self.split_dataset()
        else:
            print("Dataset Error")
            self.time_ok = False
            self.time_cols = []
            raise Exception('No date-time data found')

    def run_tgraank(self, parallel=False):
        if parallel:
            # implement parallel multi-processing
            if self.cores > 1:
                num_cores = self.cores
            else:
                num_cores = Profile.get_num_cores()

            self.cores = num_cores
            steps = range(self.max_step)
            pool = mp.Pool(num_cores)
            patterns = pool.map(self.fetch_patterns, steps)
            # patterns = Parallel(n_jobs=num_cores)(delayed(self.fetch_patterns)(s+1) for s in steps)
            return patterns
        else:
            patterns = list()
            for step in range(self.max_step):
                t_pattern = self.fetch_patterns(step)
                if t_pattern:
                    patterns.append(t_pattern)
            return patterns

    def fetch_patterns(self, step):
        step += 1  # because for-loop is not inclusive from range: 0 - max_step
        # 1. Transform data
        attr_data, time_diffs = self.transform_data(step)

        # 2. Execute aco-graank for each transformation
        self.d_set.update_attributes(attr_data)
        ac = GradACO(self.d_set)
        ac.init_pheromones()
        list_gp = ac.run_ant_colony(self.min_sup, time_diffs)
        # print("\nPheromone Matrix")
        # print(ac.p_matrix)
        if len(list_gp) > 0:
            return list_gp
        return False

    def transform_data(self, step):
        # NB: Restructure dataset based on reference item
        if self.time_ok:
            # 1. Calculate time difference using step
            ok, time_diffs = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs[0]) + " or row " + str(time_diffs[1]) + " is not valid."
                raise Exception(msg)
            else:
                ref_col = self.ref_item
                if ref_col in self.time_cols:
                    msg = "Reference column is a 'date-time' attribute"
                    raise Exception(msg)
                elif (ref_col < 0) or (ref_col >= len(self.d_set.title)):
                    msg = "Reference column does not exist\nselect column between: " \
                          "0 and "+str(len(self.d_set.title) - 1)
                    raise Exception(msg)
                else:
                    # 1. Split the original data-set into column-tuples
                    attr_data = self.orig_attr_data

                    # 2. Transform the data using (row) n+step
                    new_attr_data = list()

                    for k in range(len(attr_data)):
                        col_index = k
                        tuples = attr_data[k]
                        if col_index in self.time_cols:
                            # date-time attribute
                            temp_tuples = tuples[:]
                        elif col_index == ref_col:
                            # reference attribute
                            temp_tuples = tuples[0: tuples.size - step]
                        else:
                            # other attributes
                            temp_tuples = tuples[step: tuples.size]
                        new_attr_data.append(temp_tuples)
                    return new_attr_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_max_step(self, minrep):
        all_rows = len(self.d_set.data)
        return all_rows - int(minrep * all_rows)

    def get_time_diffs(self, step):  # optimized
        data = self.d_set.data
        # x = int(self.time_cols[0])
        # time_data1 = np.array(data[:, x].copy(), dtype='datetime64[s]')  # fetch time data
        # time_data2 = np.empty_like(time_data1)
        # time_data2[:-step] = time_data1[step:]
        # time_data1 = time_data1[0: time_data1.size - step]
        # time_data2 = time_data2[0: time_data2.size - step]
        # time_diffs = np.subtract(time_data2, time_data1)
        size = len(data)
        time_diffs = []
        for i in range(size):
            if i < (size - step):
                # for col in self.time_cols:
                col = self.time_cols[0]  # use only the first date-time value
                temp_1 = str(data[i][int(col)])
                temp_2 = str(data[i + step][int(col)])
                stamp_1 = Dataset.get_timestamp(temp_1)
                stamp_2 = Dataset.get_timestamp(temp_2)
                if (not stamp_1) or (not stamp_2):
                    return False, [i + 1, i + step + 1]
                time_diff = (stamp_2 - stamp_1)
                index = tuple([i, i+step])
                time_diffs.append([time_diff, index])
        return True, np.array(time_diffs)
