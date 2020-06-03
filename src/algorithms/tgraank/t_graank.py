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
from src.algorithms.common.dataset import Dataset
from src.algorithms.common.profile_cpu import Profile
from src.algorithms.graank.graank_v2 import graank


class Tgrad:

    def __init__(self, f_path, eq, ref_item, min_sup, min_rep, cores):
        # For tgraank
        # self.d_set = d_set
        self.d_set = Dataset(f_path, min_sup=min_sup, eq=eq, init=False)
        cols = self.d_set.time_cols
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
        d_set = self.d_set
        attr_data, time_diffs = self.transform_data(step)

        # 2. Execute t-graank for each transformation
        d_set.update_attributes(attr_data)
        D1, S1, T1 = graank(min_sup=self.min_sup, eq=False, t_diffs=time_diffs, d_set=d_set, step=step)
        d_set.clean_memory()
        if len(D1) > 0:
            return [D1, S1, T1]
        return False

    def transform_data(self, step):  # optimized
        # NB: Restructure dataset based on reference item
        if self.time_ok:
            # 1. Calculate time difference using step
            ok, time_diffs = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs[0]) \
                      + " or row " + str(time_diffs[1]) + " is not valid."
                raise Exception(msg)
            else:
                ref_col = self.ref_item
                if ref_col in self.time_cols:
                    msg = "Reference column is a 'date-time' attribute"
                    raise Exception(msg)
                elif (ref_col < 0) or (ref_col >= len(self.d_set.title)):
                    msg = "Reference column does not exist\nselect column between: " \
                          "0 and " + str(len(self.d_set.title) - 1)
                    raise Exception(msg)
                else:
                    # 1. Split the transpose data set into column-tuples
                    attr_data = self.orig_attr_data

                    # 2. Transform the data using (row) n+step
                    new_attr_data = list()
                    size = len(attr_data)
                    for k in range(size):
                        col_index = k
                        tuples = attr_data[k]
                        n = tuples.size
                        temp_tuples = np.empty(n, )
                        temp_tuples[:] = np.NaN
                        if col_index in self.time_cols:
                            # date-time attribute
                            temp_tuples = tuples[:]
                        elif col_index == ref_col:
                            # reference attribute
                            temp_tuples[:n - step] = tuples[0: n - step]
                        else:
                            # other attributes
                            temp_tuples[:n - step] = tuples[step: n]
                        # print(temp_tuples)
                        new_attr_data.append(temp_tuples)
                    return new_attr_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_max_step(self, min_rep):  # optimized
        all_rows = len(self.d_set.data)
        return all_rows - int(min_rep * all_rows)

    def get_time_diffs(self, step):  # optimized
        data = self.d_set.data
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
                index = tuple([i, i + step])
                time_diffs.append([time_diff, index])
        return True, np.array(time_diffs)
