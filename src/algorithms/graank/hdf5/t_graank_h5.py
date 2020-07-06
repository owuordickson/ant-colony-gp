# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent and Joseph Orero"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "19 November 2019"



Description: updated version that uses aco-graank and parallel multi-processing

"""

import numpy as np
from ...common.hdf5.dataset_h5 import Dataset_h5
from ..t_graank import Tgrad
from ...common.profile_cpu import Profile
from .graank_h5 import graank_h5


class Tgrad_5(Tgrad):

    def __init__(self, f_path, eq, ref_item, min_sup, min_rep, cores):
        # For tgraank
        # self.d_set = d_set
        self.d_set = Dataset_h5(f_path, min_sup=min_sup, eq=eq)
        self.d_set.init_h5_groups()
        cols = self.d_set.time_cols
        if len(cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            self.time_cols = cols
            self.min_sup = min_sup
            self.ref_item = ref_item
            self.d_set.data = self.d_set.read_h5_dataset('dataset/data')
            self.d_set.data = np.array(self.d_set.data).astype('U')
            self.max_step = self.get_max_step(min_rep)
            self.orig_attr_data = self.d_set.data.copy().T
            if cores > 1:
                self.cores = cores
            else:
                self.cores = Profile.get_num_cores()
        else:
            print("Dataset Error")
            self.time_ok = False
            self.time_cols = []
            raise Exception('No date-time data found')

    def fetch_patterns(self, step):
        step += 1  # because for-loop is not inclusive from range: 0 - max_step
        # 1. Transform data
        d_set = self.d_set
        attr_data, time_diffs = self.transform_data(step)

        # 2. Execute t-graank for each transformation
        d_set.update_gp_attributes(attr_data)
        tgps = graank_h5(t_diffs=time_diffs, d_set=d_set)

        if len(tgps) > 0:
            return tgps
        return False
