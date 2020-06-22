# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "3.0"
@email: "owuordickson@gmail.com"
@created: "22 June 2020"

"""

import numpy as np
from ..common.dataset_mp import Dataset_mp
from .aco_grad import GradACO


class GradACO_mp(GradACO):

    def __init__(self, f_path, min_supp, eq, cores):
        self.d_set = Dataset_mp(f_path, min_supp, eq, cores)
        self.d_set.init_attributes()
        self.attr_index = self.d_set.attr_cols
        # self.e_factor = 0.1  # evaporation factor
        self.p_matrix = np.ones((self.d_set.column_size, 3), dtype=float)
