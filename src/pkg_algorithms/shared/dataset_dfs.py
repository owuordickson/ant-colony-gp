# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "06 July 2020"

Changes
-------
1. This frees primary memory from storing nx1 matrices
2. Creates an Encoded Data set

"""

import numpy as np
import gc
from .dataset_bfs import Dataset


class DatasetDFS(Dataset):

    def __init__(self, file_path, min_sup=0, eq=False):
        # super().__init__(file_path, min_sup, eq)
        self.thd_supp = min_sup
        self.equal = eq
        self.titles, self.data = Dataset.read_csv(file_path)
        self.row_count, self.col_count = self.data.shape
        self.time_cols = self.get_time_cols()
        self.attr_cols = self.get_attr_cols()
        self.cost_matrix = np.ones((self.col_count, 3), dtype=int)
        self.no_bins = False
        self.step_name = ''
        self.attr_size = 0
        # self.encoded_data = np.array([])

    def encode_data(self):
        attr_data = self.data.T
        self.attr_size = len(attr_data[self.attr_cols[0]])
        size = self.attr_size
        n = len(self.attr_cols) + 2
        encoded_data = list()
        for i in range(size):
            j = i + 1
            if j >= size:
                continue

            temp_arr = np.empty([n, (size - j)], dtype=int)
            temp_arr[0] = np.repeat(i, (size - j))
            temp_arr[1] = np.arange(j, size)
            k = 2
            for col in self.attr_cols:
                row_in = attr_data[col][i]
                row_js = attr_data[col][(i+1):size]
                v = col + 1
                row = np.where(row_js > row_in, v, np.where(row_js < row_in, -v, 0))
                temp_arr[k] = row
                k += 1
                pos_cost = np.count_nonzero(row == v)
                neg_cost = np.count_nonzero(row == -v)
                inv_cost = np.count_nonzero(row == 0)
                self.cost_matrix[col][0] += (neg_cost + inv_cost)
                self.cost_matrix[col][1] += (pos_cost + inv_cost)
                self.cost_matrix[col][2] += (pos_cost + neg_cost)
            temp_arr = temp_arr.T
            encoded_data.extend(temp_arr)
        self.data = None
        gc.collect()
        return np.array(encoded_data)

    def remove_inv_attrs(self, encoded_data):
        c_matrix = self.cost_matrix
        # 1. remove invalid attributes
        valid_a1 = list()
        valid_a2 = [-2, -1]
        for i in range(len(self.attr_cols)):
            a = self.attr_cols[i]
            valid = (c_matrix[a][0] < c_matrix[a][2]) or \
                    (c_matrix[a][1] < c_matrix[a][2])
            if valid:
                valid_a1.append(i)
                valid_a2.append(i)
        self.attr_cols = self.attr_cols[valid_a1]
        valid_a2 = np.array(valid_a2) + 2
        encoded_data = encoded_data[:, valid_a2]
        return encoded_data
