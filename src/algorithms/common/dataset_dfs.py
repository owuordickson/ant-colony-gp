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


class Dataset_dfs(Dataset):

    def __init__(self, file_path, min_sup=0, eq=False):
        data = Dataset.read_csv(file_path)
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
            self.no_bins = False
            self.attr_size = 0
            self.step_name = ''
            self.thd_supp = min_sup
            self.equal = eq
            data = None
            self.cost_matrix = np.ones((self.column_size, 3), dtype=int)
            self.encoded_data = np.array([])

    def construct_bins(self, attr_data):
        # 1. Encoding data for Depth-First Search
        # [row_i, row_j, ..., data, ...]
        self.encoded_data = np.array(self.encode_data(attr_data))

    def encode_data(self, attr_data):
        size = self.attr_size  # np.arange(self.attr_size)
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
            # node = np.empty([2, (size - j)], dtype=int)
            # node[0] = np.repeat(i, (size - j))
            # node[1] = np.arange(j, size)
            # temp_zip = list(zip(node.T, temp_arr))
            encoded_data.extend(temp_arr)
        gc.collect()
        return encoded_data

    def reduce_data(self, p_matrix=None):
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
            else:
                if p_matrix is None:
                    pass
                else:
                    p_matrix[a][0] = 0
                    p_matrix[a][1] = 0
        self.attr_cols = self.attr_cols[valid_a1]
        valid_a2 = np.array(valid_a2) + 2
        self.encoded_data = self.encoded_data[:, valid_a2]
        # 2. merge similar patterns
        # 2a. get indices
        # vals, inverse, count = np.unique(self.d_set.encoded_data[:, 2:],
        #                                 return_inverse=True,
        #                                 return_counts=True,
        #                                 axis=0)
        # return vals, inverse
