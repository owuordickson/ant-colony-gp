# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "23 June 2020"

Changes
-------
1. This frees primary memory from storing nx1 matrices
2. Fetch all binaries during initialization
3. Replaced loops for fetching binary rank with numpy function

"""
import csv
from dateutil.parser import parse
import time
import numpy as np
import pandas as pd
import gc


class Dataset:

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
            self.attr_size = 0
            self.step_name = ''
            self.thd_supp = min_sup
            self.equal = eq
            data = None
            self.cost_matrix = np.ones((self.column_size, 3), dtype=int)
            self.encoded_data = np.array([])
            # self.start_node = []
            # self.init_attributes()

    def get_size(self):
        size = self.data.shape[0]
        return size

    def get_attribute_no(self):
        count = self.data.shape[1]
        return count

    def get_title(self, data):
        # data = self.raw_data
        if data[0][0].replace('.', '', 1).isdigit() or data[0][0].isdigit():
            title = self.convert_data_to_array(data)
            return title
        else:
            if data[0][1].replace('.', '', 1).isdigit() or data[0][1].isdigit():
                title = self.convert_data_to_array(data)
                return title
            else:
                title = self.convert_data_to_array(data, has_title=True)
                return title

    def convert_data_to_array(self, data, has_title=False):
        # convert csv data into array
        title = np.array([])
        if has_title:
            keys = np.arange(len(data[0]))
            values = np.array(data[0], dtype='S')
            title = np.rec.fromarrays((keys, values), names=('key', 'value'))
            data = np.delete(data, 0, 0)
        # convert csv data into array
        self.data = np.asarray(data)
        return title

    def get_attributes(self):
        all_cols = np.arange(self.get_attribute_no())
        # attr_cols = np.delete(all_cols, self.time_cols)
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def get_time_cols(self):
        time_cols = list()
        # for k in range(10, len(self.data[0])):
        #    time_cols.append(k)
        # time_cols.append(0)
        n = len(self.data[0])
        for i in range(n):  # check every column for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = Dataset.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        if len(time_cols) > 0:
            return np.array(time_cols)
        else:
            return np.array([])

    def init_gp_attributes(self):
        # (check) implement parallel multiprocessing
        # transpose csv array data
        attr_data = self.data.copy().T
        self.attr_size = len(attr_data[self.attr_cols[0]])
        # construct and store 1-item_set valid bins
        # attr_data = attr_data[self.attr_cols]
        self.construct_bins_v3(attr_data)
        attr_data = None
        gc.collect()

    def update_gp_attributes(self, attr_data):
        self.attr_size = len(attr_data[self.attr_cols[0]])
        # self.construct_bins_v1(attr_data)
        self.construct_bins(attr_data)
        gc.collect()

    def construct_bins_v3(self, attr_data):
        # 1. Encoding data for Depth-First Search
        # [row_i, row_j, ..., data, ...]
        self.encoded_data = np.array(self.encode_data_v3(attr_data))
        # print(self.attr_cols)
        # print(self.encoded_data)

        # 2. Data set reduction
        # self.reduce_data()
        # print(self.encoded_data.shape)
        #print(self.attr_cols)
        #print(self.encoded_data)
        #print(self.cost_matrix)
        # print(self.start_node)
        #print("\n\n")

    def encode_data_v3(self, attr_data):
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
                row = np.where(row_js > row_in, 1, np.where(row_js < row_in, -1, 0))
                temp_arr[k] = row
                k += 1
                pos_cost = np.count_nonzero(row == 1)
                neg_cost = np.count_nonzero(row == -1)
                inv_cost = np.count_nonzero(row == 0)
                self.cost_matrix[col][0] += (neg_cost + inv_cost)
                self.cost_matrix[col][1] += (pos_cost + inv_cost)
                self.cost_matrix[col][2] += (pos_cost + neg_cost)
            encoded_data.extend(temp_arr.T)
        gc.collect()
        return encoded_data

    def construct_bins_v2(self, attr_data):
        # Encoding data for Depth-First Search
        self.encoded_data = np.array(self.encode_data_v2(attr_data))
        print(self.encoded_data)
        print(self.cost_matrix)
        # print(self.start_node)
        print("\n\n")

    def encode_data_v2(self, attr_data):
        size = self.attr_size  # np.arange(self.attr_size)
        encoded_data = list()
        for i in range(size):
            if (i+1) >= size:
                continue
            temp_d = list()
            for col in self.attr_cols:
                row_in = attr_data[col][i]
                row_js = attr_data[col][(i+1):size]
                row = np.where(row_js > row_in, 1, np.where(row_js < row_in, -1, 0))
                temp_d.append(row)
                pos_cost = np.count_nonzero(row == 1)
                neg_cost = np.count_nonzero(row == -1)
                inv_cost = np.count_nonzero(row == 0)
                self.cost_matrix[col][0] += (neg_cost + inv_cost)
                self.cost_matrix[col][1] += (pos_cost + inv_cost)
                self.cost_matrix[col][2] += (pos_cost + neg_cost)
            # temp_arr = np.array(temp_d)
            # pos_cost = np.count_nonzero(temp_arr == 1)
            # neg_cost = np.count_nonzero(temp_arr == -1)
            # print([i, [pos_cost, neg_cost]])
            # if len(self.start_node) <= 0:
            #    self.start_node = [i, [pos_cost, neg_cost]]
            # elif (pos_cost > self.start_node[1][0] and neg_cost > 0) or\
            #        (neg_cost > self.start_node[1][1] and pos_cost > 0):
            #    self.start_node = [i, [pos_cost, neg_cost]]
            encoded_data.append([self.attr_cols, np.array(temp_d).T])
        gc.collect()
        return encoded_data
        # return self.update_cost_v2(encoded_data)

    def update_cost_v2(self, encoded_data):
        size = self.attr_size
        cost_data = []
        for obj in encoded_data:
            new_rows = list()
            rows = list(obj[1])
            for j in range(len(rows)):
                cost = 0
                for i in range(len(obj[0])):
                    col_id = obj[0][i]
                    cell = rows[j][i]
                    if cell == 1:
                        cost += self.cost_matrix[col_id][0]
                    elif cell == -1:
                        cost += self.cost_matrix[col_id][1]
                if cost > 0:
                    cost = size / cost
                else:
                    cost = 1
                new_rows.append([rows[j], cost])
            cost_data.append([obj[0], np.array(new_rows)])
        return cost_data

    def construct_bins_v1(self, attr_data):
        # Encoding data for Depth-First Search
        encode_type = np.dtype([('id', 'i'),
                                ('seq', 'i, i'),
                                ('pattern', [('col', 'i'),
                                             ('var', 'i')],
                                 (len(self.attr_cols),)),
                                ('cost', 'f')
                                ])
        # self.encoded_data = np.array(self.encode_data_v1(attr_data), dtype=encode_type)
        self.encoded_data = np.array(self.encode_data_v1(attr_data))
        print(self.encoded_data)
        print(self.cost_matrix)
        gc.collect()

    def encode_data_v1(self, attr_data):
        size = self.attr_size  # np.arange(self.attr_size)
        encoded_data = list()
        test_d = list()
        for i in range(size):
            for j in range(i+1, size):
                gp = []
                for attr in np.nditer(self.attr_cols):
                    col = int(attr)
                    row_i = attr_data[col][i]
                    row_j = attr_data[col][j]
                    if row_i < row_j:
                        # gp.append(np.array((col, 1), dtype='i, i'))
                        gp.append(tuple([col, 1]))
                        self.cost_matrix[col][0] += 1
                    elif row_i > row_j:
                        # gp.append(np.array((col, -1), dtype='i, i'))
                        gp.append(tuple([col, -1]))
                        self.cost_matrix[col][1] += 1
                    else:
                        # gp.append(np.array((col, 0), dtype='i, i'))
                        gp.append(tuple([col, 0]))
                encoded_data.append([(i, (i, j), gp)])
        return self.update_cost_v1(encoded_data)
        # return encoded_data

    def update_cost_v1(self, encoded_data):
        # encoded_data = list(encoded_data)
        size = self.attr_size
        # for obj in self.encoded_data:
        for k in range(len(encoded_data)):
            # gp = obj['pattern'][0]
            gp = encoded_data[k][0][2]
            cost = 0
            for gi in gp:
                if gi[1] == 1:
                    cost += self.cost_matrix[gi[0]][0]
                elif gi[1] == -1:
                    cost += self.cost_matrix[gi[0]][1]
            temp = list(encoded_data[k][0])
            if cost > 0:
                cost = size / cost
                # obj['cost'][0] = cost
                temp.append(cost)  # = cost
            else:
                temp.append(1)
            encoded_data[k][0] = tuple(temp)
        return encoded_data

    def construct_bins(self, attr_data):
        # execute binary rank to calculate support of pattern
        # valid_bins = list()  # numpy is very slow for append operations
        n = self.attr_size
        valid_idxs = list()
        invalid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')
            # temp_pos = Dataset.bin_rank(col_data, equal=self.equal)
            temp_pos, supp = Dataset.index_rank(col_data, n)
            # supp = float(len(temp_pos) / n)

            if supp < self.thd_supp:
                invalid_bins.append(incr)
                invalid_bins.append(decr)
            else:
                temp_neg = temp_pos[::-1]
                valid_idxs.append(np.array([incr.tolist(), temp_pos]))
                valid_idxs.append(np.array([decr.tolist(), temp_neg]))
        self.valid_idxs = np.array(valid_idxs)
        self.invalid_bins = np.array(invalid_bins)

    @staticmethod
    def index_rank(arr, n):
        sort_idx = np.argsort(arr)
        # index = np.arange(len(arr))
        # values, cnt = np.unique(arr, return_counts=True)
        # temp_idx = np.split(index[sort_idx], np.cumsum(cnt[:-1]))
        # temp_idx = np.unique(arr)  # slower than pandas
        temp_idx = pd.unique(arr)  # faster O(N)
        supp = float(len(temp_idx) / n)
        # print(str(arr) + ' - ' + str(temp_idx) + ' : ' + str(supp) + '\n')
        return sort_idx, supp

    @staticmethod
    def read_csv(file):
        # 1. retrieve data-set from file
        with open(file, 'r') as f:
            dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,' '\t")
            f.seek(0)
            reader = csv.reader(f, dialect)
            temp = list(reader)
            f.close()
        return temp

    @staticmethod
    def test_time(date_str):
        # add all the possible formats
        try:
            if type(int(date_str)):
                return False, False
        except ValueError:
            try:
                if type(float(date_str)):
                    return False, False
            except ValueError:
                try:
                    date_time = parse(date_str)
                    t_stamp = time.mktime(date_time.timetuple())
                    return True, t_stamp
                except ValueError:
                    raise ValueError('no valid date-time format found')

    @staticmethod
    def get_timestamp(time_data):
        try:
            ok, stamp = Dataset.test_time(time_data)
            if ok:
                return stamp
            else:
                return False
        except ValueError:
            return False
