# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "12 July 2019"

"""
import csv
from dateutil.parser import parse
import time
import numpy as np
from numba import prange
import numba


class Dataset:

    def __init__(self, file_path):
        data = Dataset.read_csv(file_path)
        if len(data) == 0:
            self.data = np.array([])
            print("csv file read error")
            raise Exception("Unable to read csv file")
        else:
            print("Data fetched from csv file")
            self.data = np.array([])
            self.title = self.get_title(data)  # optimized (numpy)
            self.time_cols = self.get_time_cols()  # optimized (numpy)
            self.attr_cols = self.get_attributes()  # optimized (numpy)
            self.column_size = self.get_attribute_no()  # optimized (cdef)
            self.size = self.get_size()  # optimized (cdef)
            self.thd_supp = False
            self.equal = False
            self.attr_data = np.array([])  # optimized (numpy)
            # self.lst_bin = []
            self.arr_bins = np.array([])  # optimized (numpy & numba)

    def get_size(self):
        size = self.data.shape[0]
        if self.title.size > 0:
            size += 1
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
        if has_title:
            keys = np.arange(len(data[0]))
            values = data[0]
            title = np.rec.fromarrays((keys, values), names=('key', 'value'))
            del data[0]
            # convert csv data into array
            self.data = np.asarray(data)
            return np.array(title)
        else:
            self.data = np.asarray(data)
            return np.array([])

    def get_attributes(self):
        #keys = np.array(self.title.key, dtype=int)
        all_cols = np.arange(self.get_attribute_no())
        attr_cols = np.delete(all_cols, self.time_cols)
        return attr_cols

    def get_time_cols(self):
        time_cols = np.array([])
        # for k in range(10, len(self.data[0])):
        #    time_cols.append(k)
        # time_cols.append(0)
        #time_cols.append(1)
        # time_cols.append(2)
        # time_cols.append(3)
        # time_cols.append(4)
        # time_cols.append(5)
        # time_cols.append(6)
        # time_cols.append(7)
        # time_cols.append(8)
        # time_cols.append(9)
        for i in range(len(self.data[0])):  # check every column for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = Dataset.test_time(row_data)
                if time_ok:
                    if time_cols.size > 0:
                        time_cols = np.hstack((time_cols, i))
                    else:
                        time_cols = np.array([i])
            except ValueError:
                continue
        if len(time_cols) > 0:
            return np.array(time_cols)
        else:
            return np.array([])

    def init_attributes(self, eq):
        # (check) implement parallel multiprocessing
        # transpose csv array data
        self.equal = eq
        self.attr_data = np.transpose(self.data)

    def get_bin_rank(self, attr_data, symbol):
        # execute binary rank to calculate support of pattern
        key = attr_data[0]
        data = attr_data[1]
        incr = tuple([key, '+'])
        decr = tuple([key, '-'])
        n = len(data)
        temp = np.zeros((n, n), dtype='bool')
        temp_pos, temp_neg = Dataset.bin_rank(data, n, temp, equal=self.equal)

        if symbol == '+':
            temp_bin = temp_pos
        elif symbol == '-':
            temp_bin = temp_neg
        else:
            temp_bin = np.array([])
        supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)

        if self.arr_bins.size > 0:
            self.arr_bins = np.vstack((self.arr_bins, np.array([[incr, temp_pos, supp]])))
            self.arr_bins = np.vstack((self.arr_bins, np.array([[decr, temp_neg, supp]])))
        else:
            self.arr_bins = np.array([[incr, temp_pos, supp]])
            self.arr_bins = np.vstack((self.arr_bins, np.array([[decr, temp_neg, supp]])))
        return supp, temp_bin

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def bin_rank(arr, n, temp_pos, equal=False):
        if not equal:
            for i in prange(n):
                for j in prange(i+1, n, 1):
                    temp_pos[i, j] = arr[i] > arr[j]
                    temp_pos[j, i] = arr[i] < arr[j]
        else:
            for i in prange(n):
                for j in prange(i+1, n, 1):
                    temp_pos[i, j] = arr[i] >= arr[j]
                    temp_pos[j, i] = arr[i] < arr[j]
        temp_neg = np.transpose(temp_pos)
        return temp_pos, temp_neg

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
    def write_file(data, path):
        with open(path, 'w') as f:
            f.write(data)
            f.close()

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

    @staticmethod
    def format_gp(obj_gp):
        arr_gp = list(obj_gp)
        new_gp = list()
        for item in arr_gp:
            attr = int(item[0])
            sign = item[1]
            str_gp = str(attr) + sign
            new_gp.append(str_gp)
        return set(new_gp)

