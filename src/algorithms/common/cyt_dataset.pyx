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
cimport numpy as np
import cython


cdef struct title_struct:
    int key
    char[20] value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef class Dataset:

    cdef dict __dict__
    cdef public int size, column_size, attr_size
    cdef float thd_supp
    cdef int equal
    cdef public np.ndarray time_cols
    cdef public np.ndarray attr_cols
    cdef public np.ndarray title
    cdef public np.ndarray data
    cdef public np.ndarray valid_bins
    cdef public list invalid_bins

    def __init__(self, file_path):
        cdef list data
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
            self.attr_size = 0
            self.thd_supp = 0
            self.equal = False
            self.valid_bins = np.array([])  # optimized (numpy & numba)
            self.invalid_bins = list()

    cdef int get_size(self):
        cdef int size
        size = self.data.shape[0]
        if self.title.size > 0:
            size += 1
        return size

    cdef int get_attribute_no(self):
        cdef int count
        count = self.data.shape[1]
        return count

    cdef np.ndarray get_attributes(self):
        cdef np.ndarray all_cols, attr_cols
        all_cols = np.arange(self.get_attribute_no())
        attr_cols = np.delete(all_cols, self.time_cols)
        return attr_cols

    cdef np.ndarray get_title(self, list data):
        # data = self.raw_data
        if data[0][0].replace('.', '', 1).isdigit() or data[0][0].isdigit():
            return self.convert_data_to_array(data)
        else:
            if data[0][1].replace('.', '', 1).isdigit() or data[0][1].isdigit():
                return self.convert_data_to_array(data)
            else:
                return self.convert_data_to_array(data, has_title=True)

    cdef np.ndarray convert_data_to_array(self, list data, bint has_title=False):
        # convert csv data into array
        cdef int size
        cdef np.ndarray keys
        cdef list values
        if has_title:
            size = len(data[0])
            keys = np.arange(size)
            values = data[0]
            title = np.rec.fromarrays((keys, values), names=('key', 'value'))
            del data[0]
            # convert csv data into array
            self.data = np.asarray(data)
            return np.asarray(title)
        else:
            self.data = np.asarray(data)
            return np.array([])

    cdef np.ndarray get_time_cols(self):
        cdef np.ndarray time_cols
        time_cols = np.array([])
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

    cpdef void init_attributes(self, float min_sup bint eq):
        # (check) implement parallel multiprocessing
        # transpose csv array data
        self.thd_supp = min_sup
        self.equal = eq
        attr_data = np.transpose(self.data)
        self.attr_size = attr_data.shape[1]
        self.get_bins(attr_data)

    cpdef get_bins(self, attr_data):
        cdef int n
        cdef float supp
        cdef np.ndarray col_data, temp, temp_pos, temp_neg
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = tuple([col, '+'])
            decr = tuple([col, '-'])
            n = len(col_data)

            temp = np.zeros((n, n), dtype='bool')
            temp_pos, temp_neg = Dataset.bin_rank(col_data, n, temp, equal=self.equal)
            supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)

            if supp < self.thd_supp:
                self.invalid_bins.append(incr)
                self.invalid_bins.append(decr)
            else:
                if self.valid_bins.size > 0:
                    self.valid_bins = np.vstack((self.valid_bins, np.array([[incr, temp_pos, supp]])))
                    self.valid_bins = np.vstack((self.valid_bins, np.array([[decr, temp_neg, supp]])))
                else:
                    self.valid_bins = np.array([[incr, temp_pos, supp]])
                    self.valid_bins = np.vstack((self.valid_bins, np.array([[decr, temp_neg, supp]])))

    @staticmethod
    # @numba.jit(nopython=True, parallel=True)
    def bin_rank(arr, n, temp_pos, equal=False):
        if not equal:
            for i in range(n):
                for j in range(i+1, n, 1):
                    temp_pos[i, j] = arr[i] > arr[j]
                    temp_pos[j, i] = arr[i] < arr[j]
        else:
            for i in range(n):
                for j in range(i+1, n, 1):
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
