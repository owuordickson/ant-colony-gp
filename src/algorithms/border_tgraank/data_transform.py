"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

This class prepares a dataset to be transformed by any step

"""

import csv
import numpy as np
from dateutil.parser import parse
import time


class DataTransform:

    def __init__(self, filename, ref_item, min_rep):
        # 1. Test dataset
        cols, data = DataTransform.test_dataset(filename)

        if cols:
            print("Dataset Ok")
            self.time_ok = True
            self.time_cols = cols
            self.ref_item = ref_item
            self.data = data
            self.max_step = self.get_max_step(min_rep)
            self.multi_data = self.split_dataset()
        else:
            print("Dataset Error")
            self.time_ok = False
            self.time_cols = []
            raise Exception('No date-time data found')

    def split_dataset(self):
        # NB: Creates an (array) item for each column
        # NB: ignore first row (titles) and date-time columns
        # 1. get no. of columns (ignore date-time columns)
        no_columns = (len(self.data[0]) - len(self.time_cols))

        # 2. Create arrays for each gradual column item
        multi_data = [None] * no_columns
        i = 0
        for c in range(len(self.data[0])):
            if c in self.time_cols:
                continue  # skip columns with time
            else:
                multi_data[i] = []
                for r in range(1, len(self.data)):  # ignore title row
                    item = self.data[r][c]
                    multi_data[i].append(item)
                i += 1
        return multi_data

    def transform_data(self, step):
        # NB: Restructure dataset based on reference item
        if self.time_ok:
            # 1. Calculate time difference using step
            ok, time_diffs = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs[0]) + " or row " + str(time_diffs[1]) + " is not valid."
                raise Exception(msg)
            else:
                ref_column = self.ref_item
                # 1. Load all the titles
                first_row = self.data[0]

                # 2. Creating titles without time column
                no_columns = (len(first_row) - len(self.time_cols))
                title_row = [None] * no_columns
                i = 0
                for c in range(len(first_row)):
                    if c in self.time_cols:
                        continue
                    title_row[i] = first_row[c]
                    i += 1

                ref_name = str(title_row[ref_column])
                title_row[ref_column] = ref_name + "**"
                new_dataset = [title_row]

                # 3. Split the original dataset into gradual items
                gradual_items = self.multi_data

                # 4. Transform the data using (row) n+step
                for j in range(len(self.data)):
                    ref_item = gradual_items[ref_column]

                    if j < (len(ref_item) - step):
                        init_array = [ref_item[j]]

                        for i in range(len(gradual_items)):
                            if i < len(gradual_items) and i != ref_column:
                                gradual_item = gradual_items[i];
                                temp = [gradual_item[j + step]]
                                temp_array = np.append(init_array, temp, axis=0)
                                init_array = temp_array
                        new_dataset.append(list(init_array))
                return new_dataset, time_diffs;
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_representativity(self, step):
        # 1. Get all rows minus the title row
        all_rows = (len(self.data) - 1)

        # 2. Get selected rows
        incl_rows = (all_rows - step)

        # 3. Calculate representativity
        if incl_rows > 0:
            rep = (incl_rows / float(all_rows))
            info = {"Transformation": "n+"+str(step), "Representativity": rep, "Included Rows": incl_rows,
                    "Total Rows": all_rows}
            return True, info
        else:
            return False, "Representativity is 0%"

    def get_max_step(self, minrep):
        # 1. count the number of steps each time comparing the
        # calculated representativity with minimum representativity
        for i in range(len(self.data)):
            check, info = self.get_representativity(i + 1)
            if check:
                rep = info['Representativity']
                if rep < minrep:
                    return i
            else:
                return 0

    def get_time_diffs(self, step):
        time_diffs = []
        for i in range(1, len(self.data)):
            if i < (len(self.data) - step):
                # temp_1 = self.data[i][0]
                # temp_2 = self.data[i + step][0]
                temp_1 = temp_2 = ""
                for col in self.time_cols:
                    temp_1 += " "+str(self.data[i][int(col)])
                    temp_2 += " "+str(self.data[i + step][int(col)])

                stamp_1 = DataTransform.get_timestamp(temp_1)
                stamp_2 = DataTransform.get_timestamp(temp_2)

                if (not stamp_1) or (not stamp_2):
                    return False, [i + 1, i + step + 1]

                time_diff = (stamp_2 - stamp_1)
                time_diffs.append(time_diff)
        #print("Time Diff: " + str(time_diff))
        return True, time_diffs

    @staticmethod
    def test_dataset(filename):
        # NB: test the dataset attributes: time|item_1|item_2|...|item_n
        # return true and (list) dataset if it is ok
        # 1. retrieve dataset from file
        with open(filename, 'r') as f:
            dialect = csv.Sniffer().sniff(f.read(1024), delimiters=";,' '\t")
            f.seek(0)
            reader = csv.reader(f, dialect)
            temp = list(reader)
            f.close()

        # 2. Retrieve time and their columns
        time_cols = list()
        for i in range(len(temp[1])):  # check every column for time format
            row_data = str(temp[1][i])
            try:
                time_ok, t_stamp = DataTransform.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue

        if time_cols:
            return time_cols, temp
        else:
            return False, temp

    @staticmethod
    def get_timestamp(time_data):
        try:
            ok, stamp = DataTransform.test_time(time_data)
            if ok:
                return stamp
            else:
                return False
        except ValueError:
            return False

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
