"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""
import csv
from dateutil.parser import parse
import time


class DataSet:

    def __init__(self, file_path):
        raw_data = DataSet.read_csv(file_path)

    def get_attribute_no(self):
        length = len(self.raw_data[0])
        return length

    def get_time_cols(self):
        time_cols = list()
        for i in range(len(self.raw_data[1])):  # check every column for time format
            row_data = str(self.raw_data[1][i])
            try:
                time_ok, t_stamp = DataSet.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue

        if time_cols:
            return time_cols
        else:
            return False

    def separate_columns(self):
        return self.raw_data

    @staticmethod
    def read_csv(file):
        # 1. retrieve data-set from file
        with open(file, 'r') as f:
            dialect = csv.Sniffer().sniff(f.read(1024), delimiters=";,' '\t")
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
