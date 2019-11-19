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


import skfuzzy as fuzzy
import numpy as np
from src import HandleData


class TgradACO:

    def __init__(self, d_set, ref_item, min_rep):
        # For tgraank
        self.d_set = d_set
        cols = d_set.get_time_cols()
        if len(cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            self.time_cols = cols
            self.ref_item = ref_item
            self.max_step = self.get_max_step(min_rep)
            self.multi_data = self.split_dataset()
        else:
            print("Dataset Error")
            self.time_ok = False
            self.time_cols = []
            raise Exception('No date-time data found')

    def run_tgraank(self, min_sup, ref_col, rep):
        data = self.d_set.data
        patterns = list()
        return patterns

    def transform_data(self, step):
        # NB: Restructure dataset based on reference item
        data = self.d_set.data
        if self.time_ok:
            # 1. Calculate time difference using step
            ok, time_diffs = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs[0]) + " or row " + str(time_diffs[1]) + " is not valid."
                raise Exception(msg)
            else:
                ref_column = self.ref_item
                # 1. Load all the titles
                first_row = data[0]

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
                for j in range(len(data)):
                    ref_item = gradual_items[ref_column]
                    if j < (len(ref_item) - step):
                        init_array = [ref_item[j]]
                        for i in range(len(gradual_items)):
                            if i < len(gradual_items) and i != ref_column:
                                gradual_item = gradual_items[i]
                                temp = [gradual_item[j + step]]
                                temp_array = np.append(init_array, temp, axis=0)
                                init_array = temp_array
                        new_dataset.append(list(init_array))
                return new_dataset, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_representativity(self, step):
        # 1. Get all rows minus the title row
        all_rows = (len(self.d_set.data) - 1)

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
        size = self.d_set.data
        for i in range(len(size)):
            check, info = self.get_representativity(i + 1)
            if check:
                rep = info['Representativity']
                if rep < minrep:
                    return i
            else:
                return 0

    def get_time_diffs(self, step):
        data = self.d_set.data
        time_diffs = []
        for i in range(1, len(data)):
            if i < (len(data) - step):
                # temp_1 = self.data[i][0]
                # temp_2 = self.data[i + step][0]
                temp_1 = temp_2 = ""
                for col in self.time_cols:
                    temp_1 += " "+str(data[i][int(col)])
                    temp_2 += " "+str(data[i + step][int(col)])
                stamp_1 = HandleData.get_timestamp(temp_1)
                stamp_2 = HandleData.get_timestamp(temp_2)
                if (not stamp_1) or (not stamp_2):
                    return False, [i + 1, i + step + 1]
                time_diff = (stamp_2 - stamp_1)
                time_diffs.append(time_diff)
        # print("Time Diff: " + str(time_diff))
        return True, time_diffs

    def split_dataset(self):
        # NB: Creates an (array) item for each column
        # NB: ignore first row (titles) and date-time columns
        # 1. get no. of columns (ignore date-time columns)
        data = self.d_set.data
        no_columns = (len(data[0]) - len(self.time_cols))

        # 2. Create arrays for each gradual column item
        multi_data = [None] * no_columns
        i = 0
        for c in range(len(data[0])):
            multi_data[i] = []
            for r in range(1, len(data)):  # ignore title row
                if c in self.time_cols:
                    continue  # skip columns with time
                else:
                    item = data[r][c]
                    multi_data[i].append(item)
            if not (c in self.time_cols):
                i += 1
        return multi_data

    @staticmethod
    def init_fuzzy_support(test_members, all_members, minsup):
        boundaries, extremes = TgradACO.get_membership_boundaries(all_members)
        value, sup = TgradACO.approximate_fuzzy_support(minsup, test_members, boundaries, extremes)
        return value, sup

    @staticmethod
    def get_membership_boundaries(members):
        # 1. Sort the members in ascending order
        members.sort()

        # 2. Get the boundaries of membership function
        min_ = np.min(members)
        q_1 = np.percentile(members, 25)  # Quartile 1
        med = np.percentile(members, 50)
        q_3 = np.percentile(members, 75)
        max_ = np.max(members)
        boundaries = [q_1, med, q_3]
        extremes = [min_, max_]
        return boundaries, extremes

    @staticmethod
    def approximate_fuzzy_support(minsup, timelags, orig_boundaries, extremes):
        slice_gap = (0.1 * int(orig_boundaries[1]))
        sup = sup1 = 0
        slide_left = slide_right = expand = False
        sample = np.percentile(timelags, 50)

        a = orig_boundaries[0]
        b = b1 = orig_boundaries[1]
        c = orig_boundaries[2]
        min_a = extremes[0]
        max_c = extremes[1]
        boundaries = np.array(orig_boundaries)
        time_lags = np.array(timelags)

        while sup <= minsup:
            if sup > sup1:
                sup1 = sup
                b1 = b

            # Calculate membership of frequent path
            memberships = fuzzy.membership.trimf(time_lags, boundaries)

            # Calculate support
            sup = TgradACO.calculate_support(memberships)

            if sup >= minsup:
                value = TgradACO.get_time_format(b)
                return value, sup
            else:
                if not slide_left:
                    # 7. Slide to the left to change boundaries
                    # if extreme is reached - then slide right
                    if sample <= b:
                        # if min_a >= b:
                        a = a - slice_gap
                        b = b - slice_gap
                        c = c - slice_gap
                        boundaries = np.array([a, b, c])
                    else:
                        slide_left = True
                elif not slide_right:
                    # 8. Slide to the right to change boundaries
                    # if extreme is reached - then slide right
                    if sample >= b:
                        # if max_c <= b:
                        a = a + slice_gap
                        b = b + slice_gap
                        c = c + slice_gap
                        boundaries = np.array([a, b, c])
                    else:
                        slide_right = True
                elif not expand:
                    # 9. Expand quartiles and repeat 5. and 6.
                    a = min_a
                    b = orig_boundaries[1]
                    c = max_c
                    boundaries = np.array([a, b, c])
                    slide_left = slide_right = False
                    expand = True
                else:
                    value = TgradACO.get_time_format(b1)
                    return value, False

    @staticmethod
    def calculate_support(memberships):
        support = 0
        if len(memberships) > 0:
            sup_count = 0
            total = len(memberships)
            for member in memberships:
                # if float(member) > 0.5:
                if float(member) > 0:
                    sup_count = sup_count + 1
            support = sup_count / total
        return support

    @staticmethod
    def get_time_format(value):
        if value < 0:
            sign = "-"
        else:
            sign = "+"
        p_value, p_type = TgradACO.round_time(abs(value))
        p_format = [sign, p_value, p_type]
        return p_format

    @staticmethod
    def round_time(seconds):
        years = seconds / 3.154e+7
        months = seconds / 2.628e+6
        weeks = seconds / 604800
        days = seconds / 86400
        hours = seconds / 3600
        minutes = seconds / 60
        if int(years) <= 0:
            if int(months) <= 0:
                if int(weeks) <= 0:
                    if int(days) <= 0:
                        if int(hours) <= 0:
                            if int(minutes) <= 0:
                                return seconds, "seconds"
                            else:
                                return minutes, "minutes"
                        else:
                            return hours, "hours"
                    else:
                        return days, "days"
                else:
                    return weeks, "weeks"
            else:
                return months, "months"
        else:
            return years, "years"