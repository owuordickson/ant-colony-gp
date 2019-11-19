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

    def __init__(self, d_set, ref_item, min_sup, min_rep):
        # For tgraank
        self.d_set = d_set
        cols = d_set.get_time_cols()
        if len(cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            self.time_cols = cols
            self.min_sup = min_sup
            self.ref_item = ref_item
            self.max_step = self.get_max_step(min_rep)
            # self.multi_data = self.split_dataset()
        else:
            print("Dataset Error")
            self.time_ok = False
            self.time_cols = []
            raise Exception('No date-time data found')

    def run_tgraank(self):
        # implement parallel multi-processing
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
                ref_col = self.ref_item
                if ref_col in self.time_cols:
                    msg = "Reference column is a 'date-time' attribute"
                    raise Exception(msg)
                elif (ref_col < 0) or (ref_col >= len(self.d_set.title)):
                    msg = "Reference column does not exist\nselect column between: " \
                          "0 and "+str(len(self.d_set.title) - 1)
                    raise Exception(msg)
                else:
                    # 1. Split the original data-set into column-tuples
                    attr_cols = self.d_set.attr_data

                    # 2. Transform the data using (row) n+step
                    new_data = list()
                    size = len(data)
                    for obj in attr_cols:
                        col_index = int(obj[0])
                        tuples = obj[1]
                        temp_tuples = list()
                        if (col_index -1) == ref_col:
                            # reference attribute (skip)
                            for i in range(0, (size-step)):
                                temp_tuples.append(tuples[i])
                        else:
                            for i in range(step, size):
                                temp_tuples.append(tuples[i])
                        var_attr = [col_index, temp_tuples]
                        new_data.append(var_attr)
                    return new_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_representativity(self, step):
        # 1. Get all rows minus the title row (already removed)
        all_rows = len(self.d_set.data)

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
        size = len(self.d_set.data)
        for i in range(size):
            check, info = self.get_representativity(i + 1)
            if check:
                rep = info['Representativity']
                if rep < minrep:
                    return i
            else:
                return 0

    def get_time_diffs(self, step):
        data = self.d_set.data
        size = len(data)
        time_diffs = []
        for i in range(size):
            if i < (size - step):
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