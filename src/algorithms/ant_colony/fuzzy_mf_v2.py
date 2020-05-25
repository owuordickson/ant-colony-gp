# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent and Joseph Orero"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "20 November 2019"
@modified: "28 March 2019"

"""


import numpy as np
import skfuzzy as fuzzy


class TimeLag:

    def __init__(self, tstamp=0, supp=0):
        self.timestamp = tstamp
        self.support = supp
        self.sign = self.get_sign()
        if tstamp == 0:
           self.timelag = np.array([])
        else:
            self.timelag = np.array(self.format_time())

    def get_sign(self):
        if self.timestamp < 0:
            sign = "-"
        else:
            sign = "+"
        return sign

    def format_time(self):
        stamp_in_seconds = abs(self.timestamp)
        years = stamp_in_seconds / 3.154e+7
        months = stamp_in_seconds / 2.628e+6
        weeks = stamp_in_seconds / 604800
        days = stamp_in_seconds / 86400
        hours = stamp_in_seconds / 3600
        minutes = stamp_in_seconds / 60
        if int(years) <= 0:
            if int(months) <= 0:
                if int(weeks) <= 0:
                    if int(days) <= 0:
                        if int(hours) <= 0:
                            if int(minutes) <= 0:
                                return [stamp_in_seconds, "seconds"]
                            else:
                                return [minutes, "minutes"]
                        else:
                            return [hours, "hours"]
                    else:
                        return [days, "days"]
                else:
                    return [weeks, "weeks"]
            else:
                return [months, "months"]
        else:
            return [years, "years"]

    def print_lag(self):
        txt = ("~ " + self.sign + str(self.timelag[0]) + " " + str(self.timelag[1])
               + " : " + str(self.support))
        return txt


class FuzzyMF:

    @staticmethod
    def init_fuzzy_support(test_members, all_members, minsup):
        boundaries, extremes = FuzzyMF.get_membership_boundaries(all_members)
        t_lag = FuzzyMF.approximate_fuzzy_support(minsup, test_members, boundaries, extremes)
        return t_lag

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
            sup = FuzzyMF.calculate_support(memberships)

            if sup >= minsup:
                # value = FuzzyMF.get_time_format(b)
                # return b, value, sup
                res = TimeLag(b, sup)
                return res
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
                    # value = FuzzyMF.get_time_format(b1)
                    # return b1, value, False
                    res = TimeLag(b1, 0)
                    return res

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
    def calculate_time_lag(indices, time_diffs, minsup):
        arr_timelags = FuzzyMF.get_time_lags(indices, time_diffs)
        time_lag = FuzzyMF.init_fuzzy_support(arr_timelags, time_diffs, minsup)
        return time_lag

    @staticmethod
    def get_patten_indices(D):
        indices = []
        t_rows = len(D)
        t_columns = len(D[0])
        for r in range(t_rows):
            for c in range(t_columns):
                if D[c][r] == 1:
                    index = [r, c]
                    indices.append(index)
        return indices

    @staticmethod
    def get_time_lags(indices, time_diffs):
        if len(indices) > 0:
            indxs = np.unique(indices[0])
            time_lags = []
            for i in indxs:
                if (i >= 0) and (i < len(time_diffs)):
                    time_lags.append(time_diffs[i])
            return time_lags
        else:
            raise Exception("Error: No pattern found for fetching time-lags")
