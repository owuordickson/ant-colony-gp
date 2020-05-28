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
from src.algorithms.common.gp import TimeLag
#from src.algorithms.common.cython.cyt_gp import TimeLag


def calculate_time_lag(indices, time_diffs, minsup):
    stamps = np.array(time_diffs[:, 0])  # get all stamps from 1st column
    arr_timelags = get_time_lags(indices, time_diffs)
    boundaries, extremes = get_membership_boundaries(stamps)
    time_lag = approximate_fuzzy_support(minsup, arr_timelags, boundaries, extremes)
    return time_lag


def get_time_lags(indices, time_diffs):
    pat_indices = set(tuple(map(tuple, indices)))
    time_lags = list()
    for obj in time_diffs:
        index1 = tuple([(obj[1])])
        index2 = tuple([(obj[1][1], obj[1][0])])
        exits1 = pat_indices.intersection(set(index1))
        exits2 = pat_indices.intersection(set(index2))
        if len(exits1) > 0 or len(exits2) > 0:
            time_lags.append(obj[0])
    if len(time_lags) > 0:
        return time_lags
    else:
        raise Exception("Error: No pattern found for fetching time-lags")


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


def approximate_fuzzy_support(minsup, timelags, orig_boundaries, extremes):
    # if timelags is blank return (do not process)
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
        sup = calculate_support(memberships)

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


def calculate_support(memberships):  # optimized
    sup_count = np.count_nonzero(memberships > 0)
    total = memberships.size
    support = sup_count / total
    return support


def get_indices(bin_data):  # optimized
    # indices = []
    # t_rows = len(bin_data)
    # t_columns = len(bin_data[0])
    # for r in range(t_rows):
    #    for c in range(t_columns):
    #        if bin_data[c][r] == 1:
    #            index = [r, c]
    #            indices.append(index)
    #print(bin_data)
    #print("------")
    #print(indices)
    #print("-----")
    indices = np.argwhere(bin_data == 1)
    return indices


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])
