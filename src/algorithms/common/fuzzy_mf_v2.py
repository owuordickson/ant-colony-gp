# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent and Joseph Orero"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "20 November 2019"
@modified: "28 May 2020"

"""


import numpy as np
import skfuzzy as fuzzy
from src.algorithms.common.gp import TimeLag


def calculate_time_lag(indices, time_diffs, min_sup):
    stamps = np.array(time_diffs[:, 0])  # get all stamps from 1st column
    time_lags = get_time_lags(indices, time_diffs)
    boundaries, extremes = get_membership_boundaries(stamps)
    time_lag = approximate_fuzzy_support(min_sup, time_lags, boundaries, extremes)
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
    return np.array(time_lags)


def get_membership_boundaries(members):  # optimized
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
    if len(timelags) <= 0:
        # if timelags is blank return nothing
        res = TimeLag()
        return res
    else:
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
                    res = TimeLag(b1, 0)
                    return res


def calculate_support(memberships):  # optimized
    sup_count = np.count_nonzero(memberships > 0)
    total = memberships.size
    support = sup_count / total
    return support


def get_indices(bin_data):  # optimized
    indices = np.argwhere(bin_data == 1)
    return indices

