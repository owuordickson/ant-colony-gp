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
from .gp import TimeLag


def calculate_time_lag(bin_data, time_diffs):
    indices = get_indices(bin_data)
    # stamps = np.absolute(np.array(time_diffs[:, 0]))  # get all stamps from 1st column
    time_lags = get_time_lags(indices, time_diffs)
    time_lag = approximate_fuzzy_support_v2(time_lags)
    return time_lag


def get_time_lags(indices, time_diffs):
    pat_indices_flat = np.unique(indices.flatten())
    time_lags = list()
    for obj in time_diffs:
        index1 = obj[1]
        if int(index1) in pat_indices_flat:
            time_lags.append(obj[0])
    return np.array(time_lags)


def approximate_fuzzy_support_v2(time_lags):
    if len(time_lags) <= 0:
        # if time_lags is blank return nothing
        return TimeLag()
    else:
        time_lags = np.absolute(np.array(time_lags))
        min_a = np.min(time_lags)
        max_c = np.max(time_lags)
        count = time_lags.size + 3
        tot_boundaries = np.linspace(min_a/2, max_c+1, num=count)

        sup1 = 0
        center = time_lags[0]
        size = len(tot_boundaries)
        for i in range(0, size, 2):
            if (i+3) <= size:
                boundaries = tot_boundaries[i:i+3:1]
            else:
                boundaries = tot_boundaries[size-3:size:1]
            memberships = fuzzy.membership.trimf(time_lags, boundaries)
            sup = calculate_support(memberships)
            if sup > sup1:
                sup1 = sup
                center = boundaries[1]
            if sup >= 0.5:
                return TimeLag(boundaries[1], sup)
        return TimeLag(center, sup1)


def calculate_support(memberships):  # optimized
    sup_count = np.count_nonzero(memberships > 0)
    total = memberships.size
    support = sup_count / total
    return support


def get_indices(bin_data):  # optimized
    indices = np.argwhere(bin_data == 1)
    return indices

