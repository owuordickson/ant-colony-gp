# -*- coding: utf-8 -*-
"""
@author: Olivier + modif MJL+MR 140316
@created on Fri Jun 12 14:31:16 2015

@modified by D. Owuor 04 June 2020

"""

import numpy as np
import gc
from ..common.fuzzy_mf import calculate_time_lag
from ..common.dataset import Dataset
from ..common.gp import GI, GP, TGP


def inv(g_item):
    if g_item[1] == '+':
        temp = tuple([g_item[0], '-'])
    else:
        temp = tuple([g_item[0], '+'])
    return temp


def gen_apriori_candidates(R, sup, n, d_set):
    res = []
    I = []
    if len(R) < 2:
        return []
    try:
        Ck = [{x[0]} for x in R]
    except TypeError:
        Ck = [set(x[0]) for x in R]

    for i in range(len(R) - 1):
        for j in range(i + 1, len(R)):
            try:
                R_i = {R[i][0]}
                R_j = {R[j][0]}
                R_o = {R[0][0]}
            except TypeError:
                R_i = set(R[i][0])
                R_j = set(R[j][0])
                R_o = set(R[0][0])
            temp = R_i | R_j
            invtemp = {inv(x) for x in temp}
            if (len(temp) == len(R_o) + 1) and (not (I != [] and temp in I)) \
                    and (not (I != [] and invtemp in I)):
                test = 1
                for k in temp:
                    try:
                        k_set = {k}
                    except TypeError:
                        k_set = set(k)
                    temp2 = temp - k_set
                    invtemp2 = {inv(x) for x in temp2}
                    if not temp2 in Ck and not invtemp2 in Ck:
                        test = 0
                        break
                if test == 1:
                    if R[i][1] is None:
                        # read from h5 file
                        gi = GI(R[i][0][0], R[i][0][1])
                        grp = 'dataset/' + d_set.step_name + '/valid_bins/' + gi.as_string()
                        bin_data1 = d_set.read_h5_dataset(grp)
                    else:
                        bin_data1 = R[i][1]
                    if R[j][1] is None:
                        # read from h5 file
                        gi = GI(R[j][0][0], R[j][0][1])
                        grp = 'dataset/' + d_set.step_name + '/valid_bins/' + gi.as_string()
                        bin_data2 = d_set.read_h5_dataset(grp)
                    else:
                        bin_data2 = R[j][1]
                    m = bin_data1 * bin_data2
                    t = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                    if t > sup:
                        res.append([temp, m])
                I.append(temp)
                gc.collect()
    return res


def gen_valid_bins(invalid_bins, attr_cols):
    valid_gi = list()
    invalid_cols = list()
    for obj in invalid_bins:
        invalid_cols.append(obj[0])
    invalid_cols = np.array(invalid_cols, dtype=int)
    invalid_cols = np.unique(invalid_cols)
    valid_cols = np.setdiff1d(attr_cols, np.array(invalid_cols, dtype=int))
    for col in valid_cols:
        valid_gi.append([tuple([col, '+']), None])
        valid_gi.append([tuple([col, '-']), None])
    return valid_gi


def graank(f_path=None, min_sup=None, eq=False, t_diffs=None, d_set=None):
    if d_set is None:
        d_set = Dataset(f_path, min_sup, eq)
    else:
        d_set = d_set
        min_sup = d_set.thd_supp
    patterns = []
    n = d_set.attr_size
    lst_valid_gi = gen_valid_bins(d_set.invalid_bins, d_set.attr_cols)

    while len(lst_valid_gi) > 0:
        lst_valid_gi = gen_apriori_candidates(lst_valid_gi, min_sup, n, d_set)
        i = 0
        while i < len(lst_valid_gi) and lst_valid_gi != []:
            gi_tuple = lst_valid_gi[i][0]
            bin_data = lst_valid_gi[i][1]
            # grp = 'dataset/' + d_set.step_name + '/valid_bins/' + gi.as_string()
            # bin_data = d_set.read_h5_dataset(grp)
            sup = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)
            if sup < min_sup:
                del lst_valid_gi[i]
            else:
                z = 0
                while z < (len(patterns) - 1):
                    if set(patterns[z].get_pattern()).issubset(set(gi_tuple)):
                        del patterns[z]
                    else:
                        z = z + 1
                if t_diffs is not None:
                    t_lag = calculate_time_lag(bin_data, t_diffs)
                    if t_lag.valid:
                        gp = GP()
                        for obj in lst_valid_gi[i][0]:
                            gi = GI(obj[0], obj[1])
                            gp.add_gradual_item(gi)
                        gp.set_support(sup)
                        tgp = TGP(gp=gp, t_lag=t_lag)
                        patterns.append(tgp)
                else:
                    gp = GP()
                    for obj in lst_valid_gi[i][0]:
                        gi = GI(obj[0], obj[1])
                        gp.add_gradual_item(gi)
                    gp.set_support(sup)
                    patterns.append(gp)
                i += 1
    if t_diffs is None:
        return d_set, patterns
    else:
        return patterns
