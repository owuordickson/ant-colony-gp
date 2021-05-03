# -*- coding: utf-8 -*-
"""
@author: Olivier + modif MJL+MR 140316
@created on Fri Jun 12 14:31:16 2015

@modified by D. Owuor 04 June 2020

"""

import numpy as np
import gc

from .shared.fuzzy_mf import calculate_time_lag
from .shared.dataset_bfs import Dataset
from .shared.gp import GI, GP, TGP


def inv(g_item):
    if g_item[1] == '+':
        temp = tuple([g_item[0], '-'])
    else:
        temp = tuple([g_item[0], '+'])
    return temp


def gen_apriori_candidates(R, sup, n):
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
                    m = R[i][1] * R[j][1]
                    t = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                    if t > sup:
                        res.append([temp, m])
                I.append(temp)
                gc.collect()
    return res


def graank(f_path=None, min_sup=None, eq=False, t_diffs=None, d_set=None):
    if d_set is None:
        d_set = Dataset(f_path, min_sup, eq)
        d_set.init_gp_attributes()
    else:
        d_set = d_set
        min_sup = d_set.thd_supp
    patterns = []
    n = d_set.attr_size
    # lst_valid_gi = gen_valid_bins(d_set.invalid_bins, d_set.attr_cols)
    valid_bins = d_set.valid_bins

    while len(valid_bins) > 0:
        valid_bins = gen_apriori_candidates(valid_bins, min_sup, n)
        i = 0
        while i < len(valid_bins) and valid_bins != []:
            gi_tuple = valid_bins[i][0]
            bin_data = valid_bins[i][1]
            # grp = 'dataset/' + d_set.step_name + '/valid_bins/' + gi.as_string()
            # bin_data = d_set.read_h5_dataset(grp)
            sup = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)
            if sup < min_sup:
                del valid_bins[i]
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
                        for obj in valid_bins[i][0]:
                            gi = GI(obj[0], obj[1].decode())
                            gp.add_gradual_item(gi)
                        gp.set_support(sup)
                        tgp = TGP(gp=gp, t_lag=t_lag)
                        patterns.append(tgp)
                else:
                    gp = GP()
                    for obj in valid_bins[i][0]:
                        gi = GI(obj[0], obj[1].decode())
                        gp.add_gradual_item(gi)
                    gp.set_support(sup)
                    patterns.append(gp)
                i += 1
    if t_diffs is None:
        return d_set, patterns
    else:
        return patterns


def init(f_path, min_supp, cores, eq=False):
    try:
        d_set, list_gp = graank(f_path, min_supp, eq)

        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        wr_line = "Algorithm: GRAANK \n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.row_count) + '\n'
        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(list_gp)) + '\n\n'

        for txt in d_set.titles:
            wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in list_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')

        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line
