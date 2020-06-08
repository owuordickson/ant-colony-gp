# -*- coding: utf-8 -*-
"""
@author: Olivier + modif MJL+MR 140316
@created on Fri Jun 12 14:31:16 2015

@modified by D. Owuor 04 June 2020

"""

import numpy as np
import gc
import json
from src.algorithms.common.fuzzy_mf import calculate_time_lag
from src.algorithms.common.dataset import Dataset
from src.algorithms.common.gp import GI, GP, TGP


def inv(g_item):
    if g_item[1] == '+':
        temp = tuple([g_item[0], '-'])
    else:
        temp = tuple([g_item[0], '+'])
    return temp


def gen_apriori_candidates(R, sup, n, step):
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
                    bin_obj1 = Dataset.read_json(R[i][1])
                    bin_obj2 = Dataset.read_json(R[j][1])
                    bin_data1 = np.array(bin_obj1['bin'])
                    bin_data2 = np.array(bin_obj2['bin'])
                    m = bin_data1 * bin_data2
                    t = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                    if t > sup:
                        path = store_gp(temp, m, t, step)
                        res.append(path)
                I.append(temp)
                gc.collect()
    return res


def store_gp(gi, bin_data, supp, step):
    gi_data = []
    gi_tuple = []
    gi_str = ""
    for obj in gi:
        gi_data.append([int(obj[0]), str(obj[1])])
        gi_tuple.append(tuple([obj[0], obj[1]]))
        gi_str += str(obj[0]) + str(obj[1])
    path = 'gi_' + gi_str + str(step) + '.json'
    content = {"gi": gi_data, "bin": bin_data.tolist(), "support": supp}
    Dataset.write_file(json.dumps(content), path)
    return [gi_tuple, path]


def graank(f_path=None, min_sup=None, eq=False, t_diffs=None, d_set=None, step=0):
    if d_set is None:
        d_set = Dataset(f_path, min_sup, eq)
    else:
        d_set = d_set
    patterns = []
    n = d_set.attr_size
    bin_paths = list(d_set.valid_gi_paths)

    while len(bin_paths) > 0:
        bin_paths = gen_apriori_candidates(bin_paths, min_sup, n, step)
        i = 0
        while i < len(bin_paths) and bin_paths != []:
            # temp = float(np.sum(G[i][1])) / float(n * (n - 1.0) / 2.0)
            d_set.gen_paths.append(bin_paths[i][1])
            bin_obj = Dataset.read_json(bin_paths[i][1])
            bin_data = np.array(bin_obj['bin'])
            sup = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)
            if sup < min_sup:
                del bin_paths[i]
            else:
                z = 0
                while z < (len(patterns) - 1):
                    if set(patterns[z].get_pattern()).issubset(set(bin_paths[i][0])):
                        del patterns[z]
                    else:
                        z = z + 1
                # return fetch indices (array) of G[1] where True
                if t_diffs is not None:
                    # t_lag = FuzzyMF.calculate_time_lag(FuzzyMF.get_patten_indices(bin_data), t_diffs, min_sup)
                    t_lag = calculate_time_lag(bin_data, t_diffs)
                    if t_lag:
                        gp = GP()
                        for obj in bin_paths[i][0]:
                            gi = GI(obj[0], obj[1])
                            gp.add_gradual_item(gi)
                        gp.set_support(sup)
                        tgp = TGP(gp=gp, t_lag=t_lag)
                        patterns.append(tgp)
                else:
                    gp = GP()
                    for obj in bin_paths[i][0]:
                        gi = GI(obj[0], obj[1])
                        gp.add_gradual_item(gi)
                    gp.set_support(sup)
                    patterns.append(gp)
                i += 1
    if t_diffs is None:
        return d_set, patterns
    else:
        return patterns
