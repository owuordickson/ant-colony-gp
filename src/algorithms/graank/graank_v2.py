# -*- coding: utf-8 -*-
"""
@author: Olivier + modif MJL+MR 140316
@created on Fri Jun 12 14:31:16 2015

@modified by D. Owuor 06 December 2019

"""

import numpy as np
import gc
import json
from src.algorithms.common.fuzzy_mf_v1 import FuzzyMF
from src.algorithms.common.dataset import Dataset


def init_graank(T, eq=False):
    res = []
    n = len(T[0][1])
    for i in range(len(T)):
        # npl = str(i + 1) + '+'
        # nm = str(i + 1) + '-'
        attr = T[i][0]
        bin = T[i][1]
        npl = str(attr) + '+'
        nm = str(attr) + '-'
        tempp = np.zeros((n, n), dtype='bool')
        tempm = np.zeros((n, n), dtype='bool')
        for j in range(n):
            for k in range(j + 1, n):
                if bin[j] > bin[k]:
                    tempp[j][k] = 1
                    tempm[k][j] = 1
                else:
                    if bin[j] < bin[k]:
                        # print (j,k)
                        tempm[j][k] = 1
                        tempp[k][j] = 1
                    else:
                        if eq:
                            tempm[j][k] = 1
                            tempp[k][j] = 1
                            tempp[j][k] = 1
                            tempm[k][j] = 1
        res.append(({npl}, tempp))
        res.append(({nm}, tempm))
    return res


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
    # Ck = [set([x[0]]) for x in R]
    Ck = []
    for x in R:
        try:
            Ck.append(set([x[0]]))
        except TypeError:
            Ck.append(set(x[0]))
    for i in range(len(R) - 1):
        for j in range(i + 1, len(R)):
            try:
                R_i = set([R[i][0]])
                R_j = set([R[j][0]])
                R_o = set([R[0][0]])
            except TypeError:
                R_i = set(R[i][0])
                R_j = set(R[j][0])
                R_o = set(R[0][0])

            temp = R_i | R_j
            invtemp = {inv(x) for x in temp}
            # print([temp, invtemp])
            if (len(temp) == len(R_o) + 1) and (not (I != [] and temp in I)) and (not (I != [] and invtemp in I)):
                test = 1
                for k in temp:
                    try:
                        k_set = set([k])
                    except TypeError:
                        k_set = set(k)
                    temp2 = temp - k_set
                    invtemp2 = {inv(x) for x in temp2}
                    if not temp2 in Ck and not invtemp2 in Ck:
                        test = 0
                        break
                if test == 1:
                    # print(list(temp))
                    bin_obj1 = Dataset.read_json(R[i][1])
                    bin_obj2 = Dataset.read_json(R[j][1])
                    bin_data1 = np.array(bin_obj1['bin'])
                    bin_data2 = np.array(bin_obj2['bin'])
                    # print(bin_data1)
                    # m = R[i][1] * R[j][1]
                    m = bin_data1 * bin_data2
                    # print(m)
                    t = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                    if t > sup:
                        # res.append((temp, m))
                        path = store_gp(temp, m, t)
                        # print(path)
                        res.append(path)
                I.append(temp)
                gc.collect()
    return res


def store_gp(gi, bin_data, supp):
    gi_data = []
    gi_tuple = []
    gi_str = ""
    for obj in gi:
        gi_data.append([int(obj[0]), str(obj[1])])
        gi_tuple.append(tuple([obj[0], obj[1]]))
        gi_str += str(obj[0]) + str(obj[1])
    path = 'gi_' + gi_str + '.json'
    content = {"gi": gi_data, "bin": bin_data.tolist(), "support": supp}
    Dataset.write_file(json.dumps(content), path)
    return [gi_tuple, path]


def graank(f_path, sup, eq, t_diffs=None):
    d_set = Dataset(f_path, sup, eq)
    # T = d_set.attr_data
    res = []
    res2 = []
    res3 = []
    n = d_set.attr_size
    G = d_set.valid_gi_paths
    gen_paths = []
    # n = len(T[0][1])
    # G = init_graank(T, eq)
    # for i in G:
    #    temp = float(np.sum(i[1])) / float(n * (n - 1.0) / 2.0)
    #    if temp < a:
    #        G.remove(i)
    while G != []:
        G = gen_apriori_candidates(G, sup, n)
        i = 0
        while i < len(G) and G != []:
            # temp = float(np.sum(G[i][1])) / float(n * (n - 1.0) / 2.0)
            gen_paths.append(G[i][1])
            bin_obj = Dataset.read_json(G[i][1])
            bin_data = np.array(bin_obj['bin'])
            temp = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)
            if temp < sup:
                del G[i]
            else:
                z = 0
                while z < (len(res) - 1):
                    # print(set(res[z]))
                    # print(set(G[i][0]))
                    if set(res[z]).issubset(set(G[i][0])):
                        del res[z]
                        del res2[z]
                    else:
                        z = z + 1
                # return fetch indices (array) of G[1] where True
                if t_diffs is not None:
                    # t_lag = calculateTimeLag(getPattenIndices(G[i][1]), t_diffs, a)
                    t_lag = FuzzyMF.calculate_time_lag(FuzzyMF.get_patten_indices(G[i][1]), t_diffs, sup)
                    if t_lag:
                        res.append(G[i][0])
                        res2.append(temp)
                        res3.append(t_lag)
                else:
                    res.append(G[i][0])
                    res2.append(temp)
                i += 1
    d_set.clean_memory()
    for file in gen_paths:
        Dataset.delete_file(file)
    if t_diffs is None:
        return d_set, res, res2
    else:
        return d_set, res, res2, res3
