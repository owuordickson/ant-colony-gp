# -*- coding: utf-8 -*-
"""
@author: Olivier + modif MJL+MR 140316
@created on Fri Jun 12 14:31:16 2015

@modified by D. Owuor 06 December 2019

"""

import numpy as np
import gc
# from src import FuzzyMF
from algorithms.tgraank.fuzzy_mf import FuzzyMF


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


def inv(s):
    i = len(s) - 1
    if s[i] == '+':
        return s[0:i] + '-'
    else:
        return s[0:i] + '+'


def gen_apriori_candidates(R, a, n):
    res = []
    test = 1
    temp = set()
    temp2 = set()
    # print"a"
    I = []
    if (len(R) < 2):
        return []
    Ck = [x[0] for x in R]
    # print"b"
    for i in range(len(R) - 1):
        # print"c"
        # print len(R)
        for j in range(i + 1, len(R)):
            temp = R[i][0] | R[j][0]
            invtemp = {inv(x) for x in temp}
            # print invtemp
            # print"d"+str(j)
            if ((len(temp) == len(R[0][0]) + 1) and (not (I != [] and temp in I)) and (not (I != [] and invtemp in I))):
                test = 1
                # print "e"
                for k in temp:
                    temp2 = temp - set([k])
                    invtemp2 = {inv(x) for x in temp2}
                    if not temp2 in Ck and not invtemp2 in Ck:
                        test = 0
                        break
                if test == 1:
                    m = R[i][1] * R[j][1]
                    t = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                    if t > a:
                        res.append((temp, m))
                I.append(temp)
                gc.collect()
    # print "z"
    return res


def graank(T, a, t_diffs=None, eq=False):
    res = []
    res2 = []
    res3 = []
    n = len(T[0][1])
    G = init_graank(T, eq)
    for i in G:
        temp = float(np.sum(i[1])) / float(n * (n - 1.0) / 2.0)
        if temp < a:
            G.remove(i)
    while G != []:
        G = gen_apriori_candidates(G, a, n)
        i = 0
        while i < len(G) and G != []:
            temp = float(np.sum(G[i][1])) / float(n * (n - 1.0) / 2.0)
            if temp < a:
                del G[i]
            else:
                z = 0
                while z < (len(res) - 1):
                    if res[z].issubset(G[i][0]):
                        del res[z]
                        del res2[z]
                    else:
                        z = z + 1
                # return fetch indices (array) of G[1] where True
                if t_diffs is not None:
                    # t_lag = calculateTimeLag(getPattenIndices(G[i][1]), t_diffs, a)
                    t_lag = FuzzyMF.calculate_time_lag(FuzzyMF.get_patten_indices(G[i][1]), t_diffs, a)
                    if t_lag:
                        res.append(G[i][0])
                        res2.append(temp)
                        res3.append(t_lag)
                else:
                    res.append(G[i][0])
                    res2.append(temp)
                i += 1
    # return title, res, res2, res3
    if t_diffs is None:
        return res, res2
    else:
        return res, res2, res3
