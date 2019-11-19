# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:31:16 2015

@author: Olivier + modif MJL+MR 140316

Modified on Tue Oct 23 2018 by Dickson Owuor



Description     : This code (originally) correlates gradual patterns using concordant pairs
                    We added a functionality that allows for retrieval of time lags from concordant
                    indices, approximation using a fuzzy membership function
                    T-GRAANK - denotes Temporal GRAANK (GRAdual rANKing) and User interface
Usage:
    $python t_graank.py -f DATASET.csv -c refColumn -s minSupport  -r minRepresentativity
    $python t_graank.py -f test.csv -c 0 -s 0.5 -r 0.5

"""

import sys
import numpy as np
import gc
from optparse import OptionParser
from fuzzy_temporal import init_fuzzy_support
from data_transform import DataTransform


def Trad(dataset):

    temp = dataset
    if temp[0][0].replace('.', '', 1).isdigit() or temp[0][0].isdigit():
        return [[float(temp[j][i]) for j in range(len(temp))] for i in range(len(temp[0]))]
    else:
        if temp[0][1].replace('.', '', 1).isdigit() or temp[0][1].isdigit():
            return [[float(temp[j][i]) for j in range(len(temp))] for i in range(1, len(temp[0]))]
        else:
            title = []
            for i in range(len(temp[0])):
                #print(str(i+1) + ' : ' + temp[0][i])
                sub = (str(i+1) + ' : ' + temp[0][i])
                title.append(sub)
            return title, [[float(temp[j][i]) for j in range(1, len(temp))] for i in range(len(temp[0]))]


def GraankInit(T, eq=False):
    res = []
    n = len(T[0])
    #print(T)
    for i in range(len(T)):
        npl = str(i + 1) + '+'
        nm = str(i + 1) + '-'
        tempp = np.zeros((n, n), dtype='bool')
        tempm = np.zeros((n, n), dtype='bool')
        #print(i)
        for j in range(n):
            for k in range(j + 1, n):
                if T[i][j] > T[i][k]:
                    tempp[j][k] = 1
                    tempm[k][j] = 1
                else:
                    if T[i][j] < T[i][k]:
                        # print (j,k)
                        tempm[j][k] = 1
                        tempp[k][j] = 1
                    else:
                        if eq:
                            tempm[j][k] = 1
                            tempp[k][j] = 1
                            tempp[j][k] = 1
                            tempm[k][j] = 1
        res.append((set([npl]), tempp))
        res.append((set([nm]), tempm))
    #print(npl)
    #print(res)
    #print("a")
    return res


def SetMax(R):
    i = 0
    k = 0
    test = 0
    Cb = R
    while (i < len(Cb) - 1):
        test = 0
        k = i + 1
        while (k < len(Cb)):
            if (Cb[i].issuperset(Cb[k]) or Cb[i] == Cb[k]):
                del Cb[k]
            else:
                if Cb[i].issubset(Cb[k]):
                    del Cb[i]
                    test = 1
                    break
            k += 1
        if test == 1:
            continue
        i += 1
    return Cb


def inv(s):
    i = len(s) - 1
    if s[i] == '+':
        return s[0:i] + '-'
    else:
        return s[0:i] + '+'


def APRIORIgen(R, a, n):
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


def Graank(D_in, a, t_diffs, eq=False):
    title = D_in[0]
    T = D_in[1]
    res = []
    res2 = []
    res3 = []
    temp = 0
    n = len(T[0])
    G = GraankInit(T, eq)
    # print G
    for i in G:
        #print(i[0])
        temp = float(np.sum(i[1])) / float(n * (n - 1.0) / 2.0)
        #print(temp)
        if temp < a:
            G.remove(i)
    #        else:
    #            res.append(i[0])
    while G != []:
        G = APRIORIgen(G, a, n)
        #print(G)
        i = 0
        while i < len(G) and G != []:
            temp = float(np.sum(G[i][1])) / float(n * (n - 1.0) / 2.0)
            # print temp
            if temp < a:
                del G[i]
            else:
                # print i
                z = 0
                while z < (len(res) - 1):
                    if res[z].issubset(G[i][0]):
                        del res[z]
                        del res2[z]
                    else:
                        z = z + 1

                #return fetch indices (array) of G[1] where True
                t_lag = calculateTimeLag(getPattenIndices(G[i][1]), t_diffs, a)
                if t_lag != False:
                    res.append(G[i][0])
                    res2.append(temp)
                    res3.append(t_lag)
                i += 1
                #print(res3)
                #print("---Space---")
                #print(res)
                #print("---Space---")
                #print(res2)
                #print("---Space---")
                #print(indices)
                #print("------")
                # print G
                # res=SetMax(res)
    #                j=0
    #                k=0
    #                test=0
    #        while(j<len(res)-1):
    #                test=0
    #                k=j+1
    #                while(k<len(res)):
    #                    if(res[j].issuperset(res[k]) or res[j]==res[k]):
    #                        del res[k]
    #                        del res2[k]
    #                    else:
    #                        if res[j].issubset(res[k]):
    #                            del res[j]
    #                            del res2[j]
    #                            test=1
    #                            break
    #                    k+=1
    #                if test==1:
    #                    continue
    #                j+=1
    # print res
    return title, res, res2, res3


def fuse(L):
    Res = L[0][:][:4000]
    for j in range(len(L[0])):
        for i in range(1, len(L)):
            Res[j] = Res[j] + L[i][j][:4000]
    return Res


def fuseTrad(L):
    temp = []
    for i in L:
        temp.append(Trad(i))
    return fuse(temp)


def allComb(L, supmin, eq=False):
    ll = L
    lT = []
    for i in ll:
        lT.append(Trad(i))
    lT.append(fuseTrad(ll))
    ll.append('all')
    ld = []
    for i in range(len(ll)):
        D1, S1 = Graank(lT[i], supmin, eq)
        print(ll[i])
        for i in range(len(D1)):
            print(str(D1[i]) + ' : ' + str(S1[i]))
        ld.append(D1)
    for i in range(len(ll) - 1):
        for j in range(i + 1, len(ll)):
            Res = MBDLL(ld[i], ld[j])
            print(str(ll[j]) + ' vs ' + str(ll[i]))
            for a in range(len(Res)):
                print("Bordure " + str(a) + " : L=" + str(Res[a][0]) + ", R=" + str(Res[a][1]))
            Res = MBDLL(ld[j], ld[i])
            print(str(ll[i]) + ' vs ' + str(ll[j]))
            for a in range(len(Res)):
                print("Bordure " + str(a) + " : L=" + str(Res[a][0]) + ", R=" + str(Res[a][1]))


def getSupp(T, s, eq=False):
    n = len(T[0])
    res = 0
    for i in range(len(T[0])):
        for j in range(i + 1, len(T[0])):
            temp = 1
            tempinv = 1
            for k in s:
                x = int(k[0:(len(s) - 1)]) - 1
                if (k[len(s) - 1] == '+'):
                    if (T[x][i] > T[x][j]):
                        tempinv = 0
                    else:
                        if (T[x][i] < T[x][j]):
                            temp = 0
                else:
                    if (T[x][i] < T[x][j]):
                        tempinv = 0
                    else:
                        if (T[x][i] > T[x][j]):
                            temp = 0
                if (T[x][i] == T[x][j] and not eq):
                    temp = 0
                    tempinv = 0
            res = res + temp + tempinv
    return float(res) / float(n * (n - 1.0) / 2.0)

#-------------- NEW CODE -----------------------------


def calculateTimeLag(indices, time_diffs, minsup):
    time_lags = getTimeLags(indices,time_diffs)
    time_lag, sup = init_fuzzy_support(time_lags, time_diffs, minsup)
    if sup >= minsup:
        msg = ("~ " + time_lag[0] + str(time_lag[1]) + " " + str(time_lag[2]) + " : " + str(sup))
        return msg
    else:
        #msg = ("~ " + time_lag[0] + str(time_lag[1]) + " " + str(time_lag[2]) + " : < " + str(minsup))
        return False


def getPattenIndices(D):
    indices = []
    t_rows = len(D)
    t_columns = len(D[0])
    for r in range(t_rows):
        for c in range(t_columns):
            if D[c][r] == 1:
                index = [r,c]
                indices.append(index)

    return indices


def getTimeLags(indices,time_diffs):
    if len(indices)>0:
        indxs = np.unique(indices[0])
        time_lags = []
        for i in indxs:
            if (i >= 0) and (i < len(time_diffs)):
                time_lags.append(time_diffs[i])

        return time_lags
    else:
        raise Exception("Error: No pattern found for fetching time-lags")

# --------------------- USER INTERFACE ----------------------------------------------


def algorithm_init(filename, ref_item, minsup, minrep):
    try:
        # 1. Load dataset into program
        dataset = DataTransform(filename, ref_item, minrep)

        # 2. TRANSFORM DATA (for each step)
        patterns = 0
        for s in range(dataset.max_step):
            step = s+1  # because for-loop is not inclusive from range: 0 - max_step
            # 3. Calculate representativity
            chk_rep, rep_info = dataset.get_representativity(step)
            #print(rep_info)

            if chk_rep:
                # 4. Transform data
                data, time_diffs = dataset.transform_data(step)
                #print(data)

                # 5. Execute GRAANK for each transformation
                title, D1, S1, T1 = Graank(Trad(list(data)), minsup, time_diffs, eq=False)

                pattern_found = check_for_pattern(ref_item, D1)
                if pattern_found:
                    print(rep_info)
                    for line in title:
                        print(line)
                    print('Pattern : Support')
                    for i in range(len(D1)):
                        # D is the Gradual Patterns, S is the support for D and T is time lag
                        if (str(ref_item+1)+'+' in D1[i]) or (str(ref_item+1)+'-' in D1[i]):
                            # select only relevant patterns w.r.t *reference item
                            print(str(tuple(D1[i])) + ' : ' + str(S1[i]) + ' | ' + str(T1[i]))
                            patterns = patterns + 1
                    print("---------------------------------------------------------")

        if patterns == 0:
            print("Oops! no relevant pattern was found")
            print("---------------------------------------------------------")

    except Exception as error:
        print(error)


def check_for_pattern(ref_item, R):
    pr = 0
    for i in range(len(R)):
        # D is the Gradual Patterns, S is the support for D and T is time lag
        if (str(ref_item + 1) + '+' in R[i]) or (str(ref_item + 1) + '-' in R[i]):
            # select only relevant patterns w.r.t *reference item
            pr = pr + 1
    if pr > 0:
        return True
    else:
        return False


if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='file',
                         help='path to file containing csv',
                         #default='../misc/data/ndvi_towns.csv',
                         #default='../misc/data/DATASET.csv',
                         default=None,
                         type='string')
    optparser.add_option('-c', '--refColumn',
                         dest='refCol',
                         help='reference column',
                         default=0,
                         type='int')
    optparser.add_option('-s', '--minSupport',
                         dest='minSup',
                         help='minimum support value',
                         default=0.5,
                         type='float')
    optparser.add_option('-r', '--minRepresentativity',
                         dest='minRep',
                         help='minimum representativity',
                         default=0.5,
                         type='float')

    (options, args) = optparser.parse_args()

    inFile = None
    if options.file is None:
        print('No data-set filename specified, system with exit')
        print("Usage: $python t_graank.py -f filename.csv -c refColumn -s minSup  -r minRep")
        sys.exit('System will exit')
    else:
        inFile = options.file

    file_name = inFile
    ref_col = options.refCol
    min_sup = options.minSup
    min_rep = options.minRep

    algorithm_init(file_name, ref_col, min_sup, min_rep)
