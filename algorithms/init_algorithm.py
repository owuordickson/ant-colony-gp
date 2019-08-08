# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

Usage:
    $python t_graank.py -t 1 -f DATASET.csv -s 0.5
    $python t_graank.py -t 2 -f DATASET.csv -c 0 -s 0.5 -r 0.5

Description:
    t -> pattern type: GP, temporalGP, emergingGP
    f -> file path (CSV)
    c -> reference column
    s -> minimum support
    r -> minimum representativity

"""

import sys
from optparse import OptionParser
import numpy as np
from algorithms.classes.init_data import InitData
from algorithms.classes.gradual_ant_colony import GradualAntColony

# -------------- Extract patterns from attribute (Graph) combinations ----------------


def optimize_combs(n):
    combs = []
    for i in range(n):
        even = True
        for j in range(i + 1, n):
            if (i % 2 == 0) and even:
                even = False
                continue
            temp = [i, j]
            combs.append(temp)
    return combs


def extract_patterns(lst_graphs, thd_supp, t_size):
    gp = []
    n = len(lst_graphs)
    all_combs = np.zeros((n, n), dtype=float)
    combs_index = optimize_combs(n)
    # print(combs_index)
    for obj in lst_graphs:
        attr = str(obj[0])
        pattern = (attr + str(obj[1]))
        G = obj[2]
        print(pattern)
        print(G.edges)
    # G = lst_graphs[1][2]  # Age+
    # H = lst_graphs[3][2]  # Salary+
    # I = nx.intersection(H, G)
    # print(I.nodes)
    return gp

# --------------------- EXECUTE Ant-Colony GP ----------------------------------------


def init_algorithm(f_path, min_supp):
    try:
        d_set = InitData(f_path)
        steps = 5
        max_combs = 5
        if d_set.data:
            d_set.init_attributes(min_supp)
            ac = GradualAntColony(steps, max_combs, d_set, min_supp)
            list_gp = ac.run_ant_colony()
            ac.plot_pheromone_matrix()
            print(list_gp)
            print(d_set.title)
    except Exception as error:
        print(error)


# ------------------------- main method ---------------------------------------------


if __name__ == "__main__":
    if not sys.argv:
        pType = sys.argv[1]
        filePath = sys.argv[2]
        refCol = sys.argv[3]
        minSup = sys.argv[4]
        minRep = sys.argv[5]
    else:
        optparser = OptionParser()
        optparser.add_option('-t', '--patternType',
                             dest='pType',
                             help='patterns: FtGP, FtGEP',
                             default=1,
                             type='int')
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
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

        if options.file is None:
            filePath = '../data/DATASET.csv'
            #filePath = '../data/FluTopicData-testsansdate-blank.csv'
            #print("Usage: $python t_graank.py -f filename.csv -c refColumn -s minSup
            # -r minRep")
            #sys.exit('System will exit')
        else:
            filePath = options.file
        pType = options.pType
        refCol = options.refCol
        minSup = options.minSup
        minRep = options.minRep
    #import timeit
    if pType == 1:
        init_algorithm(filePath, minSup)
