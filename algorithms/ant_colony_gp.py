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
import networkx as nx
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from algorithms.classes.data_set import DataSet
from algorithms.classes.node import Node

# -------------- Arrange rank attributes to generate Graph attribute ------------------


def init_rank(direction, raw_attr):
    lst_tuple = []
    for i in range(len(raw_attr)):
        var_node = [i, raw_attr[i]]
        lst_tuple.append(var_node)
    if direction == '+':
        ordered_tuples = sorted(lst_tuple, key=lambda x: x[1])
    elif direction == '-':
        ordered_tuples = sorted(lst_tuple, key=lambda x: x[1], reverse=True)
    # print(ordered_tuples)
    G = nx.DiGraph()
    for i in range(len(ordered_tuples)):
        # generate Graph
        try:
            node = Node(ordered_tuples[i][0], ordered_tuples[i][1])
            nxt_node = Node(ordered_tuples[i+1][0], ordered_tuples[i+1][1])
            while node.value == nxt_node.value:
                i += 1
                nxt_node = Node(ordered_tuples[i+1][0], ordered_tuples[i+1][1])
            # str_print = [node.index, nxt_node.index]
            # print(str_print)
            G.add_edge(node.index, nxt_node.index)
        except IndexError as e:
            break
    support = len(nx.dag_longest_path(G)) / len(raw_attr)
    return support, G


def init_attributes(dataset, thd_supp):
    temp = dataset.data
    cols = dataset.get_attribute_no()
    time_cols = dataset.get_time_cols()
    lst_attributes = []
    for col in range(cols):
        if time_cols and (col in time_cols):
            # exclude date-time column
            continue
        else:
            # get all tuples of an attribute/column
            raw_tuples = []
            for row in range(len(temp)):
                raw_tuples.append(float(temp[row][col]))
            # rank in ascending order and assign pheromones
            for d in {'+', '-'}:
                supp, graph_attr = init_rank(d, raw_tuples)
                if supp >= thd_supp:
                    temp_attr = [dataset.title[col][0], d, graph_attr]
                    lst_attributes.append(temp_attr)
    return lst_attributes

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

# ---------------------- Generate random patterns and pheromone matrix ---------------


def run_ant_colony(steps, max_n, attrs, thd_supp):
    descr = ['+', '-', 'x']
    p_matrix = np.ones((len(attrs), len(descr)), dtype=float)
    all_sols = []
    sols_win = []
    for t in range(steps):
        for n in range(max_n):
            sol_n = []
            for i in range(len(attrs)):
                x = (rand.randint(1, max_n)/max_n)
                pos = p_matrix[i][0]/(p_matrix[i][0]+p_matrix[i][1]+p_matrix[i][2])
                neg = (p_matrix[i][0]+p_matrix[i][1])/(p_matrix[i][0] +
                                                       p_matrix[i][1]+p_matrix[i][2])
                if x < pos:
                    temp_n = [attrs[i], '+']
                elif (x >= pos) and x < neg:
                    temp_n = [attrs[i], '-']
                else:
                    # temp_n = 'x'
                    continue
                if temp_n not in sol_n:
                    sol_n.append(temp_n)
            # if sol_n not in all_sols:
            #    all_sols.append(sol_n)
            print(sol_n)
            # test_pattern
            # update p_matrix
    return sols_win, p_matrix

# --------------------- EXECUTE Ant-Colony GP ----------------------------------------


def plot_pheromone_matrix(p_matrix, y_attrs):
    X = np.array(p_matrix)
    x_ticks = ['+', '-', 'x']
    x = [0.5, 1.5, 2.5]
    y_ticks = []
    y = []
    for i in range(len(y_attrs)):
        y.append(i+0.5)
        y_ticks.append(y_attrs[i][1])
    # Figure size (width, height) in inches
    # plt.figure(figsize=(5, 5))
    plt.title("Attribute Gray Plot")
    plt.xlabel("+ => increasing; - => decreasing; x => irrelevant")
    plt.ylabel('Attribute')
    plt.xlim(0, 3)
    plt.ylim(0, len(p_matrix))
    plt.xticks(x, x_ticks)
    plt.yticks(y, y_ticks)
    plt.pcolor(X)
    plt.gray()
    plt.show()


def init_algorithm(f_path, min_supp):
    try:
        dataset = DataSet(f_path)
        steps = 5
        max_combs = 5
        if dataset.data:
            lst_attributes = init_attributes(dataset, min_supp)
            gp, p = run_ant_colony(steps, max_combs, dataset.attributes, min_supp)
            plot_pheromone_matrix(p, dataset.title)
            # gp_patterns = extract_patterns(lst_attributes, min_supp,
            #                               dataset.get_size())
            # for obj in lst_attributes:
            #    print(obj[0])
            print(dataset.title)
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
