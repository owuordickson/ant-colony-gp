# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

Usage:
    $python init_grad_aco.py -f DATASET.csv -s 0.5

Description:
    f -> file path (CSV)
    s -> minimum support
"""

import sys
from optparse import OptionParser
from algorithms.classes.init_data import InitData
from algorithms.classes.gradual_aco import GradACO


def init_algorithm(f_path, min_supp, eq=False, steps=False, max_combs=False):
    try:
        d_set = InitData(f_path)
        if d_set.data:
            if not steps or not max_combs:
                a = d_set.get_attribute_no()
                steps = a  # (a * a)
                max_combs = a  # (a * a)
            print(d_set.title)
            # d_set.init_bin_attributes(min_supp, eq)
            # ac = GradACO(steps, max_combs, d_set)
            # list_gp = ac.run_ant_colony()
            d_set.init_attributes(eq)
            ac = GradACO(steps, max_combs, d_set)
            list_gp = ac.run_ant_colony(min_supp)
            print("\nPATTERNS")
            for obj in list_gp:
                print(str(obj[1])+' : '+str(obj[0]))
            print("\nPheromone Matrix")
            # print(ac.p_matrix)
            ac.plot_pheromone_matrix()
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
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             default=None,
                             type='string')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=0.5,
                             type='float')
        optparser.add_option('-e', '--allowEqual',
                             dest='allowEq',
                             help='allow equal',
                             default=None,
                             type='int')
        optparser.add_option('-t', '--maxSteps',
                             dest='maxSteps',
                             help='maximum steps',
                             default=None,
                             type='int')
        optparser.add_option('-n', '--maxPatterns',
                             dest='maxComb',
                             help='maximum pattern combinations',
                             default=None,
                             type='int')
        (options, args) = optparser.parse_args()

        if options.file is None:
            #filePath = '../data/DATASET.csv'
            #filePath = '../data/FluTopicData-testsansdate-blank.csv'
            filePath = '../data/transfusion.csv'
            #print("Usage: $python t_graank.py -f filename.csv -c refColumn -s minSup
            # -r minRep")
            #sys.exit('System will exit')
        else:
            filePath = options.file
        minSup = options.minSup
        allowEq = options.allowEq
        maxSteps = options.maxSteps
        maxComb = options.maxComb
    import time
    start = time.time()
    init_algorithm(filePath, minSup)
    end = time.time()
    print((end-start))

