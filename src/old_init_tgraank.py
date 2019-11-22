# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Joseph Orero and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "19 November 2019"

Usage:
    $python init_acograd.py -f ../data/DATASET.csv -c 0 -s 0.5 -r 0.5

Description:
    f -> file path (CSV)
    c -> reference column
    s -> minimum support
    r -> representativity

"""

import sys
from optparse import OptionParser
from src import DataTransform, Graank, Trad


def algorithm_init(filename, ref_item, minsup, minrep):
    try:
        # 1. Load dataset into program
        dataset = DataTransform(filename, ref_item, minrep)

        # 2. TRANSFORM DATA (for each step)
        patterns = 0
        show_title = False
        for s in range(dataset.max_step):
            step = s+1  # because for-loop is not inclusive from range: 0 - max_step
            # 3. Calculate representativity
            chk_rep, rep_info = dataset.get_representativity(step)
            # print(rep_info)

            if chk_rep:
                # 4. Transform data
                data, time_diffs = dataset.transform_data(step)
                # print(data)

                # 5. Execute GRAANK for each transformation
                title, D1, S1, T1 = Graank(Trad(list(data)), minsup, time_diffs, eq=False)

                pattern_found = check_for_pattern(ref_item, D1)
                if pattern_found:
                    # print(rep_info)
                    if not show_title:
                        show_title = True
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
                         # default=None,
                         default='../data/DATASET2.csv',
                         # default='../data/x_data.csv',
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

    import time
    start = time.time()
    algorithm_init(file_name, ref_col, min_sup, min_rep)
    end = time.time()
    print("\n" + str(end - start) + " seconds")