# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Joseph Orero and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "19 November 2019"

Usage:
    $python3 init_tgraank.py -f ../data/DATASET.csv -c 0 -s 0.5 -r 0.5

Description:
    f -> file path (CSV)
    c -> reference column
    s -> minimum support
    r -> representativity

"""

import sys
from optparse import OptionParser
from src import HandleData, TgradACO


def init_algorithm(f_path, refItem, minSup, minRep, eq=False):
    try:
        d_set = HandleData(f_path)
        if d_set.data:
            titles = d_set.title
            d_set.init_attributes(eq)
            tgp = TgradACO(d_set, refItem, minSup, minRep)
            data, time_diffs = tgp.transform_data(1)

            for txt in titles:
                col = (int(txt[0]) - 1)
                if col == refItem:
                    print(str(txt[0]) + '. ' + txt[1] + '**')
                else:
                    print(str(txt[0]) + '. ' + txt[1])
            print("\nFile: " + f_path)
            print(titles)
            print(d_set.data)
            print(d_set.attr_data)
            print(data)
            # print("Next\n")
            # print(tgp.multi_data)
        #    ac = GradACO(d_set)
        #    list_gp = ac.run_ant_colony(min_supp)
        #    print("\nPattern : Support")
        #    for gp in list_gp:
        #        print(str(gp[1])+' : '+str(gp[0]))
        #    print("\nPheromone Matrix")
        #    print(ac.p_matrix)
    except Exception as error:
        print(error)


if __name__ == "__main__":
    if not sys.argv:
        # pType = sys.argv[1]
        file_path = sys.argv[1]
        ref_col = sys.argv[2]
        min_sup = sys.argv[3]
        min_rep = sys.argv[4]
    else:
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             # default=None,
                             default='../data/DATASET2.csv',
                             type='string')
        optparser.add_option('-c', '--refColumn',
                             dest='refCol',
                             help='reference column',
                             default=2,
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
            print("Usage: $python3 init_tgraank.py -f filename.csv -c refColumn -s minSup  -r minRep")
            sys.exit('System will exit')
        else:
            inFile = options.file
        file_path = inFile
        ref_col = options.refCol
        min_sup = options.minSup
        min_rep = options.minRep

    init_algorithm(file_path, ref_col, min_sup, min_rep)
