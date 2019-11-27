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
from src import HandleData, Tgrad


def init_algorithm(f_path, refItem, minSup, minRep, allowPara, eq=False):
    try:
        wr_line = ""
        d_set = HandleData(f_path)
        if d_set.data:
            titles = d_set.title
            d_set.init_attributes(eq)
            tgp = Tgrad(d_set, refItem, minSup, minRep)
            if allowPara == 1:
                msg_para = "True"
                list_tgp = tgp.run_tgraank(parallel=True)
            else:
                msg_para = "False"
                list_tgp = tgp.run_tgraank()
            list_tgp.sort(key=lambda k: (k[0][0], k[0][1]), reverse=True)

            wr_line = "Algorithm: T-GRAANK \n"
            wr_line += "Multi-core execution: " + msg_para + '\n\n'
            for txt in titles:
                col = (int(txt[0]) - 1)
                if col == refItem:
                    wr_line += (str(txt[0]) + '. ' + txt[1] + '**' + '\n')
                else:
                    wr_line += (str(txt[0]) + '. ' + txt[1] + '\n')
                # csv_data.append(wr_line)
            wr_line += ("\nFile: " + f_path + '\n')
            wr_line += ("\nPattern : Support" + '\n')

            for obj in list_tgp:
                for i in range(len(obj[0])):
                    wr_line += (str(obj[0][i]) + ' : ' + str(obj[1][i]) + ' | ' + str(obj[2][i]) + '\n')
        return wr_line
    except Exception as error:
        print(error)


if __name__ == "__main__":
    if not sys.argv:
        # pType = sys.argv[1]
        file_path = sys.argv[1]
        ref_col = sys.argv[2]
        min_sup = sys.argv[3]
        min_rep = sys.argv[4]
        allow_p = sys.argv[5]
    else:
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             # default=None,
                             # default='../data/DATASET2.csv',
                             default='../data/Directio.csv',
                             type='string')
        optparser.add_option('-c', '--refColumn',
                             dest='refCol',
                             help='reference column',
                             default=1,
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
        optparser.add_option('-p', '--allowMultiprocessing',
                             dest='allowPara',
                             help='allow multiprocessing',
                             default=1,
                             type='int')
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
        allow_p = options.allowPara

    import time
    start = time.time()
    res_text = init_algorithm(file_path, ref_col, min_sup, min_rep, allow_p)
    end = time.time()

    wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    wr_text += res_text
    HandleData.write_file(wr_text)
