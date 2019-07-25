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
from algorithms.classes.data_set import DataSet
from algorithms.classes.item import Item


def init_pheromones(lst_items):
    count = 0
    support = 0
    for i in range(len(lst_items)):
        try:
            item = lst_items[i]
            next_item = lst_items[i+1]
            if item.value < next_item.value:
                next_item.update_pheromone(1, item.pheromone)
                if count == 0:
                    count = count + 2
                else:
                    count = count + 1
            else:
                next_item.update_pheromone(0, item.pheromone)
        except IndexError as e:
            support = count/(len(lst_items))
            break
    return support, lst_items


def init_rank(raw_attr, thd_supp):
    lst_tuple = []
    for i in range(len(raw_attr)):
        var_tuple = [i, raw_attr[i]]
        lst_tuple.append(var_tuple)
    sorted_tuples = sorted(lst_tuple, key=lambda x: x[1])
    # print(sorted_tuples)
    temp_items = []
    for obj in sorted_tuples:
        tuple_item = Item(obj[0], obj[1])
        temp_items.append(tuple_item)
    supp, lst_items = init_pheromones(temp_items)
    if supp >= thd_supp:
        return lst_items
    else:
        return False


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
            attribute = init_rank(raw_tuples, thd_supp)
            if attribute:
                temp = (dataset.title[col], attribute)
                lst_attributes.append(temp)
    for attr in lst_attributes:
        print(attr[0])
        for obj in attr[1]:
            print(obj.index)
            print(obj.value)
            print(obj.pheromone)
    return lst_attributes

# --------------------- EXECUTE Ant-Colony GP -------------------------------------------


def init_algorithm(f_path, min_supp):
    try:
        dataset = DataSet(f_path)
        if dataset.data:
            init_attributes(dataset, min_supp)
            # print(dataset.title)
    except Exception as error:
        print(error)


# ------------------------- main method -------------------------------------------------


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
