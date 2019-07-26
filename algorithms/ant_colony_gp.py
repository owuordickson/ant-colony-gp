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
                temp_attr = (dataset.title[col], attribute)
                lst_attributes.append(temp_attr)
    return lst_attributes


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

# --------------------- Extract pattern combinations ------------------------------------


def validate_pattern_path(path_set, attr_x, attr_y):
    length = 0
    path_list = list(path_set)
    for n in range(len(path_list)):
        try:
            node = path_list[n]
            next_node = path_list[n + 1]
            if (attr_x[node].pheromone < attr_x[next_node].pheromone) and \
                    (attr_y[node].pheromone < attr_y[next_node].pheromone):
                # update pheromones
                if length == 0:
                    length = length + 2
                else:
                    length = length + 1
        except IndexError as e:
            break
    return length


def generate_set(t, attr_t):
    temp_t = [attr_t[t].index]
    for i in range(t, len(attr_t)):
        try:
            itm = attr_t[i]
            next_itm = attr_t[i + 1]
            if itm.pheromone < next_itm.pheromone:
                temp_t.append(next_itm.index)
        except IndexError as e:
            break
    return set(temp_t)


def compare_attributes(attr_x, attr_y):
    path_length = 0
    for x in range(len(attr_x)):
        obj_x = attr_x[x]
        row_x = obj_x.index
        for y in range(len(attr_y)):
            obj_y = attr_y[y]
            row_y = obj_y.index
            if row_x == row_y:
                temp_x = generate_set(x, attr_x)
                temp_y = generate_set(y, attr_y)
                pattern_path = temp_x.intersection(temp_y)
                temp_l = validate_pattern_path(pattern_path, attr_x, attr_y)
                # print(temp_x)
                # print(temp_y)
                # print(temp_l)
                if temp_l > path_length:
                    path_length = temp_l
    return path_length


def extract_patterns(lst_attributes, thd_supp, t_size):
    for i in range(len(lst_attributes)):
        for j in range(len(lst_attributes)):
            if i >= j:
                # 2x2 combinations of same or repeated attributes
                continue
            else:
                attr = lst_attributes[i]
                next_attr = lst_attributes[j]
                count = compare_attributes(attr[1], next_attr[1])
                supp = count/t_size
                if supp >= thd_supp:
                    print(attr[0])
                    print(next_attr[0])
                # calculate support
    return False

# --------------------- EXECUTE Ant-Colony GP -------------------------------------------


def init_algorithm(f_path, min_supp):
    try:
        dataset = DataSet(f_path)
        if dataset.data:
            lst_attributes = init_attributes(dataset, min_supp)
            # for attr in lst_attributes:
            #    for obj in attr[1]:
                    # obj = raw_obj[1]
            #        str_obj = [obj.index, obj.value, obj.pheromone]
            #        print(str_obj)
            extract_patterns(lst_attributes, min_supp, dataset.get_size())
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
