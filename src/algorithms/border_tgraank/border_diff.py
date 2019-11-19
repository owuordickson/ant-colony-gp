"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@version: "1.0"
@email: "owuordickson@gmail.com"

Modified on Fri Jan 25 2019

This algorithm implements and modifies the design of Dong and Li, 1999

"""

import itertools as it
from collections import Iterable


def border_diff(u_list, r_list):

    # U is the Universe set
    # R is the Right (Maximal/left-rooted) set
    # L = U - R (difference of borders U and R)
    U = gen_set(list(u_list), select=1)
    R = gen_set(list(r_list), select=1)
    L = set()
    l_list = list()

    count_u = get_border_length(u_list)  # testing if set item is a single array item

    if count_u < 1:
        l_list.append(get_border_diff(u_list, r_list))
    else:
        [l_list.append(get_border_diff(u_item, r_list)) for u_item in u_list]

    if not is_list_empty(l_list):
        L = gen_set(l_list, u_list, select=2)
    return L, U, R


def get_border_diff(a, b_list):
    border_list = tuple()
    count_b = get_border_length(b_list)
    if count_b < 1:
        diff = set(a).difference(set(b_list))  # (for sets) same as diff = A - B
        temp_list = expand_border(border_list, list(diff))
        border_list = remove_non_minimal(temp_list)
    else:
        for b_item in b_list:
            try:
                set_B = set(b_item)
            except TypeError:
                set_B = set({b_item})
            diff = set(a).difference(set_B)
            temp_list = expand_border(border_list, list(diff))  # expands/updates every border item by adding diff
            border_list = remove_non_minimal(temp_list)  # removes non-minimal items from expanded list
    return border_list


def expand_border(init_list, item):
    temp = it.product(init_list, item)
    expanded_list = list()

    if not init_list:
        [expanded_list.append(a) for a in item]
    elif set(init_list) == set(item):
        expanded_list = init_list
    else:
        [expanded_list.append(tuple(combine_items(list(a)))) for a in temp]

    expanded_list.sort()
    return expanded_list


def remove_non_minimal(init_list):
    item_set = tuple()
    for item_i in init_list:
        for item_j in init_list:
            if isinstance(item_i, Iterable) and isinstance(item_j, Iterable) \
                    and not isinstance(item_i, str) and not isinstance(item_j, str):
                set_i = tuple(item_i)
                set_j = tuple(item_j)
            else:
                return init_list

            # removes those elements that are non-minimal
            # -------------------------------------------
            # Maximal item-sets: biggest set that has no superset
            # Minimal item-sets: smallest set than any other (may or may not be a subset of another set)
            # non-minimal is therefore neither maximal nor minimal
            if (not set(set_i).issubset(set(set_j))) and (not set(set_i).issuperset(set(set_j))) \
                    and (set(set_i) == set(set_j)):
                # the two sets are non-minimal
                continue
            elif not set(set_i).isdisjoint(set(set_j)):  # the two sets are not the same
                continue
            else:
                s = set(set_i)
                if len(tuple(s)) <= 1:
                    item_set = item_set + (tuple(s))
                else:
                    item_set = item_set + (tuple(s),)
    return tuple(set(item_set))


def gen_set(in_list, r_border=(), select=0):
    i_set = tuple()
    for i in range(len(in_list)):

        if isinstance(in_list[i], Iterable) and isinstance(in_list[i], Iterable) \
                and not isinstance(in_list[i], str) and not isinstance(in_list[i], str):
            item_i = tuple(in_list[i])
        else:
            item_i = in_list

        if i > 0 and item_i == in_list:  # takes care of single item lists(or sets)
            break

        S = set(item_i)
        if len(tuple(S)) <= 1:
            i_set = i_set + (tuple(S))
        else:
            i_set = i_set + (tuple(S),)

    if select == 1:            # left-rooted border
        border = (i_set, ())
        return set(border)
    elif select == 2:           # non-rooted border
        border = ((tuple(r_border),) + i_set)
        return set(border)
    else:                      # normal set
        return set(i_set)


def get_border_length(item_list):
    n = 0
    for item in item_list:
        if isinstance(item, Iterable) and not isinstance(item, str):
            n += 1

    return n


def is_list_empty(items):
    if isinstance(items, list):  # Is a list
        return all(map(is_list_empty, items))
    return False  # Not a list


def combine_items(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in combine_items(item):
                yield x
        else:
            yield item
