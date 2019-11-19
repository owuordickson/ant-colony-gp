"""
@author: "Dickson Owuor"
@credits: "Anne Laurent, Joseph Orero"
@version: "1.0"
@email: "owuordickson@gmail.com"

Modified on Mon Jan 28 2019

This algorithm implements and modifies the design of Dong and Li, 1999

"""

from src import border_diff, get_border_length, is_list_empty


def mbdll_border(dataset_1, dataset_2):
    ep_list = list()

    #if set(dataset_1).isdisjoint(set(dataset_2)):

    count_d2 = get_border_length(dataset_2)
    if count_d2 <= 1:
        temp_list = get_ep_border(dataset_2, dataset_1)
        if temp_list:
            ep_list.append(temp_list)
    else:
        for d2_item in dataset_2:  # starts at 1 - only the maximal items
            temp_list = get_ep_border(d2_item, dataset_1)
            if temp_list:
                ep_list.append(temp_list)
    return ep_list
    #else:
    #    return ep_list


def get_intersections(item_1, init_list):
    items = list()
    #print(init_list)
    #print(item_1)
    count_c = get_border_length(init_list)
    if count_c <= 1:
        C = set(init_list)
        if C.issuperset(set(item_1)):  # it means item is in both data-sets hence not emerging
            return items
        else:
            diff = C.intersection(set(item_1))
            items.append(list(diff))
    else:
        for item in init_list:
            C = set(item)
            if C.issuperset(set(item_1)):  # it means item is in both data-sets
                return items
            else:
                diff = C.intersection(set(item_1))
                if diff:
                    items.append(list(diff))
    return items


def get_ep_border(d2_item, d1_list):
    ep = list()
    r_list = get_intersections(d2_item, d1_list)
    #print(r_list)
    if not is_list_empty(r_list):
        u_list = list(d2_item)
        L, U, R = border_diff(u_list, r_list)
        ep = list(L)
    return ep
