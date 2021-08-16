# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "26 July 2021"

Breath-First Search for gradual patterns using Pure Local Search (PLS-GRAD).
PLS is used to learn gradual pattern candidates.

Adopted from: https://machinelearningmastery.com/iterated-local-search-from-scratch-in-python/

CHANGES:
1.

"""

import random
import numpy as np
from ypstruct import structure

from .shared.gp import GI, GP
from .shared.dataset_bfs import Dataset
from .shared.profile import Profile


# hill climbing local search algorithm
def run_hill_climbing(f_path, min_supp, max_iteration, max_evaluations, step_size, nvar):
    # Prepare data set
    d_set = Dataset(f_path, min_supp)
    d_set.init_gp_attributes()
    attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]
    attr_keys_spl = [attr_keys[x:x + 2] for x in range(0, len(attr_keys), 2)]

    if d_set.no_bins:
        return []

    # Parameters
    it_count = 0
    eval_count = 0

    # Empty Individual Template
    best_sol = structure()
    candidate = structure()
    # best_sol.gene = None
    # best_sol.cost = float('inf')

    # INITIALIZE
    # best_sol.gene = np.random.uniform(var_min, var_max, nvar)

    # Best Cost of Iteration
    best_costs = np.empty(max_iteration)
    best_patterns = []
    str_iter = ''
    str_eval = ''
    repeated = 0

    # generate an initial point
    best_sol.position = None
    best_sol.position = build_gp_gene(attr_keys_spl)
    best_sol.cost = cost_func(best_sol.position, attr_keys_spl, d_set)

    # run the hill climb
    while it_count < max_iteration:
        # while eval_count < max_evaluations:
        # take a step
        candidate.position = None
        if candidate.position is None:
            alpha = np.random.uniform(-step_size, 1 + step_size, best_sol.position.shape[1])
            candidate.position = alpha*best_sol.position
        candidate.cost = cost_func(candidate.position, attr_keys_spl, d_set)
        eval_count += 1

        if candidate.cost < best_sol.cost:
            best_sol = candidate.deepcopy()
        str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

        best_gp = validate_gp(d_set, decode_gp(attr_keys_spl, best_sol.position))
        is_present = is_duplicate(best_gp, best_patterns)
        is_sub = check_anti_monotony(best_patterns, best_gp, subset=True)
        if is_present or is_sub:
            repeated += 1
        else:
            if best_gp.support >= min_supp:
                best_patterns.append(best_gp)

        try:
            # Show Iteration Information
            # Store Best Cost
            best_costs[it_count] = best_sol.cost
            str_iter += "{}: {} \n".format(it_count, best_sol.cost)
        except IndexError:
            pass
        it_count += 1

        # Output
    out = structure()
    out.best_sol = best_sol
    out.best_costs = best_costs
    out.best_patterns = best_patterns
    out.str_iterations = str_iter
    out.iteration_count = it_count
    out.max_iteration = max_iteration
    out.str_evaluations = str_eval
    out.cost_evaluations = eval_count
    out.step_size = step_size
    out.titles = d_set.titles
    out.col_count = d_set.col_count
    out.row_count = d_set.row_count

    return out


def cost_func(position, attr_keys, d_set):
    pattern = decode_gp(attr_keys, position)
    temp_bin = np.array([])
    for gi in pattern.gradual_items:
        arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.gradual_item))
        if len(arg) > 0:
            i = arg[0][0]
            valid_bin = d_set.valid_bins[i]
            if temp_bin.size <= 0:
                temp_bin = valid_bin[1].copy()
            else:
                temp_bin = np.multiply(temp_bin, valid_bin[1])
    bin_sum = np.sum(temp_bin)
    if bin_sum > 0:
        cost = (1 / bin_sum)
    else:
        cost = 1
    return cost


def build_gp_gene(attr_keys):
    a = attr_keys
    temp_gene = np.random.choice(a=[0, 1], size=(len(a), 2))
    return temp_gene


def decode_gp(attr_keys, gene):
    temp_gp = GP()
    if gene is None:
        return temp_gp
    for a in range(gene.shape[0]):
        gi = None
        if gene[a][0] > gene[a][1]:
            gi = GI.parse_gi(attr_keys[a][0])
        elif gene[a][1] > gene[a][0]:
            gi = GI.parse_gi(attr_keys[a][1])
        if not(gi is None) and (not temp_gp.contains_attr(gi)):
            temp_gp.add_gradual_item(gi)
    return temp_gp


def validate_gp(d_set, pattern):

    # pattern = [('2', '+'), ('4', '+')]
    min_supp = d_set.thd_supp
    n = d_set.attr_size
    gen_pattern = GP()
    bin_arr = np.array([])

    for gi in pattern.gradual_items:
        arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.gradual_item))
        if len(arg) > 0:
            i = arg[0][0]
            valid_bin = d_set.valid_bins[i]
            if bin_arr.size <= 0:
                bin_arr = np.array([valid_bin[1], valid_bin[1]])
                gen_pattern.add_gradual_item(gi)
            else:
                bin_arr[1] = valid_bin[1].copy()
                temp_bin = np.multiply(bin_arr[0], bin_arr[1])
                supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
                if supp >= min_supp:
                    bin_arr[0] = temp_bin.copy()
                    gen_pattern.add_gradual_item(gi)
                    gen_pattern.set_support(supp)
    if len(gen_pattern.gradual_items) <= 1:
        return pattern
    else:
        return gen_pattern


def check_anti_monotony(lst_p, pattern, subset=True):
    result = False
    if subset:
        for pat in lst_p:
            result1 = set(pattern.get_pattern()).issubset(set(pat.get_pattern()))
            result2 = set(pattern.inv_pattern()).issubset(set(pat.get_pattern()))
            if result1 or result2:
                result = True
                break
    else:
        for pat in lst_p:
            result1 = set(pattern.get_pattern()).issuperset(set(pat.get_pattern()))
            result2 = set(pattern.inv_pattern()).issuperset(set(pat.get_pattern()))
            if result1 or result2:
                result = True
                break
    return result


def is_duplicate(pattern, lst_winners):
    for pat in lst_winners:
        if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                set(pattern.inv_pattern()) == set(pat.get_pattern()):
            return True
    return False


def execute(f_path, min_supp, cores, max_iteration, max_evaluations, step_size, nvar):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        out = run_hill_climbing(f_path, min_supp, max_iteration, max_evaluations, step_size, nvar)
        list_gp = out.best_patterns

        # Results
        Profile.plot_curve(out, 'Pure Local Search Algorithm (PLS)')

        wr_line = "Algorithm: PLS-GRAANK (v1.0)\n"
        wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
        wr_line += "Step size: " + str(out.step_size) + '\n'

        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(list_gp)) + '\n'
        wr_line += "Number of iterations: " + str(out.iteration_count) + '\n'
        wr_line += "Number of cost evaluations: " + str(out.cost_evaluations) + '\n\n'

        for txt in out.titles:
            try:
                wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in list_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(round(gp.support, 3)) + '\n')

        wr_line += '\n\n' + "Iteration: Cost" + '\n'
        wr_line += out.str_iterations
        # wr_line += '\n\n' + "Evaluation: Cost" + '\n'
        # wr_line += out.str_evaluations
        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line
