# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "29 April 2021"
@modified: "16 August 2021"

Breath-First Search for gradual patterns (GA-GRAANK)

"""
import numpy as np
from ypstruct import structure

from .shared.gp import GI, GP
from .shared.dataset_bfs import Dataset
from .shared.profile import Profile


def run_genetic_algorithm(f_path, min_supp, max_iteration, max_evaluations, n_pop, pc, gamma, mu, sigma, nvar):
    # Prepare data set
    d_set = Dataset(f_path, min_supp)
    d_set.init_gp_attributes()
    attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]
    attr_keys_spl = [attr_keys[x:x + 2] for x in range(0, len(attr_keys), 2)]

    if d_set.no_bins:
        return []

    # Problem Information
    # cost_func = cost_func

    # Parameters
    it_count = 0
    eval_count = 0
    nc = int(np.round(pc * n_pop / 2) * 2)  # Number of children. np.round is used to get even number of children

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.gene = None
    empty_individual.cost = None

    # Initialize Population
    pop = empty_individual.repeat(n_pop)
    for i in range(n_pop):
        pop[i].gene = build_gp_gene(attr_keys_spl)
        pop[i].cost = 1

    # Best Solution Ever Found
    best_sol = empty_individual.deepcopy()
    best_sol.gene = pop[0].gene
    best_sol.cost = cost_func(best_sol.gene, attr_keys_spl, d_set)

    # Best Cost of Iteration
    best_costs = np.empty(max_iteration)
    best_genes = []
    best_patterns = []
    str_iter = ''
    str_eval = ''

    repeated = 0
    while eval_count < max_evaluations:
        # while it_count < max_iteration:
        # while repeated < 1:

        c_pop = []  # Children population
        for _ in range(nc // 2):
            # Select Parents
            q = np.random.permutation(n_pop)
            p1 = pop[q[0]]
            p2 = pop[q[1]]

            # Perform Crossover
            c1, c2 = crossover(p1, p2, gamma)

            # Perform Mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            # Apply Bound
            # apply_bound(c1, var_min, var_max)
            # apply_bound(c2, var_min, var_max)

            # Evaluate First Offspring
            c1.cost = cost_func(c1.gene, attr_keys_spl, d_set)
            eval_count += 1
            if c1.cost < best_sol.cost:
                best_sol = c1.deepcopy()
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

            # Evaluate Second Offspring
            c2.cost = cost_func(c2.gene, attr_keys_spl, d_set)
            eval_count += 1
            if c2.cost < best_sol.cost:
                best_sol = c2.deepcopy()
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

            # Add Offsprings to c_pop
            c_pop.append(c1)
            c_pop.append(c2)

        # Merge, Sort and Select
        pop += c_pop
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:n_pop]

        best_gp = validate_gp(d_set, decode_gp(attr_keys_spl, best_sol.gene))
        is_present = is_duplicate(best_gp, best_patterns)
        is_sub = check_anti_monotony(best_patterns, best_gp, subset=True)
        if is_present or is_sub:
            repeated += 1
        else:
            if best_gp.support >= min_supp:
                best_patterns.append(best_gp)
            # else:
            #    best_sol.cost = 1

        try:
            # Show Iteration Information
            # Store Best Cost
            best_costs[it_count] = best_sol.cost
            best_genes.append(best_sol.gene)
            # print("Iteration {}: Best Cost = {}".format(it_count, best_costs[it_count]))
            str_iter += "Iteration {}: Best Cost: {} \n".format(it_count, best_costs[it_count])
        except IndexError:
            pass
        it_count += 1

    # Output
    out = structure()
    out.pop = pop
    out.best_sol = best_sol
    out.best_costs = best_costs
    out.best_patterns = best_patterns
    out.str_iterations = str_iter
    out.str_evaluations = str_eval
    out.iteration_count = it_count
    out.max_iteration = max_iteration
    out.cost_evaluations = eval_count
    out.n_pop = n_pop
    out.pc = pc
    out.titles = d_set.titles
    out.col_count = d_set.col_count
    out.row_count = d_set.row_count

    return out


def cost_func(gene, attr_keys, d_set):
    pattern = decode_gp(attr_keys, gene)
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


def crossover(p1, p2, gamma=0.1):
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, c1.gene.shape[1])
    c1.gene = alpha*p1.gene + (1-alpha)*p2.gene
    c2.gene = alpha*p2.gene + (1-alpha)*p1.gene
    return c1, c2


def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.gene.shape) <= mu
    ind = flag  # np.argwhere(flag)
    y.gene += sigma*np.random.rand(*ind.shape)
    return y


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


def execute(f_path, min_supp, cores, max_iteration, max_evaluations, n_pop, pc, gamma, mu, sigma, nvar):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        out = run_genetic_algorithm(f_path, min_supp, max_iteration, max_evaluations, n_pop, pc, gamma, mu, sigma, nvar)
        list_gp = out.best_patterns

        # Results
        Profile.plot_curve(out, 'Genetic Algorithm (GA)')

        wr_line = "Algorithm: GA-GRAANK (v1.0)\n"
        wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
        wr_line += "Population size: " + str(out.n_pop) + '\n'
        wr_line += "PC: " + str(out.pc) + '\n'

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

        # wr_line += '\n\n' + "Iteration: Best Cost" + '\n'
        # wr_line += out.str_iterations
        wr_line += '\n\n' + "Evaluation: Cost" + '\n'
        wr_line += out.str_evaluations
        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line
