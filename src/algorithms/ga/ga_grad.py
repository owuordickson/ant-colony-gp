# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "29 April 2021"
@modified: "29 April 2021"

Breath-First Search for gradual patterns (GA-GRAANK)

"""
import numpy as np
from ypstruct import structure
from src.common.gp import GI, GP
from src.common.dataset_bfs import Dataset


class GradGA:

    def __init__(self, f_path, min_supp):
        self.d_set = Dataset(f_path, min_supp)
        self.d_set.init_gp_attributes()
        self.attr_index = self.d_set.attr_cols
        self.iteration_count = 0
        self.max_it = 100
        self.npop = 20
        self.pc = 1
        self.d, self.attr_keys = self.generate_d()  # distance matrix (d) & attributes corresponding to d

    def generate_d(self):
        v_bins = self.d_set.valid_bins
        # 1. Fetch valid bins group
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in v_bins[:, 0]]

        # 2. Initialize an empty d-matrix
        n = len(attr_keys)
        d = np.zeros((n, n), dtype=np.dtype('i8'))  # cumulative sum of all segments
        for i in range(n):
            for j in range(n):
                if GI.parse_gi(attr_keys[i]).attribute_col == GI.parse_gi(attr_keys[j]).attribute_col:
                    # Ignore similar attributes (+ or/and -)
                    continue
                else:
                    bin_1 = v_bins[i][1]
                    bin_2 = v_bins[j][1]
                    # Cumulative sum of all segments for 2x2 (all attributes) gradual items
                    d[i][j] += np.sum(np.multiply(bin_1, bin_2))
        # print(d)
        return d, attr_keys

    def run_genetic_algorithm(self):
        min_supp = self.d_set.thd_supp
        a = self.d_set.attr_size

        if self.d_set.no_bins:
            return []

        # 1. Remove d[i][j] < frequency-count of min_supp
        fr_count = ((min_supp * a * (a - 1)) / 2)
        self.d[self.d < fr_count] = 0

        # gene = self.build_gp_gene()
        # gp = self.decode_gp(gene)
        # print(gene)
        # if not(gp is None):
        #    print(gp.to_string())
        # else:
        #    print("Irrelevant gene")

        # Problem Information
        costfunc = self.cost_func

        # Parameters
        it_count = 0
        max_it = self.max_it
        npop = self.npop
        pc = self.pc
        nc = int(np.round(pc * npop / 2) * 2)

        # Empty Individual Template
        empty_individual = structure()
        empty_individual.gene = None
        empty_individual.cost = None

        # Best Solution Ever Found
        bestsol = empty_individual.deepcopy()
        bestsol.cost = np.inf

        # Initialize Population
        pop = empty_individual.repeat(npop)
        for i in range(npop):
            pop[i].gene = self.build_gp_gene()
            pop[i].cost = costfunc(self.decode_gp(pop[i].gene))
            if pop[i].cost < bestsol.cost:
                bestsol = pop[i].deepcopy()

        # Best Cost of Iteration
        bestcost = np.empty(max_it)
        bestgene = []
        bestpattern = []

        while it_count < max_it:

            popc = []
            for _ in range(nc // 2):
                # Select Parents
                q = np.random.permutation(npop)
                p1 = pop[q[0]]
                p2 = pop[q[1]]

                # Perform Crossover
                c1, c2 = self.crossover(p1, p2)

                # Perform Mutation
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)

                # Apply Bound
                # apply_bound(c1, varmin, varmax)
                # apply_bound(c2, varmin, varmax)

                # Evaluate First Offspring
                c1.cost = costfunc(self.decode_gp(c1.gene))
                if c1.cost < bestsol.cost:
                    bestsol = c1.deepcopy()

                # Evaluate Second Offspring
                c2.cost = costfunc(self.decode_gp(c2.gene))
                if c2.cost < bestsol.cost:
                    bestsol = c2.deepcopy()

                # Add Offsprings to popc
                popc.append(c1)
                popc.append(c2)

            # Merge, Sort and Select
            pop += popc
            pop = sorted(pop, key=lambda x: x.cost)
            pop = pop[0:npop]

            # Store Best Cost
            bestcost[it_count] = bestsol.cost
            bestgene.append(bestsol.gene)

            best_gp = self.decode_gp(bestsol.gene)
            best_gp.support = float(1 / bestsol.cost)
            is_present = GradGA.is_duplicate(best_gp, bestpattern)
            is_sub = GradGA.check_anti_monotony(bestpattern, best_gp, subset=True)
            if not (is_present or is_sub):
                bestpattern.append(best_gp)

            # Show Iteration Information
            print("Iteration {}: Best Cost = {}".format(it_count, bestcost[it_count]))
            it_count += 1

        # Output
        out = structure()
        out.pop = pop
        out.bestsol = bestsol
        out.bestcost = bestcost
        out.bestpattern = bestpattern

        return out

    def build_gp_gene(self):
        a = self.attr_keys
        temp_gene = np.random.choice(a=[0, 1], size=(len(a),))
        return temp_gene

    def decode_gp(self, gene):
        temp_gp = GP()
        if gene is None:
            return temp_gp
        for i in range(gene.size):
            gene_val = gene[i]
            if gene_val == 1:
                gi = GI.parse_gi(self.attr_keys[i])
                # if temp_gp.contains_attr(gi):
                    # return None
                #    continue
                # else:
                if not temp_gp.contains_attr(gi):
                    temp_gp.add_gradual_item(gi)
        return self.validate_gp(temp_gp)

    def cost_func(self, gp):
        if gp is None:
            return np.inf
        else:
            if gp.support <= self.d_set.thd_supp:
                return np.inf
            return float(1 / gp.support)

    def validate_gp(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = self.d_set.thd_supp
        n = self.d_set.attr_size
        gen_pattern = GP()
        bin_arr = np.array([])

        for gi in pattern.gradual_items:
            arg = np.argwhere(np.isin(self.d_set.valid_bins[:, 0], gi.gradual_item))
            if len(arg) > 0:
                i = arg[0][0]
                valid_bin = self.d_set.valid_bins[i]
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

    @staticmethod
    def crossover(p_1, p_2):
        c_1 = p_1.copy()
        c_2 = p_1.copy()
        choice = np.random.randint(2, size=c_1.gene.size).reshape(c_1.gene.shape).astype(bool)
        c_1.gene = np.where(choice, p_1.gene, p_2.gene)
        c_2.gene = np.where(choice, p_2.gene, p_1.gene)
        return c_1, c_2

    @staticmethod
    def mutate(p_x):
        p_y = p_x.copy()
        rand_val = np.random.randint(0, p_x.gene.shape[0])
        if p_y.gene[rand_val] == 0:
            p_y.gene[rand_val] = 1
        else:
            p_y.gene[rand_val] = 0
        return p_y

    @staticmethod
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

    @staticmethod
    def is_duplicate(pattern, lst_winners):
        for pat in lst_winners:
            if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                    set(pattern.inv_pattern()) == set(pat.get_pattern()):
                return True
        return False
