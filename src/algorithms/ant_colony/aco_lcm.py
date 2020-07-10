# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "09 July 2020"

Depth-First Search for gradual patterns (ACO-LCM)

"""

import numpy as np
from numpy import random as rand
import pandas as pd
from collections import defaultdict
from sortedcontainers import SortedDict

# from joblib import Parallel, delayed
from ..common.lcm_grad import LCM_g
from ..common.gp import GI, GP
from ..common.dataset_dfs import Dataset_dfs


class LcmACO(LCM_g):

    def __init__(self, f_path, min_supp):
        print("LcmACO: Version 1.0")
        self.min_supp = min_supp  # provided by user
        self._min_supp = LCM_g.check_min_supp(self.min_supp)
        self.item_to_tids = None
        self.n_transactions = 0
        self.ctr = 0
        # self.n_jobs = 1  # to be removed

        self.d_set = Dataset_dfs(f_path, min_supp, eq=False)
        self.d_set.init_gp_attributes()
        # self.p_matrix = np.ones((self.d_set.column_size, 3), dtype=np.int64)
        # self.d_set.reduce_data(p_matrix=self.p_matrix)
        self.d_set.reduce_data()
        self.size = self.d_set.attr_size
        # self.c_matrix = self.d_set.cost_matrix
        self.c_matrix = np.ones((self.size, self.size), dtype=np.float64)
        self.p_matrix = np.ones((self.size, self.size), dtype=np.int64)
        np.fill_diagonal(self.p_matrix, 0)
        self.e_factor = 0.1  # evaporation factor
        # self.large_tids = np.array([])
        # self.attr_index = self.d_set.attr_cols
        # self.e_factor = 0.1  # evaporation factor
        # print(self.d_set.cost_matrix)
        # print(self.d_set.encoded_data)
        # print(self.p_matrix)

    def _fit(self):
        D = self.d_set.encoded_data
        self.n_transactions = 0  # reset for safety
        item_to_tids = defaultdict(set)
        # for transaction in D:
        for t in range(len(D)):
            transaction = D[t][2:]
            for item in transaction:
                item_to_tids[item].add(tuple(D[t][:2]))
                # cost_matrix[D[t][0], D[t][1]] += 1
                # cost_matrix[D[t][1], D[t][0]] += 1
            self.n_transactions += 1
        # print(D)
        # print(cost_matrix)
        # print(item_to_tids)

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.size

        low_supp_items = [k for k, v in item_to_tids.items() if len(np.unique(np.array(list(v))[:, 0], axis=0)) < self._min_supp]
        for item in low_supp_items:
            del item_to_tids[item]

        tids = item_to_tids.values()
        for nodes in tids:
            idx = np.array(list(nodes))
            np.add.at(self.c_matrix, (idx[:, 0], idx[:, 1]), 1)

        # print(self.c_matrix)
        # self.large_tids = np.argwhere(self.count_matrix == self.count_matrix.max())
        # print(self.large_tids)
        self.c_matrix = 1 / self.c_matrix
        print(self.c_matrix)
        self.generate_random_node(0)
        print("\n\n")

        # self.item_to_tids = SortedDict(item_to_tids)
        self.item_to_tids = item_to_tids
        return self

    def fit_discover(self, return_tids=False):

        self._fit()
        # self.attr_index = np.array(self.item_to_tids.keys())
        empty_df = pd.DataFrame(columns=['pattern', 'support', 'tids'])

        # reverse order of support
        supp_sorted_items = sorted(self.item_to_tids.items(), key=lambda e: len(e[1]), reverse=True)
        # print(self.attr_index)
        print(supp_sorted_items)
        # dfs = Parallel(n_jobs=self.n_jobs, prefer='processes')(
        #    delayed(self._explore_item)(item, tids, 1) for item, tids in supp_sorted_items
            # delayed(self._explore_item)(item, tids, 1) for item, tids in supp_sorted_items if item == 2
        # )
        dfs = list()
        for item, tids in supp_sorted_items:
            dfs.append(self._explore_item(item, tids, 1))

        dfs.append(empty_df)  # make sure we have something to concat
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if not return_tids:
            # df.loc[:, 'support'] = df['tids'].map(len).astype(np.uint32)
            df.drop('tids', axis=1, inplace=True)
        return df

    def _explore_item(self, item, tids, no):
        it = self._inner(frozenset(), tids, item, no)
        df = pd.DataFrame(data=it, columns=['pattern', 'support', 'tids'])
        return df

    def _inner(self, p, tids, limit, no):
        print(str(no) + '. ' + str(p) + ' + ' + str(tids) + ' + ' + str(limit))
        # project and reduce DB w.r.t P
        cp = (
            item for item, ids in reversed(self.item_to_tids.items())
            if tids.issubset(ids) if item not in p
        )

        max_k = next(cp, None)  # items are in reverse order, so the first consumed is the max
        print(str(set(cp)) + ' + ' + str({max_k}))
        if max_k and max_k == limit:
            p_prime = p | set(cp) | {max_k}  # max_k has been consumed when calling next()
            # sorted items in ouput for better reproducibility
            print(str(p) + ' u ' + str(set(cp)) + ' u ' + str({max_k}) + ' = ' + str(p_prime) + '\n')
            raw_p = tuple(sorted(p_prime))
            pat = GP()
            for a in raw_p:
                if a < 0:
                    sym = '-'
                elif a > 0:
                    sym = '+'
                else:
                    sym = 'x'
                attr = abs(a) - 1
                pat.add_gradual_item(GI(attr, sym))
                pat.set_support(self.calculate_support(tids))
            # yield tuple(sorted(p_prime)), tids
            if len(raw_p) > 1:
                yield pat.to_string(), pat.support, tids

            candidates = self.item_to_tids.keys() - p_prime
            # print(candidates)
            # print(str(limit) + ' : ' + str(candidates.bisect_left(limit)))
            candidates = candidates[:candidates.bisect_left(limit)]
            # print(str(candidates) + '\n\n')
            for new_limit in candidates:
                ids = self.item_to_tids[new_limit]
                new_limit_tids = tids.intersection(ids)
                supp = self.calculate_support(new_limit_tids)
                if supp >= self.min_supp:
                    yield from self._inner(p_prime, new_limit_tids, new_limit, 2)

    def generate_random_node(self, i):
        C = self.c_matrix
        P = self.p_matrix
        n = self.size
        ph = P[i] * (1 / C[i])
        tot_sum = np.sum(ph)
        # x = float(rand.randint(1, n) / n)
        x = rand.uniform(0, 1)
        for j in range((i + 1), n):
            p_sum = np.sum(ph[(i+1):(j+1)])
            pr = p_sum / tot_sum
            # pr = ph[j] / tot_sum
            print(pr)
            if x < pr:
                print(tuple([i, j]))
                return tuple([i, j])
        return tuple([])
        #    print(pr)
        # for i in range(len(p)):
        #    for j in range((i+1), self.size)):
        #        p = p[i][j] * (1 / c[i][j])
        # print(self.large_tids)
        # p = self.p_matrix
        # c = self.c_matrix
        # attrs = self.attr_index.copy()
        # np.random.shuffle(attrs)
        # n = len(attrs)  # * 100
        # candidate = attrs[0]
        # pattern = GP()
        # for i in attrs:
            # i -= 1  # to take care of added
            # x = float(rand.randint(1, n) / n)
            # p0 = p[i][0] * (1 / c[i][0])
            # p1 = p[i][1] * (1 / c[i][1])
            # p2 = p[i][2] * (1 / c[i][2])
            # pos = float(p0 / (p0 + p1 + p2))
            # neg = float((p0 + p1) / (p0 + p1 + p2))
            # if x < pos:
            #    temp = GI(i, '+')
            # elif (x >= pos) and (x < neg):
            #    temp = GI(i, '-')
            # else:
            #     temp = GI(self.attr_index[i], 'x')
            #    continue
            # pattern.add_gradual_item(temp)
        # return candidate

    def deposit_pheromone(self, pattern):
        lst_attr = []
        for obj in pattern.gradual_items:
            # print(obj.attribute_col)
            attr = obj.attribute_col
            symbol = obj.symbol
            lst_attr.append(attr)
            i = attr
            if symbol == '+':
                self.p_matrix[i][0] += 1
            elif symbol == '-':
                self.p_matrix[i][1] += 1
        for index in self.item_to_tids.keys():
            if (int(index) - 1) not in lst_attr:
                i = int(index)
                self.p_matrix[i][2] += 1

    def validate_gp(self, pattern):
        pass

    @staticmethod
    def largest_index(a, k):
        # idx = np.argsort(a.ravel())[:-k - 1:-1]
        # return np.column_stack(np.unravel_index(idx, a.shape))
        idx = np.argpartition(a.ravel(), a.size - k)[-k:]
        return np.column_stack(np.unravel_index(idx, a.shape))
