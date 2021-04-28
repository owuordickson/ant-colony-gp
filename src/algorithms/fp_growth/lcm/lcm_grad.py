"""
LCM: Linear time Closed item set Miner
as described in `http://lig-membres.imag.fr/termier/HLCM/hlcm.pdf`
URL: https://github.com/scikit-mine/scikit-mine/tree/master/skmine
Author: Rémi Adon <remi.adon@gmail.com>
License: BSD 3 clause

Modified by: Dickson Owuor <owuordickson@ieee.org>

"""

import numpy as np
from collections import defaultdict
from sortedcontainers import SortedDict
import gc
import multiprocessing as mp

from src.common.dataset_dfs import Dataset_dfs
from src.common.gp import GI, GP


class LCM_g:

    def __init__(self, file, min_supp=0.5, n_jobs=1, verbose=False):
        self.min_supp = min_supp  # provided by user
        self._min_supp = LCM_g.check_min_supp(self.min_supp)
        self.item_to_tids = None
        self.n_transactions = 0
        self.ctr = 0
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.d_set = Dataset_dfs(file, min_supp, eq=False)
        self.D = self.d_set.remove_inv_attrs(self.d_set.encode_data())
        self._fit()

    def _fit(self):
        self.n_transactions = 0  # reset for safety
        item_to_tids = defaultdict(set)
        # for transaction in D:
        for t in range(len(self.D)):
            transaction = self.D[t][2:]
            for item in transaction:
                item_to_tids[item].add(tuple(self.D[t][:2]))
            self.n_transactions += 1
        # print(D)
        # print(item_to_tids)

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.d_set.attr_size

        low_supp_items = [k for k, v in item_to_tids.items() if len(np.unique(np.array(list(v))[:, 0], axis=0)) < self._min_supp]
        for item in low_supp_items:
            del item_to_tids[item]

        self.item_to_tids = SortedDict(item_to_tids)
        self.D = None
        gc.collect()
        return self

    def fit_discover(self):
        # empty_df = pd.DataFrame(columns=['pattern', 'support', 'tids'])

        # reverse order of support
        supp_sorted_items = sorted(self.item_to_tids.items(), key=lambda e: len(e[1]), reverse=True)
        with mp.Pool(self.n_jobs) as pool:
            dfs = pool.map(self._explore_item, supp_sorted_items)

        # dfs.append(empty_df)  # make sure we have something to concat
        # df = pd.concat(dfs, axis=0, ignore_index=True)
        # if not return_tids:
        #     df.loc[:, 'support'] = df['tids'].map(len).astype(np.uint32)
        #    df.drop('tids', axis=1, inplace=True)
        # return df
        return dfs

    def _explore_item(self, obj):
        item = obj[0]
        tids = obj[1]
        it = self._inner(frozenset(), tids, item)
        # df = pd.DataFrame(data=it, columns=['pattern', 'support', 'tids'])
        # if self.verbose and not df.empty:
        #    print('LCM found {} new itemsets from item : {}'.format(len(df), item))
        # return df
        return list(it)

    def _inner(self, p, tids, limit):
        # project and reduce DB w.r.t P
        cp = (
            item for item, ids in reversed(self.item_to_tids.items())
            if tids.issubset(ids) if item not in p
        )

        max_k = next(cp, None)  # items are in reverse order, so the first consumed is the max

        if max_k and max_k == limit:
            p_prime = p | set(cp) | {max_k}  # max_k has been consumed when calling next()
            # sorted items in ouput for better reproducibility
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
                # yield pat.to_string(), pat.support, tids
                yield pat

            candidates = self.item_to_tids.keys() - p_prime
            candidates = candidates[:candidates.bisect_left(limit)]
            for new_limit in candidates:
                ids = self.item_to_tids[new_limit]
                new_limit_tids = tids.intersection(ids)
                supp = self.calculate_support(new_limit_tids)
                if supp >= self.min_supp:
                    yield from self._inner(p_prime, new_limit_tids, new_limit)

    def calculate_support(self, tids):
        if len(tids) > 1:
            x = np.unique(np.array(list(tids))[:, 0], axis=0)
            supp = len(x) / self.d_set.attr_size
            return supp
        else:
            return len(tids) / self.d_set.attr_size

    @staticmethod
    def check_min_supp(min_supp, accept_absolute=True):
        if isinstance(min_supp, int):
            if not accept_absolute:
                raise ValueError(
                    'Absolute support is prohibited, please provide a float value between 0 and 1'
                )
            if min_supp < 1:
                raise ValueError('Minimum support must be strictly positive')
        elif isinstance(min_supp, float):
            if min_supp < 0 or min_supp > 1:
                raise ValueError('Minimum support must be between 0 and 1')
        else:
            raise TypeError('Mimimum support must be of type int or float')
        return min_supp
