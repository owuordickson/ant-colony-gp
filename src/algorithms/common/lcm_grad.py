"""
LCM: Linear time Closed item set Miner
as described in `http://lig-membres.imag.fr/termier/HLCM/hlcm.pdf`
Author: Rémi Adon <remi.adon@gmail.com>
License: BSD 3 clause

Modified by: Dickson Owuor <owuordickson@ieee.org>

"""

from collections import defaultdict
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sortedcontainers import SortedDict
from roaringbitmap import RoaringBitmap as RB

from .dataset_dfs import Dataset_dfs


class LCM:

    def __init__(self, min_supp, n_jobs, verbose):
        # LCM.check_min_supp(min_supp)
        self.min_supp = min_supp  # provided by user
        self._min_supp = LCM.check_min_supp(self.min_supp)
        self.item_to_tids = None
        self.n_transactions = 0
        self.ctr = 0
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit(self, D):
        self.n_transactions = 0  # reset for safety
        item_to_tids = defaultdict(RB)
        for transaction in D:
            for item in transaction:
                item_to_tids[item].add(self.n_transactions)
            self.n_transactions += 1
        print(D)
        print(item_to_tids)

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.n_transactions

        low_supp_items = [k for k, v in item_to_tids.items() if len(v) < self._min_supp]
        for item in low_supp_items:
            del item_to_tids[item]

        self.item_to_tids = SortedDict(item_to_tids)
        return self

    def fit_discover(self, D, return_tids=False):

        self._fit(D)

        empty_df = pd.DataFrame(columns=['itemset', 'tids'])

        # reverse order of support
        supp_sorted_items = sorted(self.item_to_tids.items(), key=lambda e: len(e[1]), reverse=True)

        dfs = Parallel(n_jobs=self.n_jobs, prefer='processes')(
            delayed(self._explore_item)(item, tids) for item, tids in supp_sorted_items
        )

        dfs.append(empty_df) # make sure we have something to concat
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if not return_tids:
            df.loc[:, 'support'] = df['tids'].map(len).astype(np.uint32)
            df.drop('tids', axis=1, inplace=True)
        return df

    def fit_transform(self, D):

        patterns = self.fit_discover(D, return_tids=True)
        tid_s = patterns.set_index('itemset').tids
        by_supp = tid_s.map(len).sort_values(ascending=False)
        patterns = tid_s.reindex(by_supp.index)

        shape = (self.n_transactions, len(self.item_to_tids))
        mat = np.zeros(shape, dtype=np.uint32)

        df = pd.DataFrame(mat, columns=self.item_to_tids.keys())
        for pattern, tids in tid_s.iteritems():
            df.loc[tids, pattern] = len(tids)  # fill with support
        return df

    def _explore_item(self, item, tids):
        it = self._inner(frozenset(), tids, item)
        df = pd.DataFrame(data=it, columns=['itemset', 'tids'])
        if self.verbose and not df.empty:
            print('LCM found {} new itemsets from item : {}'.format(len(df), item))
        return df

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
            yield tuple(sorted(p_prime)), tids

            candidates = self.item_to_tids.keys() - p_prime
            candidates = candidates[:candidates.bisect_left(limit)]
            for new_limit in candidates:
                ids = self.item_to_tids[new_limit]
                # print(str(tids) + ' + ' + str(ids) + ' = ' + str(tids.intersection_len(ids)))
                # if tids.intersection_len(ids) >= self._min_supp:
                x = tids.intersection(ids)
                if len(x) > 1:
                    x = np.unique(np.array(list(x))[:, 0], axis=0)
                if len(x) >= self._min_supp:
                    new_limit_tids = tids.intersection(ids)
                    yield from self._inner(p_prime, new_limit_tids, new_limit)

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


class LCM_g(LCM):

    def __init__(self, file, min_supp=0.5, n_jobs=1, verbose=False):
        super().__init__(min_supp, n_jobs, verbose)
        self.d_set = Dataset_dfs(file, min_supp, eq=False)
        self.d_set.init_gp_attributes()
        self.d_set.reduce_data()

    def _fit(self, D):
        # D = self.d_set.encoded_data
        self.attr_size = 5
        self.n_transactions = 0  # reset for safety
        item_to_tids = defaultdict(set)
        # for transaction in D:
        for t in range(len(D)):
            transaction = D[t][2:]
            for item in transaction:
                item_to_tids[item].add(tuple(D[t][:2]))
            self.n_transactions += 1
        print(D)
        print(item_to_tids)

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.attr_size #self.n_transactions
            print(self._min_supp)

        # for k, v in item_to_tids.items():
        #    print(str(v) + ' = ' + str(len(v)))
            # print(np.array(list(v)).shape)
        #    print(np.unique(np.array(list(v))[:, 0], axis=0))
        # if len(v) < self._min_supp

        low_supp_items = [k for k, v in item_to_tids.items() if len(np.unique(np.array(list(v))[:, 0], axis=0)) < self._min_supp]
        for item in low_supp_items:
            del item_to_tids[item]

        self.item_to_tids = SortedDict(item_to_tids)
        return self
