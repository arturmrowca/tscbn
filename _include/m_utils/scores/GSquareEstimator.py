# A conditional independence test function for discrete data inspired by the implementation in
# https://pypi.python.org/pypi/pcalg/0.1.6

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import chi2

from general.log import Log as L


class GSquareEstimator():
    def __init__(self, adtree):
        self.adtree = adtree

    def test_conditional_independence(self, source, target, condition_set):
        adtree = self.adtree
        number_samples = adtree.count()
        source_table = adtree.table(source)
        source_values = [source_entry[0] for source_entry in source_table]
        target_table = adtree.table(target)
        target_values = [target_entry[0] for target_entry in target_table]
        dof = ((len(source_table) - 1) * (len(target_table) - 1)
               * np.prod(list(map(lambda x: len(adtree.table(x)), condition_set))))  # degrees of freedom
        if dof == 0:  # this is the case when source or target is constant
            L().log.warning('Zero degrees of freedom: Either source or target is constant!')
            pass
            return 0, 1, True  # p-value is 1
        row_size_required = 10 * dof  # test results are not really reliable if there are less than 10*dof samples
        sufficient_data = True
        if number_samples < row_size_required:
            L().log.warning(
                'Not enough samples. ' + str(number_samples) + ' is too small. Need ' + str(row_size_required)
                + '. G^2-Test may not be reliable.')
            sufficient_data = False
        pass
        g2 = 0

        # first case: empty condition set
        if len(condition_set) == 0:
            nij = pd.DataFrame(0, index=[entry[0] for entry in source_table],
                               columns=[entry[0] for entry in target_table])
            kwargs = {}  # collect arguments for ADtree lookup
            for source_value in source_values:
                for target_value in target_values:
                    kwargs.update({source: source_value, target: target_value})
                    nij.loc[source_value, target_value] = adtree.count(**kwargs)
                pass
            n_j = np.array([nij.sum(axis=1)]).T  # fix first variable and compute frequencies
            ni_ = np.array([nij.sum(axis=0)])  # fix second variable and compute frequencies
            expected_nij = n_j.dot(ni_) / number_samples  # expectation of nij
            ln_argument = nij.divide(expected_nij)  # compute argument for ln()
            ln_results = np.log(ln_argument)  # compute ln()
            g2 = np.nansum(nij.multiply(2 * ln_results))  # compute sum of lns
        pass

        # second case: non-empty condition set
        if len(condition_set) > 0:
            # calculate number of possible combinations of the values in the condition set
            prod_levels = np.prod(list(map(lambda x: len(adtree.table(x)), condition_set)))
            condition_set_values = [list([entry[0] for entry in adtree.table(node)]) for node in condition_set]
            cs_value_combinations = list(product(*condition_set_values))
            nij_ = [
                pd.DataFrame(0, index=[entry[0] for entry in source_table],
                             columns=[entry[0] for entry in target_table])
                for _ in cs_value_combinations]
            nijk = pd.concat(nij_, keys=cs_value_combinations)  # type: pd.DataFrame

            # fill in frequencies
            kwargs = {}  # collect arguments for ADtree lookup
            for source_value in source_values:
                for target_value in target_values:
                    for cs_value_combination in cs_value_combinations:
                        kwargs.update({source: source_value, target: target_value})
                        kwargs.update(zip(condition_set, cs_value_combination))
                        nijk.xs(cs_value_combination).loc[source_value, target_value] = adtree.count(**kwargs)
                    pass
                pass
            pass

            ni__ = np.ndarray((len(source_table), prod_levels))
            n_j_ = np.ndarray((len(target_table), prod_levels))
            for value_combination in cs_value_combinations:
                index = cs_value_combinations.index(value_combination)
                ni__[:, index] = nijk.xs(value_combination).sum(axis=1)
                n_j_[:, index] = nijk.xs(value_combination).sum(axis=0)
            pass
            n__k = n_j_.sum(axis=0)
            for value_combination in cs_value_combinations:
                index = cs_value_combinations.index(value_combination)
                ni_k = np.array([ni__[:, index]]).T  # fix condition set and compute source frequencies
                n_jk = np.array([n_j_[:, index]])  # fix condition set and compute target frequencies
                expected_nijk = ni_k.dot(n_jk) / n__k[index]  # expected frequencies for nijk
                ln_argument = nijk.xs(value_combination) / expected_nijk  # argument for ln()
                ln_results = np.log(ln_argument)  # compute ln()
                g2 += np.nansum(nijk.xs(value_combination).multiply(2 * ln_results))
            pass
        pass

        p_val = chi2.sf(g2, dof)  # compute p-value by using the chi^2 distribution
        return g2, p_val, sufficient_data
