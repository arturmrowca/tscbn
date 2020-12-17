# A conditional independence test function for discrete data inspired by the implementation in
# https://pypi.python.org/pypi/pcalg/0.1.6

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import chi2

from _include.toolkit import suppress_stderr
from general.log import Log as L


def g_square_test(sequences, source, target, condition_set):
    '''
    Takes the sequences and filters the non-relevant events (not source, target, or in the condition set). Then a
    conditional independence test is performed to check whether source and target are independent given the variables in
    the condition set. Returns a p-value to indicate the result.
    '''

    # store all events that are required to calculate the p-value
    required_nodes = list(set().union([source, target], [node for node in condition_set]))
    levels = dict([(node, set()) for node in required_nodes])  # store all observed values for each of the events
    reduced_sequences = []  # sequences only containing the required events
    for sequence in sequences:
        reduced_sequence = []
        for event in sequence:
            if event[0] in required_nodes:
                reduced_sequence.append(event)  # append to reduced sequence if event in required nodes
                levels.get(event[0]).add(event[1])  # store value
            pass
        if len(reduced_sequence) != len(required_nodes):  # check if all required nodes are present
            continue  # no -> skip this sequence
        reduced_sequences.append(reduced_sequence)  # yes -> add to the reduced sequences
    data_list = []  # store data rows
    for sequence in reduced_sequences:
        data_row = [np.NaN for _ in required_nodes]
        for event in sequence:
            data_row[required_nodes.index(event[0])] = event[1]
        data_list.append(data_row)
    data = pd.DataFrame(data_list, columns=required_nodes)  # create data frame that can be accessed by the event name

    number_samples = data.shape[0]
    dof = ((len(levels.get(source)) - 1) * (len(levels.get(target)) - 1)
           * np.prod(list(map(lambda x: len(levels.get(source)), condition_set))))  # degrees of freedom
    if dof == 0:  # this is the case when source or target is constant
        return 1  # p-value is 1
    row_size_required = 10 * dof  # test results are not really reliable if there are less than 10*dof samples
    if number_samples < row_size_required:
        #L().log.warning('Not enough samples. ' + str(number_samples) + ' is too small. Need ' + str(row_size_required))
        return 1
    pass
    g2 = 0

    # first case: empty condition set
    if len(condition_set) == 0:
        nij = pd.DataFrame(0, index=levels.get(source), columns=levels.get(target))  # observed frequencies
        for row_index in range(number_samples):  # fill nij
            i = data.loc[row_index, source]
            j = data.loc[row_index, target]
            nij.loc[i, j] += 1

        n_j = np.array([nij.sum(axis=1)]).T  # fix first variable and compute frequencies
        ni_ = np.array([nij.sum(axis=0)])  # fix second variable and compute frequencies
        expected_nij = n_j.dot(ni_) / number_samples  # expectation of nij
        ln_argument = nij.divide(expected_nij)  # compute argument for ln()
        with suppress_stderr():
            ln_results = np.log(ln_argument)  # compute ln()
        g2 = np.nansum(nij.multiply(2 * ln_results))  # compute sum of lns
    pass

    # second case: non-empty condition set
    if len(condition_set) > 0:
        # calculate number of possible combinations of the values in the condition set
        prod_levels = np.prod(list(map(lambda x: len(levels.get(source)), condition_set)))
        condition_set_values = [list(levels.get(node)) for node in condition_set]
        cs_value_combination = list(product(*condition_set_values))
        nij_ = [pd.DataFrame(0, index=levels.get(source), columns=levels.get(target)) for _ in cs_value_combination]
        nijk = pd.concat(nij_, keys=cs_value_combination)  # type: pd.DataFrame

        # count observed frequencies
        for row_index in range(data.shape[0]):
            source_value = data.loc[row_index, source]
            target_value = data.loc[row_index, target]
            condition_value = tuple()
            for node in condition_set:
                condition_value += (data.loc[row_index, node],)
            nijk.xs(condition_value).loc[source_value, target_value] += 1
        pass

        ni__ = np.ndarray((len(levels.get(source)), prod_levels))
        n_j_ = np.ndarray((len(levels.get(target)), prod_levels))
        for value_combination in cs_value_combination:
            index = cs_value_combination.index(value_combination)
            ni__[:, index] = nijk.xs(value_combination).sum(axis=1)
            n_j_[:, index] = nijk.xs(value_combination).sum(axis=0)
        pass
        n__k = n_j_.sum(axis=0)
        for value_combination in cs_value_combination:
            index = cs_value_combination.index(value_combination)
            ni_k = np.array([ni__[:, index]]).T  # fix condition set and compute source frequencies
            n_jk = np.array([n_j_[:, index]])  # fix condition set and compute target frequencies
            expected_nijk = ni_k.dot(n_jk) / n__k[index]  # expected frequencies for nijk
            ln_argument = nijk.xs(value_combination) / expected_nijk  # argument for ln()
            ln_results = np.log(ln_argument)  # compute ln()
            g2 += np.nansum(nijk.xs(value_combination).multiply(2 * ln_results))
        pass
    pass

    #L().log.info('G2 = ' + str(g2))
    p_val = chi2.sf(g2, dof)  # compute p-value by using the chi^2 distribution
    #L().log.info('p_val = ' + str(p_val))
    return p_val
