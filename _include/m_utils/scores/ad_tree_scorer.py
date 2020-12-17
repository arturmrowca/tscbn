from itertools import product
from math import log


class ADtreeScorer(object):

    def __init__(self, score, adtree):
        self.score = score
        self.adtree = adtree

    def local_score(self, variable, parent_set):
        # Missing Values:
        # This implementation handles missing values as additional value. Other approaches are approximating the
        # missing values or storing only records without missing values. The second idea would require a data
        # structure that is able to check for missing values and only return records and counts when all the
        # required variables take on true values. The implementation of the ADtree used here is not able to answer
        # queries like "V1_2 != ''" (no missing value for node V1_2). Therefore, already getting the number of
        # records N in which the variable and all variables in the parent set have no missing value is not possible.
        # The second approach implemented uses a pandas data frame and the pgmpy score calculators and is able to handle
        # missing values as truly missing values.
        parent_set = list(parent_set)  # assure fixed order of the variables
        score = 0
        value_frequencies = self.adtree.table(variable)
        N = sum([value_frequency[1] for value_frequency in value_frequencies])
        values = {value_frequency[0] for value_frequency in value_frequencies}
        k = (len(values) - 1)
        if self.score == 'BIC':
            k *= log(N) * 0.5
        ps_value_frequencies = {}
        ps_values = {}
        for parent_variable in parent_set:
            ps_value_frequencies.update({parent_variable: self.adtree.table(parent_variable)})
            pv_values = {ps_value_frequency[0] for ps_value_frequency in ps_value_frequencies.get(parent_variable)}
            ps_values.update({parent_variable: pv_values})
            k *= len(pv_values)
        score -= k
        for parent_set_value_combination in product(*(ps_values.values())):
            kwargs = {}  # collect arguments for ADtree lookup
            kwargs.update(list(zip(parent_set, parent_set_value_combination)))
            N_ij = self.adtree.count(**kwargs)
            for value in values:
                kwargs.update({variable: value})  # append variable constraint
                N_ijk = self.adtree.count(**kwargs)
                if N_ijk == 0:  # this case was not observed
                    continue
                score += N_ijk * log(N_ijk / N_ij)
            pass
        return score
