from itertools import chain, combinations

from pgmpy.estimators.BdeuScore import BdeuScore
from pgmpy.estimators.BicScore import BicScore
from pgmpy.estimators.K2Score import K2Score

from _include.m_utils.adtree import ADTree
from _include.m_utils.scores.AicScore import AicScore
from _include.m_utils.scores.ad_tree_scorer import ADtreeScorer
from general.log import Log as L


def powerset(iterable):
    """
    Returns the power set of the elements in iterable. Sets are ordered in increasing size, so empty set first,
    then one-element sets and so on.
    :param iterable: iterable of elements
    :return: power set of the elements as iterable of tuples
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class OptimalPSCalculator(object):
    """
    Takes a score name and an ADtree or pandas dataframe and provides methods to determine the optimal parent set for a
    variable.
    """

    def __init__(self, score, data):
        """
        :param score: score that should be used to determine the optimal parent set
        :param data: ADtree or pandas dataframe that contains the dataset counts
        """
        self.score = score
        self.data = data
        if isinstance(self.data, ADTree):
            assert self.score in ['BIC', 'AIC'], 'Only BIC and AIC scores implemented for ADtree.'
            self.scorer = ADtreeScorer(score=score, adtree=data)
        elif self.score == 'BIC':
            self.scorer = BicScore(data=self.data, complete_samples_only=False)
        elif self.score == 'K2':
            self.scorer = K2Score(data=self.data, complete_samples_only=False)
        elif self.score == 'Bdeu':
            self.scorer = BdeuScore(data=self.data, complete_samples_only=False)
        elif self.score == 'AIC':
            self.scorer = AicScore(data=self.data, complete_samples_only=False)
        else:
            assert False, 'Unknown score passed.'
        self.optimal_parent_set_cache = {}  # map from (variable, potential_parents) to (optimal_parents, score)
        self.score_cache = {}  # map from (variable, parent_set) to (optimal_parents, score)

    def get_optimal_parent_set(self, variable, potential_parents, approach='parent_graph', return_score=False):
        if approach == 'exhaustive':
            # checks all combinations of nodes, so all POPS
            return self.exhaustive_approach(variable, potential_parents, return_score)
        elif approach == 'parent_graph':
            return self.parent_graph_approach(variable, potential_parents, return_score)
        else:
            assert False, 'Unknown score calculation approach passed.'

    def exhaustive_approach(self, variable, potential_parents, return_score=False):
        """
        This method determines the incoming edges of the passed variable in a exhaustive way. Method enables
        parallelisation of score-based approach (the parent set of each variable can be determined independently
        from the other variables.
        :param variable: parent set for this variable is determined
        :param potential_parents: set of pontial
        :param return_score: If set to true, the additionally the optimal score is returned.
        :return: list of edges from optimal parent set to the variable
        """
        # check if optimal parent set is cached
        if (variable, tuple(sorted(potential_parents))) in self.optimal_parent_set_cache:
            #L().log.debug(
            #    'Optimal ' + self.score + ' score for node ' + str(variable) + ' with potential parents ' + str(
            #        potential_parents) + ' found in cache.')
            if return_score:
                return self.optimal_parent_set_cache.get((variable, tuple(sorted(potential_parents))))
            return self.optimal_parent_set_cache.get((variable, tuple(sorted(potential_parents))))[0]
        # optimal parent set was not cached -> calculate it
        optimal_score = float('inf')
        optimal_parent_set = set()
        for parent_set in powerset(potential_parents):
            # Remark: runtime of exhaustive approach easily explodes due to power set
            # check if score is cached
            if (variable, tuple(sorted(parent_set))) in self.score_cache:
                #L().log.debug(self.score + ' score for node ' + str(variable) + ' with parents ' + str(
                #    potential_parents) + ' found in cache.')
                score = self.score_cache.get((variable, tuple(sorted(parent_set))))
            else:
                # score set was not cached -> calculate it
                score = self.scorer.local_score(variable, list(parent_set))
                #L().log.debug(self.score + ' score for node ' + str(variable) + ' with parent set ' + str(
                #    parent_set) + ': ' + str(score))
                self.score_cache.update({(variable, tuple(sorted(parent_set))): score})
            if score > optimal_score:
                optimal_score = score
                optimal_parent_set = parent_set
            pass
        self.optimal_parent_set_cache.update({(variable, tuple(sorted(potential_parents))): (
            tuple(sorted(optimal_parent_set)), optimal_score)})  # cache results
        if return_score:
            return optimal_parent_set, optimal_score
        return optimal_parent_set

    def parent_graph_approach(self, variable, potential_parents, return_score=False):
        """
        This method determines the incoming edges of the passed variable using the parent graph approach. Method
        enables parallelisation of score-based approach (the parent set of each variable can be determined
        independently from the other variables.
        # Ideas from following paper:
        # Changhe Yuan, Brandon Malone, and Xiaojian Wu. "Learning Optimal Bayesian Networks Using A* Search".
        # In: Proceedings of the 22nd International Joint Conference on Artificial Intelligence (IJCAI-11).
        # Helsinki, Finland, 2011, pp. 2186-2191.
        :param variable: parent set for this variable is determined
        :param potential_parents: set of pontial
        :param return_score: If set to true, the additionally the optimal score is returned.
        :return: list of edges from optimal parent set to the variable
        """
        # check if optimal parent set is cached
        if (variable, tuple(sorted(potential_parents))) in self.optimal_parent_set_cache:
            #L().log.debug(
            #    'Optimal ' + self.score + ' score for node ' + str(variable) + ' with potential parents ' + str(
            #        potential_parents) + ' found in cache.')
            if return_score:
                return self.optimal_parent_set_cache.get((variable, tuple(sorted(potential_parents))))
            return self.optimal_parent_set_cache.get((variable, tuple(sorted(potential_parents))))[0]
        # optimal parent set was not cached -> calculate it
        optimal_score = -float('inf')
        optimal_ps_size_scores = {}  # store optimal scores for the different sizes of the parent sets
        optimal_parent_set = set()
        non_optimal_subsets = set()  # remember non optimal subsets
        non_optimal_parents = set()  # runtime optimization: remember non optimal parent sets of size 1
        open_list = [()]  # open list to store nodes that have to be expanded further
        generated_node_size = 0
        improved_score = False  # runtime optimization: remember if at least one new parent set improved the score
        while open_list:
            parent_set = open_list.pop(0)  # pop node from open list
            # check if score is cached
            if (variable, tuple(sorted(parent_set))) in self.score_cache:
                #L().log.debug(self.score + ' score for node ' + str(variable) + ' with parents ' + str(
                #    parent_set) + ' found in cache.')
                score = self.score_cache.get((variable, tuple(sorted(parent_set))))
            else:
                # score set was not cached -> calculate it
                score = self.scorer.local_score(variable, list(parent_set))
                #L().log.debug(self.score + ' score for node ' + str(variable) + ' with parent set ' + str(
                #    parent_set) + ': ' + str(score))
                self.score_cache.update({(variable, tuple(sorted(parent_set))): score})
            if score > optimal_score:
                improved_score = True
                optimal_score = score
                optimal_parent_set = parent_set
            ps_size = len(parent_set)
            if str(ps_size) not in optimal_ps_size_scores or score > optimal_ps_size_scores.get(str(ps_size)):
                optimal_ps_size_scores.update({str(ps_size): score})
            if ps_size > 0 and score <= optimal_ps_size_scores.get(str(ps_size - 1)):
                if ps_size == 1:  # runtime optimization: remember non optimal parent sets of size 1
                    non_optimal_parents.add(parent_set[0])
                non_optimal_subsets.add(parent_set)
            if not open_list:  # while loop reached new graph level --> generate next level nodes
                if not improved_score:
                    #L().log.debug('No parent_set of size ' + str(
                    #    generated_node_size) + ' improved the score. Stop for this parent graph.')
                    break
                generated_node_size += 1
                improved_score = False
                # runtime optimization: filter potential parents
                filtered_potential_parents = set(potential_parents).difference(non_optimal_parents)
                for potential_parent_set in combinations(filtered_potential_parents, generated_node_size):
                    # if parent_set contains a non-optimal parent set, then it cannot be optimal either [Theorem 2]
                    if any(not set(non_optimal_subset).difference(set(potential_parent_set)) for non_optimal_subset in
                           non_optimal_subsets):
                        continue  # this parent_set cannot be optimal, so do not calculate the score
                    open_list.append(potential_parent_set)
                pass
            pass
        self.optimal_parent_set_cache.update({(variable, tuple(sorted(potential_parents))): (
            tuple(sorted(optimal_parent_set)), optimal_score)})  # cache results
        if return_score:
            return optimal_parent_set, optimal_score
        return optimal_parent_set
