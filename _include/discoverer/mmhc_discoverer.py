from itertools import combinations

from pgmpy.base import UndirectedGraph
from pgmpy.estimators.base import BaseEstimator

from _include.discoverer.hill_climb_discoverer import HillClimbDiscoverer
from _include.discoverer.naive_discoverer import NaiveDiscoverer


class MMHCDiscoverer(NaiveDiscoverer):
    """
    This class is used to call the pgmpy MMHC (max-min hill climb) structure discoverer using the data sampled by the
    TSCBN generator. The approach is described in
    # Tsamardinos, I. ; Brown, L. E. ; Aliferis, C. F.: The Max-Min Hill-Climbing Bayesian Network Structure Learning
    # Algorithm. In: Machine Learning Bd. 65(1), Springer, 2006, S. 31-78
    """

    def __init__(self, score='BIC', alpha=0.05, tabu_length=0, max_reach=2, **kwargs):
        """
        :param score: scoring method (BIC, AIC, K2 or Bdeu)
        :param alpha: significance level
        :param tabu_length: The last `tabu_length` graph modifications cannot be reversed during the search procedure.
            This serves to enforce a wider exploration of the search space.
        :param max_reach: maximum size of condition sets in the CI test
            Very important parameter regarding the runtime of the algorithm! (max_reach=2 already relatively slow)
        :param kwargs: further keyworded arguments
        """
        super(MMHCDiscoverer, self).__init__(**kwargs)
        self.score = score
        self.alpha = alpha
        self.tabu_length = tabu_length
        self.max_reach = max_reach
        self.p_val_cache = {}  # map from X, Y, Zs to p_val

    def discover_structure_from_statistics(self, data, nodes):
        """
        :param data: pandas dataframe that contains the data
        :param nodes: all signal_occurrence values that appear in the data set
        :return: list of edges
        """
        skel = self.mmpc(data, nodes)  # determine skeleton
        hc = HillClimbDiscoverer(score=self.score, tabu_length=self.tabu_length)  # score-based hill climb on skeleton
        edges = hc.discover_structure_from_statistics(data, nodes, white_list=skel.to_directed().edges())
        return edges

    def mmpc(self, data, nodes):
        """
        Estimates a graph skeleton (UndirectedGraph) for the data set, using the MMPC (max-min parents-and-children) algorithm.
        :return: graph skeleton
        """

        def is_independent(X, Y, Zs, cb_estimator):
            """
            Returns result of hypothesis test for the null hypothesis that
            X _|_ Y | Zs, using a chi2 statistic and threshold `significance_level`.
            """
            if (tuple(sorted([X, Y])), tuple(sorted(Zs))) in self.p_val_cache:
                p_value, sufficient_data = self.p_val_cache.get((tuple(sorted([X, Y])), tuple(sorted(Zs))))
            else:
                chi2, p_value, sufficient_data = cb_estimator.test_conditional_independence(X, Y, Zs)
                self.p_val_cache.update({(tuple(sorted([X, Y])), tuple(sorted(Zs))): (p_value, sufficient_data)})
            return p_value >= self.alpha and sufficient_data

        def assoc(X, Y, Zs, cb_estimator):
            """
            Measure for (conditional) association between variables. Use negative
            p-value of independence test.
            """
            if (tuple(sorted([X, Y])), tuple(sorted(Zs))) in self.p_val_cache:
                p_value, sufficient_data = self.p_val_cache.get((tuple(sorted([X, Y])), tuple(sorted(Zs))))
            else:
                chi2, p_value, sufficient_data = cb_estimator.test_conditional_independence(X, Y, Zs)
                self.p_val_cache.update({(tuple(sorted([X, Y])), tuple(sorted(Zs))): (p_value, sufficient_data)})
            return 1 - p_value

        def min_assoc(X, Y, Zs, cb_estimator):
            """
            Minimal association of X, Y given any subset of Zs.
            """
            min_association = float('inf')
            for size in range(min(self.max_reach, len(Zs)) + 1):
                partial_min_association = min(
                    assoc(X, Y, Zs_subset, cb_estimator) for Zs_subset in combinations(Zs, size))
                if partial_min_association < min_association:
                    min_association = partial_min_association
            return min_association

        def max_min_heuristic(X, Zs):
            """
            Finds variable that maximizes min_assoc with `node` relative to `neighbors`.
            """
            max_min_assoc = 0
            best_Y = None

            for Y in set(nodes) - set(Zs + [X]):
                min_assoc_val = min_assoc(X, Y, Zs, cb_estimator)
                if min_assoc_val >= max_min_assoc:
                    best_Y = Y
                    max_min_assoc = min_assoc_val

            return best_Y, max_min_assoc

        cb_estimator = BaseEstimator(data=data, complete_samples_only=False)

        # Find parents and children for each node
        neighbors = dict()
        for node in nodes:
            neighbors[node] = []

            # Forward Phase
            while True:
                new_neighbor, new_neighbor_min_assoc = max_min_heuristic(node, neighbors[node])
                if new_neighbor_min_assoc > 0:
                    neighbors[node].append(new_neighbor)
                else:
                    break

            # Backward Phase
            for neigh in neighbors[node]:
                other_neighbors = [n for n in neighbors[node] if n != neigh]
                sep_sets = [sep_set for sep_set_size in range(min(self.max_reach, len(other_neighbors)) + 1) for sep_set
                            in combinations(other_neighbors, sep_set_size)]
                for sep_set in sep_sets:
                    if is_independent(node, neigh, sep_set, cb_estimator):
                        neighbors[node].remove(neigh)
                        break

        # correct for false positives
        for node in nodes:
            for neigh in neighbors[node]:
                if node not in neighbors[neigh]:
                    neighbors[node].remove(neigh)

        skel = UndirectedGraph()
        skel.add_nodes_from(nodes)
        for node in nodes:
            skel.add_edges_from([(node, neigh) for neigh in neighbors[node]])

        return skel
