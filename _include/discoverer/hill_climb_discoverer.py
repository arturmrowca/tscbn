from itertools import permutations

import networkx as nx
from pgmpy.estimators.BdeuScore import BdeuScore
from pgmpy.estimators.BicScore import BicScore
from pgmpy.estimators.K2Score import K2Score
from pgmpy.models import BayesianModel
from _include.discoverer.naive_discoverer import NaiveDiscoverer
from _include.m_utils.scores.AicScore import AicScore


class HillClimbDiscoverer(NaiveDiscoverer):
    """
    This class is used to call the pgmpy hill climb structure discoverer using the data sampled by the TSCBN generator.
    Remark: Using networkx 2.x will lead to some errors in the original HillClimbSearch class. All occurrences of
    model.edges() have to be replaced by list(model.edges()) and model.get_parents(Y) by list(model.get_parents(Y)).
    Code is mostly copied from package pgmpy. Implementation is extended by a score cache.
    """

    def __init__(self, score='BIC', tabu_length=0, max_in_degree=None, **kwargs):
        """
        :param score: scoring method (BIC, AIC, K2 or Bdeu)
        :param tabu_length: The last `tabu_length` graph modifications cannot be reversed during the search procedure.
            This serves to enforce a wider exploration of the search space.
        :param max_in_degree: all nodes have at most `max_in_degree` parents
        :param kwargs: further keyworded arguments
        """
        super(HillClimbDiscoverer, self).__init__(**kwargs)
        self.score = score
        self.tabu_length = tabu_length
        self.max_in_degree = max_in_degree
        self.score_cache = {}  # map from (variable, parent_set) to score

    def discover_structure_from_statistics(self, data, nodes, black_list=None, white_list=None):
        """
        :param data: pandas dataframe that contains the data
        :param nodes: all signal_occurrence values that appear in the data set
        :param black_list: If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None
        :param white_list: If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
        :return: list of edges
        """
        if self.score == 'BIC':
            scoring_method = BicScore(data=data, complete_samples_only=False)
        elif self.score == 'K2':
            scoring_method = K2Score(data=data, complete_samples_only=False)
        elif self.score == 'Bdeu':
            scoring_method = BdeuScore(data=data, complete_samples_only=False)
        elif self.score == 'AIC':
            scoring_method = AicScore(data=data, complete_samples_only=False)
        else:
            assert False, 'Unknown score passed.'

        epsilon = 1e-8
        start = BayesianModel()
        start.add_nodes_from(nodes)

        tabu_list = []
        current_model = start

        while True:
            best_score_delta = 0
            best_operation = None

            for operation, score_delta in self._legal_operations(current_model, scoring_method, nodes, tabu_list,
                                                                 black_list, white_list):
                if score_delta > best_score_delta:
                    best_operation = operation
                    best_score_delta = score_delta

            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == '+':
                current_model.add_edge(*best_operation[1])
                tabu_list = ([('-', best_operation[1])] + tabu_list)[:self.tabu_length]
            elif best_operation[0] == '-':
                current_model.remove_edge(*best_operation[1])
                tabu_list = ([('+', best_operation[1])] + tabu_list)[:self.tabu_length]
            elif best_operation[0] == 'flip':
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list = ([best_operation] + tabu_list)[:self.tabu_length]

        edges = [list(edge) for edge in current_model.edges()]
        return edges

    def _legal_operations(self, model, scoring_method, nodes, tabu_list, black_list=None, white_list=None):
        """
        Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Fridman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_in_degree` is provided, only modifications that keep the number
        of parents for each node below `max_in_degree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        local_score = scoring_method.local_score
        potential_new_edges = (set(permutations(nodes, 2)) -
                               set(list(model.edges())) -
                               set([(Y, X) for (X, Y) in list(model.edges())]))

        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(list(model.edges()) + [(X, Y)])):
                operation = ('+', (X, Y))
                if (operation not in tabu_list and (black_list is None or (X, Y) not in black_list) and (
                        white_list is None or (X, Y) in white_list)):
                    old_parents = list(model.get_parents(Y))
                    new_parents = old_parents + [X]
                    if self.max_in_degree is None or len(new_parents) <= self.max_in_degree:
                        if (Y, tuple(sorted(new_parents))) in self.score_cache:
                            new_score = self.score_cache.get((Y, tuple(sorted(new_parents))))
                        else:
                            new_score = local_score(Y, new_parents)
                            self.score_cache.update({(Y, tuple(sorted(new_parents))): new_score})  # cache score
                        if (Y, tuple(sorted(old_parents))) in self.score_cache:
                            old_score = self.score_cache.get((Y, tuple(sorted(old_parents))))
                        else:
                            old_score = local_score(Y, old_parents)
                            self.score_cache.update({(Y, tuple(sorted(old_parents))): old_score})  # cache score
                        score_delta = new_score - old_score
                        yield (operation, score_delta)
                    pass
                pass
            pass
        pass

        for (X, Y) in list(model.edges()):  # (2) remove single edge
            operation = ('-', (X, Y))
            if operation not in tabu_list:
                old_parents = list(model.get_parents(Y))
                new_parents = old_parents[:]
                new_parents.remove(X)
                if (Y, tuple(sorted(new_parents))) in self.score_cache:
                    new_score = self.score_cache.get((Y, tuple(sorted(new_parents))))
                else:
                    new_score = local_score(Y, new_parents)
                    self.score_cache.update({(Y, tuple(sorted(new_parents))): new_score})  # cache score
                if (Y, tuple(sorted(old_parents))) in self.score_cache:
                    old_score = self.score_cache.get((Y, tuple(sorted(old_parents))))
                else:
                    old_score = local_score(Y, old_parents)
                    self.score_cache.update({(Y, tuple(sorted(old_parents))): old_score})  # cache score
                score_delta = new_score - old_score
                yield (operation, score_delta)
                pass
            pass
        pass

        for (X, Y) in list(model.edges()):  # (3) flip single edge
            new_edges = list(model.edges()) + [(Y, X)]
            new_edges.remove((X, Y))
            if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                operation = ('flip', (X, Y))
                if (operation not in tabu_list and ('flip', (Y, X)) not in tabu_list and (
                        black_list is None or (X, Y) not in black_list) and (
                        white_list is None or (X, Y) in white_list)):
                    old_X_parents = list(model.get_parents(X))
                    old_Y_parents = list(model.get_parents(Y))
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if self.max_in_degree is None or len(new_X_parents) <= self.max_in_degree:
                        if (X, tuple(sorted(new_X_parents))) in self.score_cache:
                            new_X_score = self.score_cache.get((X, tuple(sorted(new_X_parents))))
                        else:
                            new_X_score = local_score(X, new_X_parents)
                            self.score_cache.update({(X, tuple(sorted(new_X_parents))): new_X_score})  # cache score
                        if (Y, tuple(sorted(new_Y_parents))) in self.score_cache:
                            new_Y_score = self.score_cache.get((Y, tuple(sorted(new_Y_parents))))
                        else:
                            new_Y_score = local_score(Y, new_Y_parents)
                            self.score_cache.update({(Y, tuple(sorted(new_Y_parents))): new_Y_score})  # cache score
                        if (X, tuple(sorted(old_X_parents))) in self.score_cache:
                            old_X_score = self.score_cache.get((X, tuple(sorted(old_X_parents))))
                        else:
                            old_X_score = local_score(X, old_X_parents)
                            self.score_cache.update({(X, tuple(sorted(old_X_parents))): old_X_score})  # cache score
                        if (Y, tuple(sorted(old_Y_parents))) in self.score_cache:
                            old_Y_score = self.score_cache.get((Y, tuple(sorted(old_Y_parents))))
                        else:
                            old_Y_score = local_score(Y, old_Y_parents)
                            self.score_cache.update({(Y, tuple(sorted(old_Y_parents))): old_Y_score})  # cache score
                        score_delta = new_X_score + new_Y_score - old_X_score - old_Y_score
                        yield (operation, score_delta)
                    pass
                pass
            pass
        pass
