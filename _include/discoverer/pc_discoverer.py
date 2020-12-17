from itertools import combinations
from itertools import permutations
from math import isnan

import networkx as nx
from pgmpy.estimators.base import BaseEstimator

from _include.discoverer.naive_discoverer import NaiveDiscoverer
from _include.m_utils.adtree import ADTree
from _include.m_utils.scores.GSquareEstimator import GSquareEstimator
from general.log import Log as L

class PCDiscoverer(NaiveDiscoverer):
    """
    Implements the PC algorithm. Original PC algorithm returns a PDAG (partially directed acyclic graph). Not oriented
    edges are oriented using the occurrence number, so e.g. V2_0 --- V2_1 will be oriented to V2_0 --> V2_1.
    Implementation mainly copied from package pcalg.
    """
    def __init__(self, alpha=0.05, max_reach=2, **kwargs):
        super(PCDiscoverer, self).__init__(**kwargs)
        self.alpha = alpha
        self.max_reach = max_reach

    def discover_structure_from_statistics(self, data, nodes):
        """
        Implements the PC algorithm.
        :param nodes: all signal_occurrence values that are in the data set
        :param data: ADtree or pandas dataframe that contains the dataset counts
        :return: list of edges
        """
        skeleton, sep_set = self.estimate_skeleton(data, nodes)
        pdag = self.estimate_cpdag(skeleton, sep_set)

        # orient remaining undirected edges according to occurrence number
        for scc in nx.strongly_connected_components(pdag):
            if len(scc) == 1:
                continue
            scc_nodes = sorted(scc, key=lambda node: int(node.rsplit('_')[-1]))
            for (parent, child) in combinations(scc_nodes, 2):
                if int(parent.rsplit('_')[-1]) <= int(child.rsplit('_')[-1]) and (child, parent) in pdag.edges:
                    pdag.remove_edge(child, parent)
                pass
            pass
        pass

        edges = [list(edge) for edge in pdag.edges]
        L().log.debug('Edges: ' + str(edges))
        return edges

    def estimate_skeleton(self, data, nodes):
        def create_max_skeleton(nodes):
            skeleton = nx.Graph()
            skeleton.add_nodes_from(nodes)  # create nodes
            edges = set()
            for node in nodes:
                for neigh in nodes:
                    if node != neigh:
                        edges.add((node, neigh))
                    pass
                pass
            skeleton.add_edges_from(edges)  # add edges
            return skeleton

        max_skeleton = create_max_skeleton(nodes)

        if isinstance(data, ADTree):
            cb_estimator = GSquareEstimator(adtree=data)
        else:
            cb_estimator = BaseEstimator(data=data, complete_samples_only=False)
        # procedure similar to PC algorithm
        skeleton = max_skeleton.copy()
        condition_set_size = 0
        sep_set = {}
        L().log.debug('---------------------------------------------------')
        L().log.debug('---- Conditional Independence Tests ---------------')
        L().log.debug('---------------------------------------------------')
        while True:
            cont = False
            remove_edges = []
            for (source, target) in permutations(nodes, 2):
                neighbors = list(skeleton.neighbors(source))
                if target not in neighbors:
                    continue
                else:
                    neighbors.remove(target)
                if len(neighbors) >= condition_set_size:
                    L().log.debug('testing ' + source + ' --> ' + target)
                    L().log.debug('neighbors of ' + source + ' are ' + str(neighbors))
                    for condition_set in combinations(neighbors, condition_set_size):
                        L().log.debug(
                            'independence test of ' + source + ' and ' + target + ' with subset ' + str(condition_set))
                        _, p_val, _ = cb_estimator.test_conditional_independence(source, target, list(condition_set))
                        if isnan(p_val):  # pgmpy CI test returns NaN instead of 1
                            p_val = 1
                        L().log.debug('p_val = ' + str(p_val))
                        if p_val > self.alpha:
                            if skeleton.has_edge(source, target):
                                L().log.debug('remove edge ' + str((source, target)))
                                remove_edges.append((source, target))
                            key = tuple(sorted((source, target)))
                            if key in sep_set:
                                sep_set[key] |= set(condition_set)
                            else:
                                sep_set[key] = set(condition_set)
                            break
                        pass
                    cont = True
                pass
            condition_set_size += 1
            skeleton.remove_edges_from(remove_edges)
            if cont is False:
                break
            if condition_set_size > self.max_reach:
                break
            pass
        return skeleton, sep_set

    def estimate_cpdag(self, skel_graph, sep_set):
        dag = skel_graph.to_directed()
        nodes = skel_graph.nodes()
        for (source, target) in combinations(nodes, 2):
            source_neighbors = set(dag.successors(source))
            if target in source_neighbors:
                continue
            target_neghbors = set(dag.successors(target))
            if source in target_neghbors:
                continue
            common_neighbors = source_neighbors & target_neghbors
            key = tuple(sorted((source, target)))
            for k in common_neighbors:
                if k not in sep_set[key]:
                    if dag.has_edge(k, source):
                        dag.remove_edge(k, source)
                        L().log.debug('S: remove edge (' + k + ', ' + source + ')')
                        pass
                    if dag.has_edge(k, target):
                        dag.remove_edge(k, target)
                        L().log.debug('S: remove edge (' + k + ', ' + target + ')')
                        pass
                    pass
                pass
            pass

        def _has_both_edges(dag, i, j):
            return dag.has_edge(i, j) and dag.has_edge(j, i)

        def _has_any_edge(dag, i, j):
            return dag.has_edge(i, j) or dag.has_edge(j, i)

        # For all the combination of nodes source and target, apply the following
        # rules.
        for (source, target) in combinations(nodes, 2):
            # Rule 1: Orient source-target into source->target whenever there is an arrow k->source
            # such that k and target are nonadjacent.
            #
            # Check if source-target.
            if _has_both_edges(dag, source, target):
                # Look all the predecessors of source.
                for k in dag.predecessors(source):
                    # Skip if there is an arrow source->k.
                    if dag.has_edge(source, k):
                        continue
                    # Skip if k and target are adjacent.
                    if _has_any_edge(dag, k, target):
                        continue
                    # Make source-target into source->target
                    dag.remove_edge(target, source)
                    L().log.debug('R1: remove edge (' + target + ', ' + source + ')')
                    break
                pass

            # Rule 2: Orient source-target into source->target whenever there is a chain
            # source->k->target.
            #
            # Check if source-target.
            if _has_both_edges(dag, source, target):
                # Find nodes k where k is source->k.
                succs_i = set()
                for k in dag.successors(source):
                    if not dag.has_edge(k, source):
                        succs_i.add(k)
                        pass
                    pass
                # Find nodes target where target is k->target.
                preds_j = set()
                for k in dag.predecessors(target):
                    if not dag.has_edge(target, k):
                        preds_j.add(k)
                        pass
                    pass
                # Check if there is any node k where source->k->target.
                if len(succs_i & preds_j) > 0:
                    # Make source-target into source->target
                    dag.remove_edge(target, source)
                    L().log.debug('R2: remove edge (' + target + ', ' + source + ')')
                    break
                pass

            # Rule 3: Orient source-target into source->target whenever there are two chains
            # source-k->target and source-l->target such that k and l are nonadjacent.
            #
            # Check if source-target.
            if _has_both_edges(dag, source, target):
                # Find nodes k where source-k.
                source_neighbors = set()
                for k in dag.successors(source):
                    if dag.has_edge(k, source):
                        source_neighbors.add(k)
                        pass
                    pass
                # For all the pairs of nodes in source_neighbors,
                for (k, l) in combinations(source_neighbors, 2):
                    # Skip if k and l are adjacent.
                    if _has_any_edge(dag, k, l):
                        continue
                    # Skip if not k->target.
                    if dag.has_edge(target, k) or (not dag.has_edge(k, target)):
                        continue
                    # Skip if not l->target.
                    if dag.has_edge(target, l) or (not dag.has_edge(l, target)):
                        continue
                    # Make source-target into source->target.
                    dag.remove_edge(target, source)
                    L().log.debug('R3: remove edge (' + target + ', ' + source + ')')
                    break
                pass

        return dag
