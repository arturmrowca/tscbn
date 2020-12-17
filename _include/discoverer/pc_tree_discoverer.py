from itertools import combinations
from math import isnan

import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators.base import BaseEstimator

from _include.discoverer.tree_discoverer import TreeDiscoverer
from _include.m_utils.adtree import ADTree
from _include.m_utils.scores.GSquareEstimator import GSquareEstimator
from general.log import Log as L

class PCTreeDiscoverer(TreeDiscoverer):
    """
    Tree discoverer using PC algorithm.
    """

    def __init__(self, alpha=0.05, max_reach=2, chi_square_thresh= 1,optimization_chi_square=False, **kwargs):
        """
        :param alpha: significance level
        :param max_reach: maximal number of nodes in the condition set when doing the CI tests
        """
        super(PCTreeDiscoverer, self).__init__(**kwargs)
        self.alpha = alpha
        self.max_reach = max_reach
        self.optimization_chi_square = optimization_chi_square # if true use chi square for optimization
        self.chi_square_thresh = chi_square_thresh # threshold for chi square variant - bigger = more restrictive

    def discover_structure_from_pops(self, pops, data):
        """
        This method takes the potential parents of all nodes and the ADtree with all the data. An approach similar to
        PC algorithm is performed to determine the parent set for each node.
        :param pops: map from nodes to their potential parents
        :param data: ADtree or pandas dataframe
        :return nodes: list of nodes
        :return edges: list of inter edges
        """

        def create_maximal_pgm(pops):
            pgm = nx.DiGraph()
            pgm.add_nodes_from(pops)  # create nodes
            for node in pops:
                edges = [(parent, node) for parent in pops.get(node) if
                         node.rsplit('_', 1)[0] != parent.rsplit('_', 1)[0]]
                pgm.add_edges_from(edges)  # add edges
            return pgm

        def markov_blanket(graph, parent_node, node):
            mb = set(pa for pa in graph.predecessors(node))  # add parent nodes
            mb |= set(ch for ch in graph.successors(node))  # add child nodes
            for child in graph.successors(node):  # add parents of children
                mb |= set(pa for pa in graph.predecessors(child))
            if node in mb:  # remove node
                mb.remove(node)
            if parent_node in mb:  # remove parent_node
                mb.remove(parent_node)
            return mb

        max_pgm = create_maximal_pgm(pops)

        if self.draw:
            plt.title('Maximal PGM (only intra-edges)')
            signal_pos_map = {}
            pos = {}
            for node in max_pgm.nodes:
                if node.rsplit('_', 1)[0] not in signal_pos_map:
                    signal_pos_map.update({node.rsplit('_', 1)[0]: len(signal_pos_map)})
                x_coordinate = int(node[-1:])
                y_coordinate = signal_pos_map.get(node.rsplit('_', 1)[0])
                pos.update({node: [x_coordinate, y_coordinate]})
            nx.draw(max_pgm, pos=pos, with_labels=True)
            plt.show()
        pass

        if isinstance(data, ADTree):
            cb_estimator = GSquareEstimator(adtree=data)
        else:
            cb_estimator = BaseEstimator(data=data, complete_samples_only=False)
        # procedure similar to PC algorithm
        pgm = max_pgm.copy()
        condition_set_size = 0
        L().log.debug('---------------------------------------------------')
        L().log.debug('---- Conditional Independence Tests ---------------')
        L().log.debug('---------------------------------------------------')

        #
        if self.optimization_chi_square:
            import scipy.stats as scs
            def chi_square_of_df_cols(df, col1, col2):
                df_col1, df_col2 = df[col1], df[col2]
                categories_2 = list(df_col2.unique())
                categories_1 = list(df_col1.unique())
                result = [[sum((df_col1 == cat1) & (df_col2 == cat2))
                           for cat2 in categories_2]
                          for cat1 in categories_1]

                chi = scs.chi2_contingency(result)


                return chi

            remove_edges = []
            for (source, target) in pgm.edges():
                # check how correlated those two edges are / independent of MB and all the other stuff
                dat = chi_square_of_df_cols(self.data, source, target) # 1 = more corr. 0 = less corr.
                chi2, p, sufficient_data = dat[0], dat[1], dat[2]
                #print("%s  Chi = %s, p=%s" % (str([source, target]), str(chi2), str(p)))

                if chi2 < self.chi_square_thresh and pgm.has_edge(source, target):
                    L().log.debug('remove edge ' + str((source, target)))
                    remove_edges.append((source, target))
            pgm.remove_edges_from(remove_edges)
            #import sys
            #sys.exit(0)


            # additionally remove edges which are conditionally independent
            # e.g. given a-> b  c->b   and given a, c is independent of b, then I can remove c!!!
            remove_edges = []
            for (source, target) in pgm.edges():
                condition_set = [a for a in pgm.predecessors(target) if a != source]
                if not condition_set:continue
                _, p_val, _ = cb_estimator.test_conditional_independence(source, target, list(condition_set))
                if p_val > self.alpha:
                    if pgm.has_edge(source, target):
                        L().log.debug('remove edge ' + str((source, target)))
                        remove_edges.append((source, target))
            pgm.remove_edges_from(remove_edges)

        else:
            while True:
                cont = False
                remove_edges = []
                for (source, target) in pgm.edges():
                    mb = markov_blanket(pgm, target, source)
                    if len(mb) >= condition_set_size:
                        L().log.debug('testing ' + source + ' --> ' + target)
                        L().log.debug('markov blanket of ' + source + ' is ' + str(mb))
                        for condition_set in combinations(mb, condition_set_size):
                            L().log.debug(
                                'independence test of ' + source + ' and ' + target + ' with subset ' + str(condition_set))
                            _, p_val, _ = cb_estimator.test_conditional_independence(source, target, list(condition_set))
                            #if isnan(p_val):  # pgmpy CI test returns NaN instead of 1
                            #    p_val = 1
                            L().log.debug('p_val = ' + str(p_val))
                            if p_val > self.alpha:
                                if pgm.has_edge(source, target):
                                    L().log.debug('remove edge ' + str((source, target)))
                                    remove_edges.append((source, target))
                                break
                            pass
                        cont = True
                    pass
                condition_set_size += 1
                pgm.remove_edges_from(remove_edges)
                if cont is False:
                    break
                if condition_set_size > self.max_reach:
                    break

        if self.draw:
            plt.title('PGM after CI tests (only inter-edges)')
            signal_pos_map = {}
            pos = {}
            for node in pgm.nodes:
                if node.rsplit('_', 1)[0] not in signal_pos_map:
                    signal_pos_map.update({node.rsplit('_', 1)[0]: len(signal_pos_map)})
                x_coordinate = int(node[-1:])
                y_coordinate = signal_pos_map.get(node.rsplit('_', 1)[0])
                pos.update({node: [x_coordinate, y_coordinate]})
            nx.draw(pgm, pos=pos, with_labels=True)
            plt.show()
        pass

        nodes = list(pops.keys())
        edges = [list(edge) for edge in pgm.edges]
        return nodes, edges
