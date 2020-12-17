import multiprocessing

from joblib import Parallel, delayed

from _include.discoverer.optimal_ps_calculator import OptimalPSCalculator
from _include.discoverer.tree_discoverer import TreeDiscoverer
from general.log import Log as L

class SBTreeDiscoverer(TreeDiscoverer):
    """
    Score-based tree discoverer.
    """

    def __init__(self, score='BIC', approach='parent_graph', **kwargs):
        """
        :param score: Score that will be used to to rate networks. (Possible scores at the moment: BIC, AIC, K2, Bdeu)
        :param approach: Approach to determine the optimal parent sets. (Possible approaches: exhaustive, parent_graph)
        """
        super(SBTreeDiscoverer, self).__init__(**kwargs)
        self.score = score
        self.approach = approach

    def discover_structure_from_pops(self, pops, data):
        """
        This method takes the potential parents of all nodes and the ADtree with all the data. A score-based approach to
        determine the optimal parent set for each node is performed.
        :param pops: map from nodes to their potential parents
        :param data: ADtree or pandas dataframe
        :return nodes: list of nodes
        :return edges: list of inter-edges
        """
        nodes = list(pops.keys())
        optimal_ps_calculator = OptimalPSCalculator(score=self.score, data=data)
        # optimize each node independent from other nodes
        if self.parallel:
            num_cores = multiprocessing.cpu_count()
            L().log.debug('Parallel optimization of scoring method using ' + str(num_cores) + ' cores.')
            optimal_parent_sets = Parallel(n_jobs=num_cores, backend="threading")(
                delayed(optimal_ps_calculator.get_optimal_parent_set)(variable, pops.get(variable), self.approach) for
                variable in nodes)
        else:
            optimal_parent_sets = [
                optimal_ps_calculator.get_optimal_parent_set(variable, pops.get(variable), self.approach) for variable
                in nodes]
        edges = [[parent, node] for node, optimal_parent_set in zip(nodes, optimal_parent_sets) for parent in
                 optimal_parent_set]
        return nodes, edges
