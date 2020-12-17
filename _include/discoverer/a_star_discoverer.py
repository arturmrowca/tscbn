from _include.discoverer.naive_discoverer import NaiveDiscoverer
from _include.discoverer.optimal_ps_calculator import OptimalPSCalculator
from general.log import Log as L

class OrderGraphNode(object):
    def __init__(self, g, h, node_set, leaf, closed=False):
        self.g = g
        self.h = h
        self.score = g + h
        self.node_set = node_set
        self.leaf = leaf
        self.closed = closed

    def __str__(self):
        return ('Leaf: ' + self.leaf + '\tNodes: ' + str(self.node_set) + '\tScore (g): ' + str(
            self.g) + '\tScore (h): ' + str(self.h))


class AStarDiscoverer(NaiveDiscoverer):
    """
    Remark: This approach is still computationally infeasible for a large number of nodes.
    """

    def __init__(self, score='BIC', **kwargs):
        super(AStarDiscoverer, self).__init__(**kwargs)
        self.score = score

    def discover_structure_from_statistics(self, data, nodes):
        """
        Implements the approach presented in
        # Changhe Yuan, Brandon Malone, and Xiaojian Wu. "Learning Optimal Bayesian Networks Using A* Search".
        # In: Proceedings of the 22nd International Joint Conference on Artificial Intelligence (IJCAI-11).
        # Helsinki, Finland, 2011, pp. 2186-2191.
        :param nodes: all signal_occurrence values that are in the data set
        :param data: ADtree or pandas dataframe that contains the dataset counts
        :return: list of edges
        """

        def compute_score(optimal_ps_calculator, node, parent_nodes):
            #L().log.debug('############# compute score ##############')
            _, g = optimal_ps_calculator.get_optimal_parent_set(node, parent_nodes, 'parent_graph', return_score=True)
            return g

        def compute_heuristic(optimal_ps_calculator, node, parent_nodes, nodes):
            """
            :param optimal_ps_calculator: optimal parent set calculator
            :param node: current leaf node
            :param parent_nodes: potential parents of current leaf node
            :param nodes: list of all nodes
            :return: approximation of the remaining score
            """
            #L().log.debug('########### compute heuristic ############')
            h = 0
            remaining_nodes = [_ for _ in nodes if _ not in parent_nodes and _ != node]
            for remaining_node in remaining_nodes:  # all the remaining nodes can chose an optimal parent set
                potential_parents = [_ for _ in nodes if _ != remaining_node]  # graph may be cyclic
                _, score = optimal_ps_calculator.get_optimal_parent_set(remaining_node, potential_parents,
                                                                        'parent_graph', return_score=True)
                h += score
            return h

        optimal_ps_calculator = OptimalPSCalculator(score=self.score, data=data)
        generated_order_graph_nodes = {}  # remember already generated nodes and their order graph nodes
        open_list = []  # open list to store nodes that can be expanded further

        root = OrderGraphNode(0.0, 0.0, set(), '')  # create root node of order graph
        open_list.append(root)  # add root node to open list
        goal = None

        # start algorithm
        while open_list:
            order_graph_node = max(open_list, key=lambda _: _.score)  # pop node from open list
            open_list.remove(order_graph_node)
            #L().log.debug('---------------------------------------------------')
            #L().log.debug('Order graph node expanded: ' + str(order_graph_node))
            #L().log.debug('---------------------------------------------------')
            if len(order_graph_node.node_set) == len(nodes):  # check if node is goal node
                goal = order_graph_node
                break
            order_graph_node.closed = True
            for leaf in nodes:  # expand node
                if leaf in order_graph_node.node_set:  # make sure that this node was not already present
                    continue
                new_node_set = order_graph_node.node_set | {leaf}
                if tuple(
                        sorted(new_node_set)) not in generated_order_graph_nodes:  # check if node was already generated
                    # if not, then calculate g and h, create new node and store it in open list
                    g = order_graph_node.g + compute_score(optimal_ps_calculator, leaf, order_graph_node.node_set)
                    h = compute_heuristic(optimal_ps_calculator, leaf, order_graph_node.node_set, nodes)
                    succ_order_graph_node = OrderGraphNode(g, h, new_node_set, leaf)
                    open_list.append(succ_order_graph_node)
                    generated_order_graph_nodes.update({tuple(sorted(new_node_set)): succ_order_graph_node})
                    continue
                # node was already generated -> continue if it's in closed list, update if it's in open list
                succ_order_graph_node = generated_order_graph_nodes.get(tuple(sorted(new_node_set)))
                if succ_order_graph_node.closed:
                    continue
                g = order_graph_node.g + compute_score(optimal_ps_calculator, leaf, order_graph_node.node_set)
                if g < succ_order_graph_node.g:  # better path found
                    succ_order_graph_node.g = g
                    succ_order_graph_node.leaf = leaf
                pass
            pass
        pass

        # create edges
        edges = []
        optimal_path = []
        while len(optimal_path) < len(nodes):
            variable = goal.leaf
            potential_parents = goal.node_set - {variable}
            optimal_path = [goal.leaf] + optimal_path
            for parent in optimal_ps_calculator.get_optimal_parent_set(variable, potential_parents):
                edges.append([parent, variable])
            goal = generated_order_graph_nodes.get(tuple(sorted(potential_parents)))
        return edges
