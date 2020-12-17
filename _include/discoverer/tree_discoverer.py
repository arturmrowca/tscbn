from time import clock

import matplotlib.pyplot as plt
import networkx as nx

from _include.discoverer.prefix_structure_discoverer import PrefixStructureDiscoverer
from _include.m_utils.prefix_tree import create_prefix_tree
from _include.structure._tscbn_structure_model import TSCBNStructureModel
from general.log import Log as L
from network.tscbn import TSCBN


def create_tscbn(states_dict, nodes, edges, ran_gen = True):
    """
    Creates a TSCBN given a set of nodes and edges
    :param states_dict:
    :param nodes:
    :param edges:
    :param ran_gen:
    :return:
    """

    self = TSCBNStructureModel()

    # Define Vertices
    v, node_cpds = [], dict()

    # Initialize nodes
    print("Do")
    for node in nodes:
        tv = "_".join(node.split("_")[:-1])
        if tv[0]=="_":tv=tv[1:]
        v += self._dynamic_node(node, "disc", states_dict[tv], node_cpds)

    # Define Temporal Information
    dL_mean = 0
    dL_var = 0.1

    # Add default entries for temporal nodes
    skel, node_cpds = self._add_temporal_basics_dump(dL_mean, dL_var, node_cpds, v, edges)

    # Create Network
    tscbn = TSCBN("", skel, node_cpds, unempty=True, forbid_never=False, discrete_only=True, default_is_distributed=True, random_gen=ran_gen)  # Discrete case - later continuous nodes

    return tscbn

def structure_consistency(edges, nodes):
    """
    Ensures structural consistency within edges and nodes as defined by TSCBNs
    :param edges: discovered NW edges
    :param nodes: discovered NW nodes
    :return: cleaned and extended edges and nodes
    """
    tv_node_map = {}
    tv_node_max_idx = {}
    for node in nodes:
        x = node.split("_")
        tv_name = "_".join(x[:-1])
        next_idx = str(int(x[-1]) +1)
        next_node = "_".join([tv_name, next_idx])

        # A. Each node needs to have connection to its successive element
        if next_node in nodes:
            if not [node, next_node] in edges:
                edges += [[node, next_node]]

        if not tv_name in tv_node_map:
            tv_node_map[tv_name]  = []
            tv_node_max_idx[tv_name] = int(x[-1])
        tv_node_map[tv_name] += [node]
        if int(x[-1]) > tv_node_max_idx[tv_name]:
            tv_node_max_idx[tv_name] = int(x[-1])

    # B. Each node is not allowed to have a connection to an index higher than one away / per definition
    rem = []
    for edge in edges:
        st_node = edge[0]
        end_node = edge[1]

        x_start = st_node.split("_")
        x_end = end_node.split("_")

        tv_start = "_".join(x_start[:-1])
        start_next_idx = str(int(x_start[-1]) +1)
        tv_end = "_".join(x_end[:-1])
        next_node = "_".join([tv_start, start_next_idx])

        # only intra nodes
        if tv_start != tv_end: continue

        # violation
        if next_node != end_node:
            rem += [edge]

    for e in rem:
        edges.remove(e)

    return edges, nodes

class TreeDiscoverer(PrefixStructureDiscoverer):

    def __init__(self, draw=False, filtering=True, min_out_degree=0.25, k_infrequent=0.1,
                 max_time_difference=float('inf'), parallel=False, **kwargs):
        """
        :param draw: Set to True if graphs should be drawn after each step.
        :param filtering: Set to True if prefix tree and DAWG should be filtered.
        :param min_out_degree: A node in the prefix tree gets filtered when the sum of outgoing frequencies is smaller
            than min_out_degree times the incoming frequencies. This is necessary to assure a good transformation into
            the DAWG. Example: A --100--> B --100--> C --1--> D, D will be filtered.
        :param k_infrequent: An edge in the prefix tree gets filtered when it appeared less than k_infrequent times the
            sum of the incoming frequencies of the node.
        :param max_time_difference: Maximal time difference between nodes A --> B up to which node A is included in the
            POPS (potentially optimal parent sets) of node B.
        :param parallel: Set to True to compute the scores of the nodes in parallel using joblib.
        """
        super(TreeDiscoverer, self).__init__(**kwargs)
        self.draw = draw
        self.filtering = filtering
        self.min_out_degree = min_out_degree
        self.k_infrequent = k_infrequent
        self.max_time_difference = max_time_difference
        self.parallel = parallel
        self.parent_set_identification_time = 0.0
        self.structure_optimization_time = 0.0

    def discover_structure(self, sequences):
        """
        This method discovers the structure of a PGM. It returns a list of nodes and a list of edges.
        :param sequences: input sequences
        :return: list of nodes, list of edges
        """
        ping = clock()
        event_sequence_map, event_sequences, extendend_event_sequences = self.create_ordered_event_sequences(sequences)
        dawg, paths = self.create_dawg(event_sequences)
        unfiltered_pops, intra_edges = self.add_pops_and_occurrence_numbers(dawg)
        numbered_event_sequences, average_times = self.update_sequences(extendend_event_sequences, paths,
                                                                        event_sequence_map, dawg)
        pops = self.temporal_filtering_of_pops(unfiltered_pops, average_times)
        self.parent_set_identification_time = clock() - ping
        #L().log.debug('--------------------------------------------------------')
        #L().log.debug('Time for sequence handling and POPS determination: ' + str(self.parent_set_identification_time))
        #L().log.debug('--------------------------------------------------------')

        ping = clock()
        self.data = self.get_datastructure(numbered_event_sequences)
        nodes, inter_edges = self.discover_structure_from_pops(pops, self.data)
        edges = intra_edges + inter_edges
        self.structure_optimization_time = clock() - ping
        #L().log.debug('----------------------------------------------------------------------')
        #L().log.debug('Time for structure discovery using POPS: ' + str(self.structure_optimization_time))
        #L().log.debug('----------------------------------------------------------------------')

        return nodes, edges

    def create_dawg(self, sequences):
        """
        Takes the event sequences without values and creates the Directed Acyclic Word Graph (DAWG). Also known as
        Deterministic Acyclic Finite State Automaton (DAFSA) or Minimal Acyclic Deterministic Finite Automaton (MADFA).
        This is done by first creating the prefix tree, optionally filtering the prefix tree and then performing a
        minimization similar to a DFA (deterministic finite automaton) minimization (frequent subtree mining).
        :param sequences: ordered event sequences without values and timestamps
        :return dawg: DAWG created out of sequences by creating and filtering the prefix tree and afterwards doing
            frequent subtree mining.
        :return paths: List of all paths through the DAWG. Later in the algorithm, this list is used to correctly number
            the events.
        """

        def filter_prefix_tree(prefix_tree):
            """
            This method filters the prefix tree according to the settings of the user. Relevant variables are
            k_infrequent and min_out_degree (description in init method).
            :param prefix_tree: prefix tree
            :return: filtered prefix tree
            """
            #L().log.debug('---------------------------------------------------')
            #L().log.debug('-------- Prefix Tree Filtering --------------------')
            #L().log.debug('---------------------------------------------------')
            # check if all outgoing edges of a node have to be filtered (because it is the final node of many sequences)
            min_out_degree = self.min_out_degree
            k_infrequent = self.k_infrequent
            for node in nx.topological_sort(prefix_tree):
                signal = prefix_tree.nodes[node]['source']
                if signal == 'NIL':  # synthetic leaf node
                    continue
                in_edges = list(prefix_tree.in_edges(nbunch=node, data=True))
                out_edges = list(prefix_tree.out_edges(nbunch=node, data=True))
                frequency_in = sum(data['frequency'] for (_, _, data) in in_edges)
                frequency_out = sum(data['frequency'] for (_, _, data) in out_edges)
                # check if all out edges of a node have to be filtered (because it is the final node of many sequences)
                if 0 < frequency_out < min_out_degree * frequency_in:
                    #L().log.debug('remove all outgoing edges of node ' + node + ' (frequency_in: ' + str(frequency_in)
                    #              + ', frequency_out: ' + str(frequency_out) + ')')
                    # remove all descendants except NIL
                    prefix_tree.remove_edges_from([edge for edge in prefix_tree.out_edges(node) if edge[1] != 'NIL'])
                    continue
                pass
                # check if some single edges have to be filtered (because they are infrequent)
                if not signal:  # special case of root node
                    frequency_in = frequency_out
                for (u, v, data) in in_edges:  # filter incoming edges
                    if 0 < data['frequency'] < k_infrequent * frequency_out:
                        prefix_tree.remove_edge(u, v)
                        #L().log.debug('remove incoming edge ' + str((u, v)) + ' to node ' + node + ' (frequency: '
                        #              + str(data['frequency']) + ', total frequency out: ' + str(frequency_out) + ')')
                        pass
                    pass
                for (u, v, data) in out_edges:  # filter outgoing edges
                    if 0 < data['frequency'] < k_infrequent * frequency_in:
                        prefix_tree.remove_edge(u, v)
                        #L().log.debug('remove outgoing edge ' + str((u, v)) + ' from node ' + node + ' (frequency: '
                        #              + str(data['frequency']) + ', total frequency in: ' + str(frequency_out) + ')')
                        pass
                    pass
                pass
            assert nx.has_path(prefix_tree, root,
                               'NIL'), 'Too many edges filtered. No more path from start node to end node.'
            # remove nodes that are unreachable from root node or cannot reach the synthetic leaf node
            for node in list(prefix_tree.nodes):
                if not nx.has_path(prefix_tree, root, node) or not nx.has_path(prefix_tree, node, 'NIL'):
                    prefix_tree.remove_node(node)
                    #L().log.debug('remove unreachable node ' + node)
                pass
            return prefix_tree

        prefix_tree, root = create_prefix_tree(sequences)  # create prefix tree and store root node in variable

        if self.draw:
            plt.title('Unfiltered prefix_tree with frequencies')
            signal_labels = nx.get_node_attributes(prefix_tree, 'source')
            frequencies = nx.get_edge_attributes(prefix_tree, 'frequency')
            pos = nx.spring_layout(prefix_tree)
            nx.draw(prefix_tree, pos=pos, labels=signal_labels)
            nx.draw_networkx_edge_labels(prefix_tree, pos=pos, labels=frequencies)
            plt.show()
        pass

        if self.filtering:
            prefix_tree = filter_prefix_tree(prefix_tree)
        pass

        if self.draw:
            plt.title('Filtered prefix_tree with frequencies')
            signal_labels = nx.get_node_attributes(prefix_tree, 'source')
            frequencies = nx.get_edge_attributes(prefix_tree, 'frequency')
            pos = nx.spring_layout(prefix_tree)
            nx.draw(prefix_tree, pos=pos, labels=signal_labels)
            nx.draw_networkx_edge_labels(prefix_tree, pos=pos, labels=frequencies)
            plt.show()
        pass

        #L().log.debug('---------------------------------------------------')
        #L().log.debug('---------- Prefix Tree -> DAWG --------------------')
        #L().log.debug('---------------------------------------------------')
        prev_node = None  # remember previous node
        register = {}  # map subtree codes to nodes
        dawg = prefix_tree.copy()  # copy of prefix_tree will get modified during depth-first tree traversal
        for node in nx.dfs_postorder_nodes(prefix_tree, root):
            signal = prefix_tree.nodes[node]['source']  # get signal name that belongs to node
            if signal == 'NIL' or not signal:  # synthetic leaf node or root node
                continue
            # generate subtree code to detect similar subtrees
            if prefix_tree.has_edge(node, prev_node):
                subtree_code = signal + '*'
                for successor in [suc for suc in prefix_tree.successors(node) if suc != 'NIL']:
                    subtree_code += prefix_tree.nodes[successor]['subtree_code']
                subtree_code += '#'
            else:  # leaf node
                subtree_code = signal + '#'
            prefix_tree.nodes[node]['subtree_code'] = subtree_code
            #L().log.debug('subtree code of node ' + node + ' is ' + subtree_code)
            if subtree_code not in register:  # first occurrence of this subtree
                register.update({subtree_code: node})
            else:  # subtree already in prefix_tree, merge nodes with similar subtree
                representative = register.get(subtree_code)
                #L().log.debug('merge ' + node + ' and ' + representative + ' with subtree code ' + subtree_code)
                # store frequencies of outgoing edges of representative in DAWG in dict
                out_edge_frequencies = dict([(dawg.nodes[v]['source'], data['frequency']) for (_, v, data)
                                             in dawg.out_edges(nbunch=representative, data=True)])
                dawg = nx.contracted_nodes(dawg, representative, node)
                for (_, v, data) in list(dawg.out_edges(nbunch=representative, data=True)):
                    signal = dawg.nodes[v]['source']
                    if not signal or signal == 'NIL':
                        continue
                    dawg.out_edges[representative, v]['frequency'] = \
                        data['frequency'] + out_edge_frequencies[signal]  # sum up frequencies of merged nodes
            prev_node = node
        pass

        # create list of all paths through DAWG
        paths = nx.all_simple_paths(dawg, root, 'NIL')

        if self.draw:
            plt.title('DAWG with frequencies')
            signal_labels = nx.get_node_attributes(dawg, 'source')
            frequencies = nx.get_edge_attributes(dawg, 'frequency')
            pos = nx.spring_layout(dawg)
            nx.draw(dawg, pos=pos, labels=signal_labels)
            nx.draw_networkx_edge_labels(dawg, pos=pos, labels=frequencies)
            plt.show()
        pass

        return dawg, paths

    def add_pops_and_occurrence_numbers(self, dawg):
        """
        This method takes the DAWG created in the previous step and adds occurrence numbers and POPS (potentially
        optimal parent sets) to each node. Afterwards all nodes in the DAWG corresponding to the same signal and
        occurrence number are merged. This is done by relabeling the nodes to their complete_name. Networkx merges nodes
        automatically. Additionally, edges in strongly connected components are removed because otherwise the graph
        would contain circles.
        :param dawg: DAWG without occurrence numbers and POPS
        :return pops: Map from the nodes (V0_0, V1_0, ...) to their potential parents.
        :return intra_edges: All the edges between same signal that have to be part of the PGM in the end (e.g.
            V0_0 --> V0_1 or V3_24 --> V3_25).
        """
        intra_edges = set()
        # store largest occurrence numbers, this map will be needed in the second topological traversal
        largest_occurrence_numbers = {}

        # first topological traversal --> add occurrence numbers and create POPS
        for node in nx.topological_sort(dawg):
            signal = dawg.nodes[node]['source']
            if signal == 'NIL' or not signal:
                dawg.nodes[node]['complete_name'] = None
                dawg.nodes[node]['occurrence'] = -1
                continue
            pops = set()
            occurrence = 0
            for pred in dawg.predecessors(node):
                if not dawg.nodes[pred]['source']:
                    continue
                for pop in dawg.nodes[pred]['pops']:
                    pops.add(pop)
                    if dawg.nodes[pop]['source'] == signal and dawg.nodes[pop]['occurrence'] >= occurrence:
                        occurrence = dawg.nodes[pop]['occurrence'] + 1
                pops.add(pred)
                if dawg.nodes[pred]['source'] == signal and dawg.nodes[pred]['occurrence'] >= occurrence:
                    occurrence = dawg.nodes[pred]['occurrence'] + 1
            dawg.nodes[node]['occurrence'] = occurrence
            largest_occurrence_numbers.update({signal: occurrence})
            if occurrence >= 1:  # add intra edge if node is not an initial node
                intra_edges.add(tuple([signal + '_' + str(occurrence - 1), signal + '_' + str(occurrence)]))
            dawg.nodes[node]['complete_name'] = signal + '_' + str(occurrence)
            dawg.nodes[node]['pops'] = pops

        # second topological traversal (reversed) --> check if assignment of occurrence numbers is unique
        #L().log.debug('---------------------------------------------------')
        #L().log.debug('------- Occurrence Number Update ------------------')
        #L().log.debug('---------------------------------------------------')
        for node in reversed(list(nx.topological_sort(dawg))):
            signal = dawg.nodes[node]['source']
            occurrence = dawg.nodes[node]['occurrence']
            if signal == 'NIL' or not signal:
                continue
            descendants = set()  # store all descendants of node to check for minimal occurrence number
            for successor in dawg.successors(node):
                if dawg.nodes[successor]['source'] == 'NIL':
                    continue
                for descendant in dawg.nodes[successor]['descendants']:
                    descendants.add(descendant)
                descendants.add(successor)
            dawg.nodes[node]['descendants'] = descendants
            descendant_numbers = [dawg.nodes[descendant]['occurrence'] for descendant in descendants if
                                  dawg.nodes[descendant]['source'] == signal]
            if not descendant_numbers:  # no descendants of same signal -> all occurrence numbers up to largest possible
                largest_occurrence_number = largest_occurrence_numbers.get(signal)
                possible_occurrence_numbers = range(occurrence, largest_occurrence_number)
            else:
                min_descendant_number = min(descendant_numbers)
                possible_occurrence_numbers = range(occurrence, min_descendant_number)
            if len(possible_occurrence_numbers) <= 1:  # occurrence number is unique
                continue
            # choose occurrence_number that generates the largest number of SCCs
            # if equal, then choose the largest occurrence number to restrict the further numbering as less as possible
            optimal_num_sccs = 0
            optimal_occurrence_number = occurrence
            for occurrence_number in possible_occurrence_numbers:
                dawg.nodes[node]['complete_name'] = signal + '_' + str(occurrence_number)
                reduced_dawg = nx.relabel_nodes(dawg, nx.get_node_attributes(dawg, 'complete_name'))
                reduced_dawg.remove_node(None)  # remove synthetic nodes
                num_sccs = nx.number_strongly_connected_components(reduced_dawg)
                #L().log.debug('Node ' + node + ' (' + signal + ')' + ' with occurrence number ' + str(
                #    occurrence_number) + ' leads to ' + str(num_sccs) + ' strongly connected components.')
                if num_sccs >= optimal_num_sccs:
                    optimal_occurrence_number = occurrence_number
                    optimal_num_sccs = num_sccs
                pass
            dawg.nodes[node]['complete_name'] = signal + '_' + str(optimal_occurrence_number)
            #L().log.debug('Optimal occurrence number for node ' + node + ' is ' + str(optimal_occurrence_number))

        # filter POPS (initial nodes have no incoming edges, same signals have edges anyway)
        for node in [node for node in dawg.nodes if dawg.nodes[node]['occurrence'] == 0]:
            dawg.nodes[node]['pops'] = set()  # initial nodes are not allowed to have incoming edges
        for node in dawg.nodes:
            signal = dawg.nodes[node]['source']
            if signal == 'NIL' or not signal:
                continue
            prev_nodes = list(pred for pred in dawg.nodes[node]['pops'] if dawg.nodes[pred]['source'] == signal)
            if not prev_nodes:
                continue
            for prev_node in prev_nodes:
                dawg.nodes[node]['pops'].remove(prev_node)
            # POPS do not need to contain previous nodes of the same signal as there will be a edge anyway

        if self.draw:
            plt.title('DAWG with occurrence numbers')
            signal_occurrence_labels = {}
            for node in dawg.nodes:
                label = str(dawg.nodes[node]['complete_name'])
                signal_occurrence_labels.update({node: label})
            pos = nx.spring_layout(dawg)
            nx.draw(dawg, pos=pos, labels=signal_occurrence_labels)
            plt.show()
        pass

        # merge nodes corresponding to the same signal and occurrence number, update POPS during this process
        pops = {}  # dict with merged POPS sets
        for node in dawg.nodes:
            signal = dawg.nodes[node]['source']
            if signal == 'NIL' or not signal:
                continue
            if not dawg.nodes[node]['complete_name'] in pops:  # new dict entry
                pops.update({dawg.nodes[node]['complete_name']: dawg.nodes[node]['pops']})
            else:  # merge POPS sets
                pops_union = dawg.nodes[node]['pops'] | pops.get(dawg.nodes[node]['complete_name'])
                pops.update({dawg.nodes[node]['complete_name']: pops_union})
            pass
        node_name_map = nx.get_node_attributes(dawg, 'complete_name')
        dawg = nx.relabel_nodes(dawg, node_name_map)
        dawg.remove_node(None)  # remove synthetic nodes
        # update POPS attribute of nodes
        for node in dawg.nodes:
            relabeled_pops = set()
            for pop in pops.get(dawg.nodes[node]['complete_name']):
                relabeled_pops |= {node_name_map.get(pop)}
            dawg.nodes[node]['pops'] = relabeled_pops
        pass

        reduced_dawg = dawg.copy()
        for scc in nx.strongly_connected_components(dawg):
            if len(scc) > 1:
                #L().log.debug('remove edges in strongly connected component ' + str(scc))
                # remove all edges between nodes in the strongly connected component to avoid circles, except edges
                # between nodes of the same variable, e.g. V0_1 --> V0_2
                reduced_dawg.remove_edges_from(
                    edge for edge in dawg.edges(nbunch=scc) if
                    edge[1] in scc and edge[0].rsplit('_', 1)[0] != edge[1].rsplit('_', 1)[0])
                for node in scc:  # remove nodes from each others POPS
                    reduced_dawg.nodes[node]['pops'] -= scc
                pass
            pass
        pass
        # there should be no non-trivial strongly connected component anymore
        assert nx.number_strongly_connected_components(reduced_dawg) == len(reduced_dawg.nodes)

        if self.draw:
            plt.title('Reduced DAWG')
            signal_occurrence_labels = {}
            for node in reduced_dawg.nodes:
                label = str(reduced_dawg.nodes[node]['complete_name'])
                signal_occurrence_labels.update({node: label})
            pos = nx.spring_layout(reduced_dawg)
            nx.draw(reduced_dawg, pos=pos, labels=signal_occurrence_labels)
            plt.show()
        pass

        intra_edges = [list(edge) for edge in intra_edges]
        return nx.get_node_attributes(reduced_dawg, 'pops'), intra_edges

    def update_sequences(self, extended_event_sequences, paths, event_sequence_map, dawg):
        """
        Takes the list of event sequences with value and the mapping from event sequences with value to the event
        sequences and the mapping from the event sequences to the event sequences with occurrence numbers to create
        and return all event sequences with values and occurrence numbers as required in the further steps of the
        structure discovery process. Moreover, a map with the means of the occurrence times is created and returned.
        :param extended_event_sequences: event sequences with value and timestamp
        :param paths: all paths through the DAWG
        :param event_sequence_map: map from extended event sequences to event sequences
        :param dawg: DAWG created in the previous steps
        :return event_sequences: updated event sequences with occurrence numbers
        :return average_times: mean of the occurrence times of all the events (later used to filter the POPS)
        """
        #L().log.debug('---------------------------------------------------')
        #L().log.debug('----------- Sequence Filtering --------------------')
        #L().log.debug('---------------------------------------------------')
        # create mapping from event sequences without occurrence numbers to sequences with occurrence numbers
        numbered_event_sequence_map = {}
        for path in paths:
            path_without_occurrences = []
            path_with_occurrences = []
            for node in path:
                signal = dawg.nodes[node]['source']
                complete_name = dawg.nodes[node]['complete_name']
                if signal == 'NIL' or not signal:  # synthetic leaf node or root node
                    continue
                else:
                    path_without_occurrences.append(signal)
                    path_with_occurrences.append(complete_name)
                pass
            numbered_event_sequence_map.update({tuple(path_without_occurrences): path_with_occurrences})
        pass
        event_sequences = []
        times = {}
        for extended_event_sequence in extended_event_sequences:
            event_sequence = []
            es = event_sequence_map.get(tuple(extended_event_sequence))
            eswon = numbered_event_sequence_map.get(tuple(es))
            if not eswon:  # sequence was infrequent and is filtered
                #L().log.debug('event sequence ' + str(extended_event_sequence) + ' was infrequent')
                # Here one could think about using the infrequent sequences to calculate the scores at least partially.
                # The problem it is not possible to number the events in the sequences as there is no matching path in
                # the DAWG created in the earlier steps of the algorithm. Therefore, it is hard to assign the events to
                # the correct interval (signal + number).
                continue
            for event, event_won in zip(extended_event_sequence, eswon):
                event_sequence.append((event_won, event[1]))
                if event_won not in times:
                    times.update({event_won: [event[2]]})
                times.get(event_won).append(event[2])
            event_sequences.append(event_sequence)

        #L().log.debug('---------------------------------------------------')
        #L().log.debug('----------------- Times ---------------------------')
        #L().log.debug('---------------------------------------------------')
        average_times = {node: sum(time_list) / len(time_list) for node, time_list in times.items()}
        #for node, time in average_times.items():
        #    L().log.debug('Average time of node ' + node + ' is ' + str(time))
        return event_sequences, average_times

    def temporal_filtering_of_pops(self, pops, average_times):
        """
        This method removes variables out of the POPS if the average time distance is larger than the threshold set by
        the user.
        :param pops: map from all the nodes to their potential parents
        :param average_times: mean of the occurrence times of all the events
        :return: filtered POPS
        """
        #L().log.debug('---------------------------------------------------')
        #L().log.debug('----------- Potential parents ... -----------------')
        #L().log.debug('---------------------------------------------------')
        for node in pops:
            reduced_pops = set()
            for parent in pops.get(node):
                if average_times.get(node) - average_times.get(parent) < self.max_time_difference:
                    reduced_pops.add(parent)
                pass
            pops.update({node: reduced_pops})
            #L().log.debug('... of node ' + node + ' are ' + str(list(reduced_pops)))
        return pops

    def discover_structure_from_pops(self, pops, data):
        raise NotImplementedError("discover_structure_from_pops() not implemented in class %s" % str(__class__))
