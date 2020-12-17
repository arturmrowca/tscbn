from collections import defaultdict

import networkx as nx
from networkx.utils import generate_unique_node

NIL = 'NIL'


def create_prefix_tree(paths):
    '''
    Creates the prefix tree (also known as trie) out of the paths. Implementation copied from package networkx v2.1.
    Implementation is extended by edge weights that correspond to the frequency of the transition in the input paths.
    '''
    def _helper(paths, root, B):
        children = defaultdict(list)
        for path in paths:
            if not path:
                B.add_edge(root, NIL, frequency=0)
                continue
            child, *rest = path
            children[child].append(rest)
        for head, tails in children.items():
            new_head = generate_unique_node()
            B.add_node(new_head, source=head)
            B.add_edge(root, new_head, frequency=len(tails))
            _helper(tails, new_head, B)

    T = nx.DiGraph()
    root = generate_unique_node()
    T.add_node(root, source=None)
    T.add_node(NIL, source=NIL)
    _helper(paths, root, T)
    return T, root