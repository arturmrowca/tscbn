from pyBN import read_bn, gs, hc, pc, replace_strings
import numpy as np
from _include.discoverer.prefix_structure_discoverer import PrefixStructureDiscoverer
import pandas as pd
import random


class PyBNStructureDiscoverer(PrefixStructureDiscoverer):

    def __init__(self, approach, avg_interval_number, arguments):
        super(PyBNStructureDiscoverer, self).__init__()
        self._app = approach
        self._arg = arguments
        self._avg_intervals_per_tv = avg_interval_number

    def _create_trivial_sequence(self, sequences):
        '''
        Todo: Sequences need to have same RVs - use distribution for this
            - if too short -> extend and fill with random last state
            - if too long -> cut
        Note: so wird dann auch die Zeit gar nicht einbezogen - könnte man noch trivial einbauen
        '''

        abs_res = []
        for sequence in sequences:
            res = {}

            for signal in sequence:
                target_number = int(np.round(self._avg_intervals_per_tv[signal])) # target number of elements

                # fill random with last state
                if len(sequence[signal]) < target_number:
                    # determine which positions to fill randomly
                    to_dist = target_number - len(sequence[signal])
                    pos_candidate = list(range(target_number))[1:] # 0 and -1 does not make sense to replace those are fixed

                    if len(pos_candidate) == to_dist:
                        # dann is es klar -> n kandidaten 2 positionen
                        pos = pos_candidate
                    else:
                        # choose positions to use
                        random.shuffle(pos_candidate)
                        pos = pos_candidate[:to_dist]
                    pos= sorted(pos)

                    # create event list [1, 5, 7]
                    # insert at pos whats missing e.g. [1, 2, 8]
                    # start at 1 -> insert -> then shift index +1
                    shift_idx = 0
                    i, k = 0, 0
                    last_val = None
                    occ_cnt = -1
                    for i in range(target_number):

                        # insert dummy from last
                        if i in pos:
                            cur_val = last_val
                        else:
                            cur_val = sequence[signal][k]
                            k += 1

                        occ_cnt += 1
                        res[signal+"_"+str(occ_cnt)] = cur_val[0]
                        last_val = cur_val


                if len(sequence[signal]) >= target_number:

                    # cut to target if too long
                    if len(sequence[signal]) > target_number:
                        sequence[signal] = sequence[signal][:target_number]

                    occ_cnt = -1
                    for event in sequence[signal]:

                        occ_cnt += 1
                        res[signal+"_"+str(occ_cnt)] = event[0]
            abs_res += [res]
        return abs_res

    def discover_structure(self, sequences):

        # Problem: RV Sequenzen haben alle verschiedene Längen mal A_0, A_1 mal A_0, A_1, A_2 etc
        # Lösung: nehme average Länge gerundet pro TV - dann kürze oder extend randomly um volle Seq. gleicher länger zu kriegen
        sequences_prep = self._create_trivial_sequence(sequences)
        df= pd.DataFrame(sequences_prep)#.replace(np.nan, '8888', regex=True) # note hier müsste eher ein random wert rein
        df = df#[df.columns[:8]];print("Watch out - only first RVs!!!!!!!!!!!!!!!!!!") # restrict to first RVs only
        data = df.as_matrix()
        data, value_dict = replace_strings(data, return_values=True)
        data = data.astype(int)

        # Start discovery algorithm
        # Logik: spalte= RV; Zeile ist Wert // e.g. bn = gs(np.array([["b2", "a1"],["b", "a"],["b2", "a1"],["b", "a"],["b", "a"],["b", "a"]]))
        print("Starting Discovery")
        bn = gs(data)
        print("Done")
        print(bn.E)
        print(bn.V)

        # map back from int to string
        map = {}
        nodes = []
        for i in bn.E:
            map[i] = df.columns[i]
            nodes += [df.columns[i]]

        print(map)
        edges = []
        for i in bn.E:
            for j in bn.E[i]:
                print(str(j))
                target = map[j]
                # forbid _0 to be target
                if not target.split("_")[-1] == "0":
                    edges += [[map[i], target]]

        print(edges)
        print(nodes)
        return nodes, edges



