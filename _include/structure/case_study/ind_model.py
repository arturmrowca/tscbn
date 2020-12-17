#!/usr/bin/env python 
# -*- coding: utf-8 -*-
from _include.m_libpgm.graphskeleton import GraphSkeleton
import copy
import pandas as pd
from _include.structure.base_structure_model import BaseStructureModel
from network.tscbn import TSCBN


class IndModel(BaseStructureModel):
    '''
    This generator creates a Temporal state change BN from the given specification
    '''
    def __init__(self):
        super(IndModel, self).__init__()

    def generate_model(self, structure_specification = {}):
        '''
        Using the structure extracted with the naive SPM approach and expert knowledge a TSCBN is extracted
        The data to learn this TSCBN is also preprocessed manually to fit the definition

        :param structure_specification:
        :return:
        '''

        # Define Vertices
        v, node_cpds = [], dict()
        defaults = {"S-E_0": "s", "S-D_0" : "p", "S-C_0": "m", "S-B_0": "j", "S-A_0": "a"}

        # Initial nodes
        v += self._dynamic_node("S-A_0", "disc", ["e", "c", "f", "d", "a"], node_cpds)  # NEVER STATE IST SINNLOS
        v += self._dynamic_node("S-E_0", "disc", ["s", "t", "r"],
                                node_cpds)  # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("S-D_0", "disc", ["p", "q"], node_cpds)
        v += self._dynamic_node("S-C_0", "disc", ["m", "n", "o"], node_cpds)
        v += self._dynamic_node("S-B_0", "disc", ["j", "i", "k", "u", "l", "v"], node_cpds)

        # More nodes
        v += self._dynamic_node("S-A_1", "disc",
                                ["e", "c", "f", "d", "a"],
                                node_cpds)  # NEVER STATE IST SINNLOS
        v += self._dynamic_node("S-E_1", "disc", ["s", "t", "r"],
                                node_cpds)  # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("S-D_1", "disc", ["p", "q"], node_cpds)
        v += self._dynamic_node("S-C_1", "disc", ["m", "n", "o"], node_cpds)
        v += self._dynamic_node("S-B_1", "disc", ["j", "i", "k", "u", "l", "v"], node_cpds)

        v += self._dynamic_node("S-A_2", "disc",
                                ["e", "c", "f", "d", "a"],
                                node_cpds)  # NEVER STATE IST SINNLOS
        v += self._dynamic_node("S-E_2", "disc", ["s", "t", "r"],
                                node_cpds)  # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("S-C_2", "disc", ["m", "n", "o"], node_cpds)
        v += self._dynamic_node("S-B_2", "disc", ["j", "i", "k", "u", "l", "v"], node_cpds)

        v += self._dynamic_node("S-C_3", "disc", ["m", "n", "o"], node_cpds)

        # Define Edges
        e = []
        e += self._self_dependencies(v, ["S-A", "S-E", "S-D", "S-C", "S-B"])
        e += [["S-A_0", "S-D_1"]]  # FL Assistant muss an sein
        e += [["S-A_0", "S-B_1"]]  # If reason
        e += [["S-A_1", "S-D_1"]]  # If reason
        e += [["S-A_2", "S-B_2"]]  # If reason
        e += [["S-A_2", "S-E_2"]]  # If reason

        e += [["S-D_1", "S-C_2"]]  # If reason

        e += [["S-B_1", "S-C_1"]]  # If reason

        e += [["S-C_1", "S-E_1"]]  # If reason


        # ------------------------------------------------------------------
        #                Define Temporal Information - Manually
        # ------------------------------------------------------------------
        dL_mean = 3
        dL_var = 1
        skel, node_cpds = self._add_temporal_basics(dL_mean, dL_var, node_cpds, v, e, defaults)

        # ------------------------------------------------------------------
        #                Create Network
        # ------------------------------------------------------------------
        tscbn = TSCBN("", skel, node_cpds, unempty=True, forbid_never=False, discrete_only=True,
                      default_states=defaults, default_is_distributed=True)  # Discrete case - later continuous nodes


        return tscbn

    def map(self, signal_name):
        return signal_name

    def map_value(self, val):
        return val

    def get_raw_sequences(self, data_path):
        ind_df_done = pd.read_csv(data_path, sep = ";")
        ind_df_done = ind_df_done.sort_values("timestamp")

        last = {}
        result_sequences = []

        gathering = False
        timeout = 0
        done = False
        lastt = False

        for i in range(len(ind_df_done)):
            time = ind_df_done.iloc[i]["timestamp"]
            signal = ind_df_done.iloc[i]["Signalname"]
            value = ind_df_done.iloc[i]["interpreted_value"]

            if signal == "S-A" and value != "a":
                done = False
                if gathering:
                    # stop current gathering and get new
                    result_sequences += [cur_sequence]

                gathering = True
                cur_sequence = []

                # add all known last values as initial states
                for k in last:
                    if k != "S-A":
                        cur_sequence += [last[k]]

            if gathering and signal == "S-A" and value == "a":
                gathering = False
                timeout = time + 3000000000

            # store all incoming
            if gathering or time < timeout:
                cur_sequence += [[value, signal, time, False]]
                lastt = False
            else:
                if 0 != timeout:
                    if not done:
                        lastt = True

            if lastt == True:
                done = True
                if not cur_sequence in result_sequences:
                    result_sequences += [cur_sequence]

            # store last state
            last[signal] = [value, signal, time, True]
        return result_sequences

    def extract_signals(self, input_path, translate_signals=None, translate_states = None):
        raw_seqs = self.get_raw_sequences(input_path)

        # translate this to intervals for the model
        all_sequences = []
        for sequence in raw_seqs:
            #print(sequence)
            seq_count = {}
            first = True
            required = ["S-E", "S-D", "S-C", "S-B", "S-A"]
            defaults = {"S-E": "s", "S-D": "p", "S-C": "m", "S-B": "j",
                        "S-A": "a"}
            unprocessed = []
            last_entry = {}
            result_dict = {}
            t_max = 0

            for element in sequence:
                # VALUE AND NAME
                signal_name = self.map(element[1])
                value = self.map_value(element[0])
                unprocessed.append((signal_name, value))

                # TIME
                if not signal_name in seq_count: seq_count[signal_name] = -1
                seq_count[signal_name] += 1
                if element[3]:
                    timestamp = 0.0
                else:
                    if first:
                        t_0 = element[2]
                        first = False
                        timestamp = 0.0
                    else:
                        timestamp = element[2] - t_0

                # if last value exist append to result set
                if signal_name in last_entry:
                    if not signal_name in result_dict: result_dict[signal_name] = []
                    result_dict[signal_name].append([last_entry[signal_name][1], last_entry[signal_name][0], timestamp])
                    unprocessed.remove((signal_name, last_entry[signal_name][1]))
                    try:
                        required.remove(signal_name)
                    except:
                        pass

                # Store last entry
                last_entry[signal_name] = [timestamp, value]
                if timestamp > t_max: t_max= timestamp

            t_end = t_max + 0.1 * t_max
            for p in unprocessed:
                sig_name = p[0]
                value = p[1]
                if not sig_name in result_dict: result_dict[sig_name] = []
                result_dict[sig_name].append([value, last_entry[sig_name][0], t_end])

                try:
                    required.remove(sig_name)
                except:
                    pass

            for r in required:
                if not r in result_dict: result_dict[r] = []
                result_dict[r].append([defaults[r], 0, t_end])

            all_sequences.append(result_dict)


        for s in all_sequences:
            # translate if given
            if translate_signals is not None:
                k = list(s.keys())
                for tv in k:
                    s[translate_signals[tv]] = s.pop(tv)
            if translate_states is not None:
                for tv in s:
                    for st in s[tv]:
                        st[0] = translate_states[st[0]]
            print(str(s))

        return all_sequences

    def _print_tree_info(self, nodes):
        print("\n ------------------ \n Tree Information \n ------------------ ")
        a_ll = copy.deepcopy(list(nodes.keys()))

        for obj_id in range(len(a_ll)):
            fst = True
            for state_id in range(len(a_ll)):
                try:
                    ver = "V%s_%s" % (str(obj_id), str(state_id))
                    gap_ver = "dL_%s" % ver
                    nodes[ver]
                    if fst:
                        print("\nCurrent Object: %s" % str(obj_id))
                        fst = False
                    print(ver +" Parents: "+str(nodes[ver].Vdataentry["parents"]))
                    print(gap_ver + " Parents: " + str(nodes[gap_ver].Vdataentry["parents"]))
                    print("Mean dL: %s" % str(nodes[gap_ver].Vdataentry["mean_base"]))
                    print("\n")
                except:
                    break

    def _add_temporal_basics(self, dL_mean, dL_var, node_cpds, v, e, defaults):
        # 1. tp and dL information needs to be given - is distributed in any case
        #    currently same for all nodes - in reality distinct
        for vert in v:
            if vert in defaults.keys():
                node_cpds[vert]["dL_mean"] = 0
                node_cpds[vert]["dL_var"] = 0
            else:
                node_cpds[vert]["dL_mean"] = dL_mean
                node_cpds[vert]["dL_var"] = dL_var

        # Generate Network from this information
        skel = GraphSkeleton()
        skel.V = v
        skel.E = e
        skel.toporder()
        return skel, node_cpds
