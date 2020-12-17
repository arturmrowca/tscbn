#!/usr/bin/env python 
# -*- coding: utf-8 -*-
from network.tscbn import TSCBN
from _include.m_libpgm.graphskeleton import GraphSkeleton
import copy
import pandas as pd
from _include.structure.base_structure_model import BaseStructureModel

class HBModel(BaseStructureModel):
    '''
    This generator creates a Temporal state change BN from the given specification
    '''
    def __init__(self):
        super(HBModel, self).__init__()

    def map(self, signal_name):
        return signal_name

    def map_value(self, val):
        return val

    def generate_model(self, structure_specification = {}):
        '''
        Using the structure extracted with the naive SPM approach and expert knowledge a TSCBN is extracted
        The data to learn this TSCBN is also preprocessed manually to fit the definition

        :param structure_specification:
        :return:
        '''

        # Define Vertices
        v, node_cpds = [], dict()
        defaults = {"S-D_0": "d", "S-E_0" : "d", "S-B_0": "d", "S-A_0": "d", "S-C_0": "c"}

        # Initial nodes
        v += self._dynamic_node("S-D_0", "disc", ["b", "d"], node_cpds)  # NEVER STATE IST SINNLOS
        v += self._dynamic_node("S-E_0", "disc", ["b", "d"],
                                node_cpds)  # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("S-A_0", "disc", ["c", "a"], node_cpds)
        v += self._dynamic_node("S-B_0", "disc", ["b", "d"], node_cpds)
        v += self._dynamic_node("S-C_0", "disc", ["e", "f"], node_cpds)

        # More nodes
        v += self._dynamic_node("S-D_1", "disc", ["b", "d"], node_cpds)  # NEVER STATE IST SINNLOS
        v += self._dynamic_node("S-E_1", "disc", ["b", "d"],
                                node_cpds)  # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        #v += self._dynamic_node("S-A_1", "disc", ["NO_PRESS", "PRESS"], node_cpds)
        v += self._dynamic_node("S-B_1", "disc", ["b", "d"], node_cpds)
        #v += self._dynamic_node("S-C_1", "disc", ["e", "f"], node_cpds)

        # Define Edges
        e = []
        e += self._self_dependencies(v, ["S-D", "S-E", "S-B", "S-A", "S-C"])
        e += [["S-A_0", "S-B_1"]]  # FL Assistant muss an sein
        e += [["S-B_1", "S-D_1"]]  # If reason
        e += [["S-C_0", "S-D_1"]]  # If reason
        e += [["S-D_1", "S-E_1"]]  # If reason

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

    def extract_signals(self, data_path, translate_signals=None, translate_states = None):
        global PREV_SEQUENCE, PREV_TV_STATE

        PREV_SEQUENCE = {}
        PREV_TV_STATE = dict()
        df = pd.read_csv(data_path, sep = ";")

        required_signals = ["S-A", "S-B", "S-C", "S-D", "S-E"]
        defaults = {"S-A":"c", "S-B":"d", "S-C":"f", "S-D":"d", "S-E":"d"}
        df = df.sort_values("timestamp")
        sequences = df.groupby("sequence_id").apply(self._to_interval, df, required_signals, defaults)

        sequences = [eval(i) for i in sequences["sequence"].tolist()]
        for s in sequences:
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

        return sequences

    def _save_append(self, lst, entry, tv):

        shifto = {}
        shifto["S-A"] = 0.0001
        shifto["S-B"] = 0.0002
        shifto["S-C"] = 0.0000
        shifto["S-D"] = 0.0003
        shifto["S-E"] = 0.0004


        if len(lst) == 0:
            lst.append(entry)
            return lst
        else:
            if lst[-1][0] == entry[0]:
                lst[-1][-1] = entry[-1]
            else:
                lst.append(entry)

        if lst[-1][1] != 0:
            lst[-1][1] += shifto[tv]# SHIFT TIME TO AVOID RANDOMNESS
        return lst

    def _to_interval(self, df, whole_df, required_sig, defaults):

        global PREV_TV_STATE

        # 1. Normalize time to start at 0
        t_min = df.timestamp.min()
        t_max = df["timestamp"].max() + 0.2 * df["timestamp"].max() - t_min
        df = df.sort_values("timestamp")
        df.timestamp = df.timestamp - t_min # normalize

        # 2. Create dictionary per TV
        result = dict()
        for tv in required_sig:
            result[tv] = []
            tv_df = df[df["Signalname"] == tv]

            # 3. If Entry exists for TV
            if len(tv_df) > 0:
                for i in range(len(tv_df)):
                    row = tv_df.iloc[i]
                    if i < len(tv_df)-1:
                        next_row = tv_df.iloc[i+1]
                    else: next_row = None

                    # set values
                    t_start = float(row["timestamp"])
                    if next_row is None: t_end = t_max
                    else: t_end = float(next_row["timestamp"])

                    if t_start != 0:
                        if tv in PREV_TV_STATE:
                            defaults[tv] = PREV_TV_STATE[tv] # use last valid state from real data
                        if len(result[tv])==0:
                            result[tv] = self._save_append(result[tv], [defaults[tv], 0.0, t_start], tv)
                            #result[tv].append([defaults[tv], 0.0, t_start])
                    result[tv] = self._save_append(result[tv], [row.interpreted_value, t_start, t_end], tv)
            else:
                if tv in PREV_TV_STATE:
                    defaults[tv] = PREV_TV_STATE[tv]# use last valid state from real data
                result[tv] = self._save_append(result[tv], [defaults[tv], 0.0, t_max], tv)
            PREV_TV_STATE[tv] = result[tv][-1][0]

        df["sequence"] = str(result)

        return df.iloc[0]

    def _to_interval1(self, df, required_sig, defaults):
        '''
        Expected output
        list of [{'V0': [['o0_1', 0.0, 3.8012818666739947], ['o0_0', 3.8012818666739947, 8.348323044825703], ['o0_2', 8.348323044825703, 16.890927297752704]],
        'V1': [['o1_2', 0.0, 6.241821034812132], ['o1_1', 6.241821034812132, 16.890927297752704]],
        'V2': [['o2_1', 0.0, 3.7959815266200594], ['o2_0', 3.7959815266200594, 5.275699969803371], ['o2_2', 5.275699969803371, 6.839187865254665], ['o2_1', 6.839187865254665, 16.890927297752704]]}
        '''
        global PREV_SEQUENCE
        required_signals = copy.deepcopy(required_sig)

        df = df.apply(self._map_signalname, axis = 1)
        df = df.sort_values("timestamp")
        t_0 = df["timestamp"].min()
        t_end = df["timestamp"].max() + 0.2 * df["timestamp"].max() - t_0

        last_time, last_value, signal_count, result_sequences, last_index = {}, {}, {}, {}, {} # key: signalname value = count
        unprocessed = []

        last_time["S-D"] = 0
        try:
            last_value["S-D"] = PREV_SEQUENCE["S-D"]
        except:
            last_value["S-D"] = defaults["S-D"]
        last_index["S-D"] = 0
        signal_count["S-D"] = 0

        for line in range(len(df)):
            timestamp = df.iloc[line]["timestamp"] - t_0
            value = df.iloc[line]["interpreted_value"]

            # Signalname
            signal_raw = df.iloc[line]["Signalname"]
            if not signal_raw in signal_count: signal_count[signal_raw] = -1
            signal_count[signal_raw] += 1

            signal = signal_raw + "_" + str(signal_count[signal_raw])

            unprocessed += [signal]

            # create interval
            if signal_raw in last_time:
                start_interval = last_time[signal_raw]
                if signal == "S-C_1": start_interval = 0
                end_interval = timestamp
                if not signal_raw in result_sequences: result_sequences[signal_raw] = []
                result_sequences[signal_raw] += [[self.map_value(last_value[signal_raw]), start_interval, end_interval]]
                try:
                    unprocessed.remove(signal_raw + "_" + str(last_index[signal_raw]))
                except:
                    pass
                    #print("ignore")

                if signal_raw in required_signals:
                    required_signals.remove(signal_raw)


            last_time[signal_raw] = timestamp
            last_value[signal_raw] = value
            last_index[signal_raw] = signal_count[signal_raw]

            PREV_SEQUENCE[signal_raw] = value


        for sig in unprocessed:
            sigraw = "_".join(sig.split("_")[:-1])
            start_interval = last_time[sigraw]
            end_interval = t_end
            if not sigraw in result_sequences: result_sequences[sigraw] = []
            result_sequences[sigraw] += [[self.map_value(last_value[sigraw]), start_interval, end_interval]]

        for sigraw in required_signals:
            if not sigraw in result_sequences: result_sequences[sigraw] = []
            if not sigraw in PREV_SEQUENCE:
                PREV_SEQUENCE[sigraw] = defaults[sigraw]
            result_sequences[sigraw] += [[self.map_value(PREV_SEQUENCE[sigraw]), 0.0, t_end]]

        df["sequence"] = str(result_sequences)

        return df.iloc[0]

    def _map_signalname(self, df):
        df["Signalname"] = self.map(df["Signalname"])
        return df

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
