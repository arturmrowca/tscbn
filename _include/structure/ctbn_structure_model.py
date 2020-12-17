#!/usr/bin/env python 
# -*- coding: utf-8 -*-
from _include.m_libpgm.graphskeleton import GraphSkeleton
from _include.m_libpgm.nodedata import NodeData
import itertools
import numpy as np
import pandas as pd
from network.dbndiscrete import DBNDiscrete
from _include.structure.base_structure_model import BaseStructureModel
import os

class CTBNStructureModel(BaseStructureModel):
    '''
    This generator creates a Dynamic BN from the given specification
    Zustände als Knoten
    '''

    def __init__(self):
        super(CTBNStructureModel, self).__init__()

        self.EXPLICIT_DISABLING = False # For evaluation only to save time during generation - avoid CPD loop explosion
        self.eval_cpd_entries = 0 # For evaluation

    def get_cpd_raw(self, nodes, edges, node_data, COUNT_ONLY = True, val_map = dict()):
        self.eval_cpd_entries = 0
        # extract parents and children
        parents = dict() # key = node, value = parent list
        children = dict() # key = node, value = children list
        for node in nodes:
            if not node in children:children[node] = []
            if not node in parents:parents[node] = []
        for ed in edges:
            parents[ed[1]] += [ed[0]]
            children[ed[0]] += [ed[1]]
        for node in nodes:
            node_data[node]["parents"] = parents[node]
            node_data[node]["children"] = children[node]

        # per node extract CPDs
        transition_params = 0
        intensity_params = 0 # cprob fällt
        ctbn_dyn_str, ctbn_bncpts = "", ""
        #COUNT_ONLY = True # save computational cost by calculation
        for name in node_data:
            # add me pointing to myself
            '''vals = val_map[name].values()
            length = len(node_data[name]["vals"])
            ctbn_bncpts += "\n" + name + "$" + name
            ctbn_bncpts += "\nVec," + ",".join([str(v) for v in vals])
            ctbn_bncpts += '\n"0"' + ',' + ','.join(['0.0'] * length)

            ctbn_dyn_str += "\n" + name + "$" + name
            ctbn_dyn_str += "\nMat," + ",".join([str(v) for v in vals])
            for v in vals:
                ctbn_dyn_str += '\n"%s",' % str(v) + ",".join(['0.0'] * length)'''

            if node_data[name]["parents"]:
                node_data[name]["transition"] = {}
                node_data[name]["intensity"] = {}

                # Articifial execution -> only for counting -> yields same result as real
                if COUNT_ONLY:
                    # parent outcome combinations
                    parent_outcome_combos = 1
                    for p in node_data[name]["parents"]:
                        parent_outcome_combos *= len(node_data[p]["vals"])

                    transition_params += parent_outcome_combos * len(node_data[name]["vals"])**2
                    intensity_params += parent_outcome_combos * len(node_data[name]["vals"])

                # Real execution -> vs. for parameter count this can be computed
                else:
                    keys = []
                    if val_map:#map to numeric given map (as CTBN RLE only accepts numbers)
                        [keys.append([p + "=" + str(val_map[p][o]) for o in node_data[p]["vals"]]) for p in node_data[name]["parents"]]
                    else:
                        for p in node_data[name]["parents"]:keys.append(node_data[p]["vals"])

                    #ctbn_bncpts = ""
                    #ctbn_dyn_str = ""
                    vals = val_map[name].values()
                    length = len(node_data[name]["vals"])
                    for element in itertools.product(*keys):

                        # Intensity matrix given condition
                        node_data[name]["intensity"][str(list(element))] = np.zeros((len(node_data[name]["vals"]), len(node_data[name]["vals"])))
                        transition_params += len(node_data[name]["vals"])**2
                        ctbn_dyn_str += "\n" + name + "$" + ",".join(element)
                        ctbn_dyn_str += "\nMat," + ",".join([str(v) for v in vals])
                        for v in vals:
                            ctbn_dyn_str += '\n"%s",' % str(v) + ",".join(['0.0'] * length)

                        # Transition matrix given condition
                        z = np.zeros(length)
                        node_data[name]["transition"][str(list(element))] = z
                        intensity_params += length
                        ctbn_bncpts += "\n" + name + "$" + ",".join(element)
                        ctbn_bncpts += "\nVec," + ",".join([str(v) for v in vals])
                        ctbn_bncpts += '\n"0"' + ',' + ','.join(['0.0'] * length)


            else:
                node_data[name]["transition"] = np.zeros((len(node_data[name]["vals"]), len(node_data[name]["vals"])))
                transition_params += len(node_data[name]["vals"])**2
                node_data[name]["intensity"] = np.zeros(len(node_data[name]["vals"]))
                intensity_params += len(node_data[name]["vals"])

        return node_data, intensity_params, transition_params, ctbn_dyn_str, ctbn_bncpts


    def model_from_tscbn_ground_truth(self, tscbn, var_df, path, rename_dict = dict()):

        nodes = set([q.split("_")[0] for q in tscbn.V if not str.startswith(q.split("_")[0], "dL")])
        edges = set([tuple([q.split("_")[0] for q in e]) for e in tscbn.E if (not str.startswith(e[1], "dL_"))])
        edges = [e for e in edges if e[0]!=e[1]] # leave self edges out
        ndata_reduced = dict([(k, tscbn.Vdata[k+"_0"]) for k in nodes])
        node_data, intensity_params, transition_params, ctbn_dyn_str, ctbn_bncpts = self.get_cpd_raw(nodes, edges, ndata_reduced, COUNT_ONLY=False, val_map = rename_dict)

        from_to = "<VARS>\n"
        from_to += "Name,Value\n"
        from_to += var_df.to_string(index=False, header=False).replace("  ", ",")
        from_to += "\n</VARS>"

        bn_struct = "\n<BNSTRUCT>\n"
        bn_struct += "From,To\n"
        bn_struct += pd.DataFrame(list(edges), columns=["From", "To"]).to_string(index=False, header=False).replace("  ", ",")
        bn_struct += "\n</BNSTRUCT>"

        cpds = "\n<BNCPTS>"
        ctbn_bncpts = ctbn_bncpts
        cpds += ctbn_bncpts
        cpds += "\n</BNCPTS>"

        dyn = "\n<DYNSTRUCT>\n"
        dyn += "From,To\n"
        dyn += pd.DataFrame(list(edges), columns=["From", "To"]).to_string(index=False, header=False).replace("  ", ",")
        dyn += "\n</DYNSTRUCT>"

        cims = "\n<DYNCIMS>"
        ctbn_dyn_str = ctbn_dyn_str
        cims += ctbn_dyn_str
        cims += "\n</DYNCIMS>"

        data = from_to + bn_struct + cpds + dyn + cims + '\n'
        import io
        text_file = io.open(os.path.join(path, "net.rctbn"), 'w', newline='\r\n')
        text_file.write(data)
        text_file.close()

        return intensity_params, transition_params, node_data

    def generate_model(self, structure_specification):
        ''' Dynamic Bayesian NW '''

        # 1. define static structure
        ndata, edges, first_time, last_time, resolution = self._create_nodes_edges(structure_specification)

        # 2. learn parameters from it
        edges = set([tuple([q.split("_")[0] for q in e]) for e in edges])
        nodes = set([q.split("_")[0] for q in ndata])
        ndata_reduced = dict([(k, ndata[k+"_0"]) for k in nodes])
        node_data, intensity_params, transition_params, _,_ = self.get_cpd_raw(nodes, edges, ndata_reduced)

        # 3. learn Transition matrices given condition
        # pro edge der zu mir geht habe eine Matrix mit
        # meinen Werten e.g. [0.2 0.5, 0.3]|ParA=0, ParB=1
        # erzeuge also pro ParentValueCombo
        #   matrix mit meinen Transition Werten wie in BN

        # 4. learn Intensity matrix
        # Pro Parent Value Combo eine:
        # Matrix X x X wobei X die Anzahl an Werten sind
        # die diese Variable annehmen kann

        # d.h Modell = nodes, edges + Trans.mats + Int.mats

        self.eval_cpd_entries = transition_params + intensity_params

        # 2. learn parameters from it
        nd = NodeData()
        nd.Vdata = ndata
        skel = GraphSkeleton()
        skel.E = edges
        skel.V = nodes
        skel.alldata = dict()
        skel.alldata["eval_cpd_entries_count"] = self.eval_cpd_entries

        skel.alldata["ndata"] = ndata_reduced# states = transition_matrix (= dim_x + dim_y) + intensity
        return skel

    def _create_edges(self):
        pass

    def _create_nodes_edges(self, spec):
        resolution, max_t, min_t = self._get_resolution(spec["temp_gap_between_objects"], spec["dbn_tolerance"])
        first_time = float(int(min_t / resolution))* resolution
        last_time = float(int(max_t / resolution)+1.0)* resolution

        # run through nodes
        edges = []
        node_data = {}
        nodes_per_object = {}

        for i in range(spec["object_number"]):
            obj_name = spec["object_names"][i]
            nodes_per_object[obj_name] = 0

            # per object create - number of nodes
            prev = None
            for k in range(int(1+((last_time-first_time)/resolution))):
                name = obj_name.replace("O", "V") + "_" +str(k)
                nodes_per_object[obj_name] += 1

                # links
                parents = []
                if not prev is None:
                    edges.append([prev, name])
                    parents.append(prev)
                prev = name

                if not name in node_data: node_data[name] = dict()
                node_data[name]["vals"] = spec["object_states"][obj_name]
                node_data[name]["numoutcomes"] = len(node_data[name]["vals"])
                node_data[name]["parents"] = None

                if not "children" in node_data[name] or not node_data[name]["children"]: node_data[name]["children"] = []

                # set edge of static structure - with same index easy peasy
                # parents

                for t in spec["inter_edges_to_this_object"][int(obj_name.replace("O", ""))]:
                    if k == 0:
                        continue
                    par_node = t.replace("O", "V") + "_" + str(k-1) # shifted BY ONE!!!
                    parents.append(par_node)
                    edges.append([par_node, name]) # node from parent to me BUT INDEX BACKWARD
                node_data[name]["parents"] = parents

                for par in parents:
                    try:
                        node_data[par]["children"].append(name)
                    except:
                        if not par in node_data:
                            node_data[par] = dict()
                        node_data[par]["children"] = [name]

        # for evaluation only - compute cpd entries virtually
        if self.EXPLICIT_DISABLING:
            self.eval_cpd_entries = 0
            for name in spec["object_names"]:
                # ref node
                ref_name = name.replace("O", "V") +"_1"

                # my node number
                node_nr = nodes_per_object[name]

                # my state number
                state_nr = len(node_data[ref_name]["vals"])

                # parent outcome combinations
                parent_outcome_combos = 1
                for p in node_data[ref_name]["parents"]:
                    parent_outcome_combos *= len(node_data[p]["vals"])

                self.eval_cpd_entries = self.eval_cpd_entries + state_nr + (node_nr-1) * state_nr * parent_outcome_combos

        # set random cpds
        if not self.EXPLICIT_DISABLING: # for structure evaluation only
            for name in node_data:
                keys = []
                for p in node_data[name]["parents"]:
                    keys.append(node_data[p]["vals"])

                if keys:
                    node_data[name]["cprob"] = {}
                    for element in itertools.product(*keys):
                        node_data[name]["cprob"][str(list(element))] = np.zeros(len(node_data[name]["vals"]))
                        node_data[name]["cprob"][str(list(element))][0] = 1.0
                else:
                    node_data[name]["cprob"] = np.zeros(len(node_data[name]["vals"]))
                    node_data[name]["cprob"][0] = 1.0

        return node_data, edges, first_time, last_time, resolution

    def _get_resolution(self, temp_gaps, tolerance):

        distances, sums, mins = [],[], []
        for obj in temp_gaps:
            distances += obj
            sums += [np.sum(obj)]
            mins += [np.min(obj)]
        distances = np.array(distances)

        min_d = min(distances)

        for i in list(np.arange(0.0, min_d, tolerance/4))[::-1]:
            t = np.modf(distances/i)[0]
            td = t[np.where(t<=0.5)]*i
            if not np.all(td<tolerance):
                continue
            tu = np.abs((t[np.where(t>0.5)]-1.0)*i)
            if not np.all(tu<tolerance):
                continue
            return i, np.max(sums), np.min(mins)


