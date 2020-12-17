#!/usr/bin/env python 
# -*- coding: utf-8 -*-
from _include.m_libpgm.graphskeleton import GraphSkeleton
from _include.m_libpgm.nodedata import NodeData
import itertools
import numpy as np
from network.dbndiscrete import DBNDiscrete
from _include.structure.base_structure_model import BaseStructureModel


class DBNStructureModel(BaseStructureModel):
    '''
    This generator creates a Dynamic BN from the given specification
    Zustände als Knoten
    '''

    def __init__(self):
        super(DBNStructureModel, self).__init__()

        self.EXPLICIT_DISABLING = False # For evaluation only to save time during generation - avoid CPD loop explosion
        self.eval_cpd_entries = 0 # For evaluation

    def generate_model(self, structure_specification):
        ''' Dynamic Bayesian NW '''

        # 1. define static structure
        ndata, edges, first_time, last_time, resolution = self._create_nodes_edges(structure_specification)

        # 2. learn parameters from it
        nd = NodeData()
        nd.Vdata = ndata
        skel = GraphSkeleton()
        skel.E = edges
        skel.V = list(ndata.keys())
        skel.toporder()
        bn = DBNDiscrete(skel, nd)
        bn.start_time = 0.0#first_time
        bn.end_time = last_time
        bn.resolution = resolution

        if self.EXPLICIT_DISABLING: bn.eval_cpd_entries = self.eval_cpd_entries

        return bn

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


