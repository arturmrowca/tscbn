#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import random
from _include.structure.base_structure_model import BaseStructureModel
from _include.visual.interval_plotter import IntervalPlotter
from network.tscbn import TSCBN
from _include.m_libpgm.graphskeleton import GraphSkeleton
import numpy as np
import copy


class TSCBNStructureModel(BaseStructureModel):
    '''
    This generator creates a Temporal state change BN from the given specification
    '''

    def __init__(self):
        super(TSCBNStructureModel, self).__init__()
        self._diabled_previous_inter_edge = True
        self._temporal_variance = 0.01

    def generate_model(self, structure_specification):

        # 1. create defined number of nodes per object
        v, node_cpds, temp_dict = self._create_nodes(structure_specification)

        # 2. create edges incl. time between vertices
        e, temp_gap_dict = self._create_edges(v, structure_specification, temp_dict)

        # 3. add temporal information
        inverted_temp_dict = self._invert_dict(temp_dict)
        self._temporal_information(v, temp_gap_dict, node_cpds, self._parents_dict_from_edges(e), temp_dict,
                                   inverted_temp_dict)  # node_cpds passed by reference and contain temporal information

        # 4. Skeleton
        skel = GraphSkeleton()
        skel.V = v
        skel.E = e
        skel.toporder()

        # 5. Create Model
        tbn = TSCBN("", skel, node_cpds, unempty=True, forbid_never=True,
                    discrete_only=True)  # Discrete case - later continuous nodes

        # 6. Set cpds of value nodes
        self._set_cpds(tbn, structure_specification)

        return tbn

    def _set_cpds(self, tbn, structure_specification):

        # 1. Probability that a state change occurred
        prob_state_change = structure_specification["state_change"]

        # 2. Node name defined
        for object_idx in range(len(prob_state_change)):
            for node_idx in range(len(prob_state_change[object_idx])):

                # get relevant data
                if node_idx == 0: continue
                vertex = "V%s_%s" % (str(object_idx), str(node_idx))
                parent_vertex = "V%s_%s" % (str(object_idx), str(node_idx - 1))
                probability_no_sc = 1.0 - prob_state_change[object_idx][node_idx]

                # set diagonals to zero
                if isinstance(tbn.nodes[vertex].Vdataentry["cprob"], dict):
                    for k in tbn.nodes[vertex].Vdataentry["cprob"]:
                        # set all diagonals to zero - get preceding parent
                        parent_idx = tbn.nodes[vertex].Vdataentry["parents"].index(parent_vertex)
                        obj = eval(k)[parent_idx]
                        val_idx = tbn.nodes[vertex].Vdataentry["vals"].index(obj)

                        #
                        tbn.nodes[vertex].Vdataentry["cprob"][k][val_idx] = 0.0
                        tbn.nodes[vertex].Vdataentry["cprob"][k] = (1.0 - probability_no_sc) * (
                                    tbn.nodes[vertex].Vdataentry["cprob"][k] / sum(
                                tbn.nodes[vertex].Vdataentry["cprob"][k]))
                        tbn.nodes[vertex].Vdataentry["cprob"][k][val_idx] = probability_no_sc

                        tbn.Vdata[vertex]["cprob"][k] = tbn.nodes[vertex].Vdataentry["cprob"][k]

    def _invert_dict(self, d_dict):
        out_d = {}

        for d in d_dict:
            out_d[d_dict[d]] = d
        return out_d

    def _parents_dict_from_edges(self, edges):
        '''
        From a list of edges extract parents per edge
        '''
        res_dict = {}
        for e in edges:
            if e[1] in res_dict and not e[0] in res_dict[e[1]]:
                res_dict[e[1]].append(e[0])
            else:
                res_dict[e[1]] = [e[0]]
        return res_dict

    def _temporal_information(self, vertices, temp_gap_dict, node_cpds, parents, temp_dict, inv_temp_dict):
        # Mean value ist der Abstand wenn ich alle meine Parents anschaue und dann
        # den nehme der am nähesten bei mir ist
        for v in vertices:
            # welches ist spaeter passiert
            # - has no parent
            variance = self._temporal_variance
            if v not in parents:
                dL = 0
            else:
                # has parent
                pars = np.array([k for k in temp_dict if temp_dict[k] in parents[v]])
                if len(pars) == 0:
                    fr_n = parents[v][0]
                else:
                    idx = max(pars)
                    fr_n = temp_dict[idx]
                # keey = str([fr_n, v])
                try:
                    dL = inv_temp_dict[v] - inv_temp_dict[fr_n]  # temp_gap_dict[keey]
                except:
                    dL = inv_temp_dict[v]  # if it does not appear it is a parent
            if v.split("_")[-1] == "0":
                dL = 0
                variance = 0
            node_cpds[v]["dL_mean"] = dL
            node_cpds[v]["dL_var"] = variance  # assume little variance

    def _create_nodes(self, spec):
        v, node_cpds = [], dict()
        temp_dict = {}  # key: absolute time, value: vertex

        # Initial nodes
        for i in range(spec["object_number"]):
            number_of_nodes_per_obj = spec["per_object_chain_number"][i]
            obj_name = spec["object_names"][i]
            node_names = self._get_node_names(obj_name, number_of_nodes_per_obj)
            states = spec["object_states"][obj_name]
            # t_gaps = spec["temp_gap_between_objects"]
            t = 0
            kk = -1
            for n_name in node_names:
                kk += 1
                v += self._dynamic_node(n_name, "disc", states,
                                        node_cpds)  # self._dynamic_node(n_name, "disc", states + ["Never"], node_cpds)
                temp_dict[t] = n_name
                try:
                    t += spec["temp_gap_between_objects"][i][kk]
                except:
                    pass
        return v, node_cpds, temp_dict

    def _add_temporal_basics_dump(self, dL_mean, dL_var, node_cpds, v, e):
        ''' basic part that is same for all cases

        Note to myself: Pro Discrete Parent Combination - habe eigene Mean, und habe eigene Variance
                        Pro Continuous Parent - habe keine distributions - weil sich einfach kombinieren lässt de Sach
        Type definition:  http://pythonhosted.org/libpgm/CPDtypes.html
            If i bim continuous:
                - Wenn ich nur continuous parents hob, oder goa koane parents - then type: lg
                - Wenn ich discrete parents hob oder discrete und continuous - type: lgandd

            If i bim discrete
                - if I have no parents or discrete parents - type is: discrete
                - if I have continuous parents - type is: "to be implemented" - do it yourself bro
        '''

        # 1. Require initial node, per temporal variable to be at t=0

        # 1. tp and dL information needs to be given - is distributed in any case
        #    currently same for all nodes - in reality distinct
        for vert in v:
            node_cpds[vert]["dL_mean"] = dL_mean
            node_cpds[vert]["dL_var"] = dL_var

        # Generate Network from this information
        skel = GraphSkeleton()
        skel.V = v
        skel.E = e
        try:
            skel.toporder()
        except:
            print("Warning -> Graph has cycles -> may be irrelevant depending on use-case")

        return skel, node_cpds


    def _create_edges(self, vertices, structure_specification, temp_dict):
        e = []
        self._temporal_variance = structure_specification["temporal_variance"]

        # 1. add self dependencies
        e += self._self_dependencies(vertices, [u.replace("O", "V") for u in structure_specification["object_names"]])

        # 2. add connection between objects - Abh. von mir und meinem Vorgaenger
        # kenne pro Objekt die Distanz zwischen den Zustandswechseln
        # store all abs times to values - then choose what I need
        temp_gap = {}  # stores the gap between object and its son per object
        times_a = []
        for i in range(structure_specification["object_number"]):
            # get relevant subset of our dictionary
            rel_sub_list = structure_specification['inter_edges_to_this_object'][i] + [
                structure_specification["object_names"][i]]
            sub_dict = self._get_subdict(rel_sub_list, temp_dict)
            target = structure_specification["object_names"][i].replace("O", "V")

            latest_time_of_target = sorted([k for k in temp_dict.keys() if str.startswith(temp_dict[k], target)])[-1]
            # choose number of destination nodes to connect
            try:
                sub_dict = self._choose_subset_of_nodes(structure_specification, sub_dict, i, temp_dict,
                                                        latest_time_of_target)
            except:
                sub_dict = self._choose_subset_of_nodes(structure_specification, sub_dict, i, temp_dict,
                                                        latest_time_of_target)

            times = sorted(list(sub_dict.keys()))
            times_a += [times]

            last = {}
            for time in times:
                if temp_dict[time].split("_")[0] == target:
                    # edge from all entries in last to target
                    for app_time in last:
                        new_e = [last[app_time], temp_dict[time]]
                        # Zusaetzlich brauche Verbindung zum Vorgaenger per Definition (oder auch nicht)
                        if not self._diabled_previous_inter_edge:
                            nex = int(last[app_time].split("_")[1]) - 1
                            if nex != -1:
                                fr_n = last[app_time].split("_")[0] + "_" + str(nex)
                                new_e2 = [fr_n, temp_dict[time]]

                                if not new_e2 in e:
                                    e += [new_e2]
                                    t = None
                                    for k, v in temp_dict.items():
                                        if v == fr_n:
                                            t = k
                                            break
                                    # store temporal gap to this object
                                    if str(nex) == "0": k = 0.0
                                    temp_gap[str(new_e2)] = time - k

                        # Add if not existing
                        if not new_e in e:
                            e += [new_e]
                            # store temporal gap to this object
                            temp_gap[str(new_e)] = time - app_time

                    last = {}
                else:
                    last[time] = temp_dict[time]

        return e, temp_gap

    def _random_pick_uniform(self, elements, nr_elements):
        res = []
        l = copy.deepcopy(list(elements.keys()))
        for _ in range(nr_elements):
            idx = l[round(random.random() * len(l)) - 1]
            res += [elements[idx]]
            l.remove(idx)
        return res

    def _choose_subset_of_nodes(self, structure_specification, sub_dict, i, temp_dict, latest_time_of_target):
        '''
        Fordere das mein letzter Knoten der letze sein muss!
        :param structure_specification:
        :param sub_dict:
        :param i:
        :param temp_dict:
        :return:
        '''
        node_number_per_object = int(structure_specification['nodes_per_object'][i])
        pot_connects = self._get_subdict(structure_specification['inter_edges_to_this_object'][i], temp_dict)

        # pot connects are only the ones occurring before me!
        if len(pot_connects) < node_number_per_object:
            raise ValueError(
                "Number of nodes to connect is %s, while only %s valid nodes were found - change percentage_inter value (lower)" % (
                str(node_number_per_object), str(pot_connects)))
        pot_connects = dict([(p, pot_connects[p]) for p in pot_connects if p < latest_time_of_target])

        nodes_to_connect = self._random_pick_uniform(pot_connects, node_number_per_object)
        rel_dict = {}
        dest_vals = list(
            self._get_subdict([structure_specification["object_names"][i]], temp_dict).values()) + nodes_to_connect
        for k in sub_dict:
            if sub_dict[k] in dest_vals:
                rel_dict[k] = sub_dict[k]
        return rel_dict

    def _get_subdict(self, rel_objects, temp_dict):
        '''
        takes a list of relevant objects and returns the relevant part of the dictionary
        '''
        rel_vers = [u.replace("O", "V") for u in rel_objects]
        sub = {}
        for k in temp_dict:
            if temp_dict[k].split("_")[0] in rel_vers:
                sub[k] = temp_dict[k]
        return sub

    def _get_node_names(self, obj_name, number_of_nodes_per_obj):
        new_name = obj_name.replace("O", "V")
        return ["%s_%s" % (str(new_name), str(k)) for k in range(number_of_nodes_per_obj)]

    def unique(self, sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    def _dynamic_node(self, node_name, type_dc, vals, node_cpds):
        node_cpds[node_name] = dict()
        node_cpds[node_name]["vals"] = self.unique(
            [str(a) for a in vals if not str(a) == "nan"])  # nan is not an outcome!
        node_cpds[node_name]["type_dc"] = type_dc  # preliminary - will be adjusted depending on parents!
        return [node_name]

    def _self_dependencies(self, vertices, rel_nodes):
        try:
            edges = []
            verts = sorted(vertices)
            fr = verts[0]

            for v in verts[1:]:
                to = v
                fst = fr.split("_")[0]  # ''.join([i for i in fr if not i.isdigit()])
                if fst == to.split("_")[0] and fst in rel_nodes:
                    edges.append([fr, to])
                fr = to
            return edges
        except:
            return []

    def _self_dependencies_digit_format(self, vertices, rel_nodes):
        try:
            edges = []
            verts = sorted(vertices)
            fr = verts[0]

            for v in verts[1:]:
                to = v
                fst = ''.join([i for i in fr if not i.isdigit()])
                if fst == ''.join([i for i in to if not i.isdigit()]) and fst in rel_nodes:
                    edges.append([fr, to])
                fr = to
            return edges
        except:
            return []

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
                    print(ver + " Parents: " + str(nodes[ver].Vdataentry["parents"]))
                    print(gap_ver + " Parents: " + str(nodes[gap_ver].Vdataentry["parents"]))
                    print("Mean dL: %s" % str(nodes[gap_ver].Vdataentry["mean_base"]))
                    print("\n")
                except:
                    break

    def _show_random_samples(self, tscbn, nr_samples, da_sep = "_"):
        evidence = {}
        ctbn = copy.deepcopy(tscbn)

        # Show results
        # a. Print distribution
        #ctbn.print_distributions()

        # b. Print/Plot samples
        print("\n")
        samps = ctbn.randomsample(nr_samples, evidence)
        kk = 0
        for a in samps:
            kk += 1
            print("\n\nSAMPLE %s: " % str(kk));
            i = -1
            for k in a:
                i += 1
                # if i%2==0:print(" ")
                print("%s:%s" % (str(k), str(a[k])))

        IntervalPlotter().plot_random_example_tbn(samps, ctbn.Vdata, sep = da_sep)

class TSCBNSimpleStructureModel(TSCBNStructureModel):

    def generate_model(self, spec = None):


        #tbn = self._model_1()

        #tbn = self._model_2()

        #tbn = self._model_3()

        tbn = self._easy_model_1()

        #tbn = self._super_easy_model_1()

        #samps = tbn.randomsample(2, {})
        #IntervalPlotter().plot_random_example_tbn(samps, tbn.Vdata, sep="_")

        return tbn

    def _super_easy_model_1(self):
        # Reference model
        # Define Vertices
        v, node_cpds = [], dict()
        defaults = {"V0_0": "o0_0"}

        # Initial nodes
        v += self._dynamic_node("V0_0", "disc", ["o0_0", "o0_1", "o0_2"], node_cpds)  # NEVER STATE IST SINNLOS

        # More nodes
        v += self._dynamic_node("V0_1", "disc", ["o0_0", "o0_1", "o0_2"], node_cpds)  # NEVER STATE IST SINNLOS

        # Succeeding
        v += self._dynamic_node("V0_2", "disc", ["o0_0", "o0_1", "o0_2"], node_cpds)  # NEVER STATE IST SINNLOS

        v += self._dynamic_node("V0_3", "disc", ["o0_0", "o0_1", "o0_2"], node_cpds)  # NEVER STATE IST SINNLOS

        # Define Edges
        e = []
        e += self._self_dependencies(v, ["V0"])

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
        #tscbn.draw("ext")

        # set some probabilities that I am sure about
        tscbn.Vdata["V0_0"]["cprob"] = np.array([0.2, 0.5, 0.3])

        # wenn V11 auf an geht geht auch StaLicht1 auf ein
        tscbn.Vdata["V0_1"]["cprob"]["['o0_0']"] = np.array([0.2, 0.3, 0.5])
        tscbn.Vdata["V0_1"]["cprob"]["['o0_1']"] = np.array([0.2, 0.3, 0.5])
        tscbn.Vdata["V0_1"]["cprob"]["['o0_2']"] = np.array([0.2, 0.3, 0.5])

        tscbn.Vdata["V0_2"]["cprob"]["['o0_0']"] = np.array([0.2, 0.3, 0.5])
        tscbn.Vdata["V0_2"]["cprob"]["['o0_1']"] = np.array([0.2, 0.3, 0.5])
        tscbn.Vdata["V0_2"]["cprob"]["['o0_2']"] = np.array([0.2, 0.3, 0.5])

        tscbn.Vdata["V0_3"]["cprob"]["['o0_0']"] = np.array([0.2, 0.3, 0.5])
        tscbn.Vdata["V0_3"]["cprob"]["['o0_1']"] = np.array([0.2, 0.3, 0.5])
        tscbn.Vdata["V0_3"]["cprob"]["['o0_2']"] = np.array([0.2, 0.3, 0.5])

        return tscbn

    def _easy_model_1(self):
        # Reference model
        # Define Vertices
        v, node_cpds = [], dict()
        defaults = {"V0_0": "o0_0", "V1_0": "o1_0", "V2_0": "o2_1"}

        # Initial nodes
        v += self._dynamic_node("V0_0", "disc", ["o0_0", "o0_1", "o0_2"], node_cpds)  # NEVER STATE IST SINNLOS
        v += self._dynamic_node("V1_0", "disc", ["o1_0", "o1_1", "o1_2"],
                                node_cpds)  # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("V2_0", "disc", ["o2_0", "o2_1", "o2_2"], node_cpds)

        # More nodes
        v += self._dynamic_node("V0_1", "disc", ["o0_0", "o0_1", "o0_2"], node_cpds)  # NEVER STATE IST SINNLOS
        v += self._dynamic_node("V1_1", "disc", ["o1_0", "o1_1", "o1_2"],
                                node_cpds)  # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("V2_1", "disc", ["o2_0", "o2_1", "o2_2"], node_cpds)

        # Succeeding
        v += self._dynamic_node("V0_2", "disc", ["o0_0", "o0_1", "o0_2"], node_cpds)  # NEVER STATE IST SINNLOS
        v += self._dynamic_node("V1_2", "disc", ["o1_0", "o1_1", "o1_2"],
                                node_cpds)  # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("V2_2", "disc", ["o2_0", "o2_1", "o2_2"], node_cpds)

        v += self._dynamic_node("V0_3", "disc", ["o0_0", "o0_1", "o0_2"], node_cpds)  # NEVER STATE IST SINNLOS
        v += self._dynamic_node("V1_3", "disc", ["o1_0", "o1_1", "o1_2"],
                                node_cpds)  # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("V2_3", "disc", ["o2_0", "o2_1", "o2_2"], node_cpds)

        # Define Edges
        e = []
        e += self._self_dependencies(v, ["V0", "V1", "V2"])
        e += [["V0_1", "V1_2"]]  # Bedienung passiert immer vor Steuerung
        e += [["V2_1", "V0_2"]]  # Steuerung passiert immer vor Status

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
        #tscbn.draw("ext")

        # set some probabilities that I am sure about
        tscbn.Vdata["V0_0"]["cprob"] = np.array([0.2, 0.5, 0.3])
        tscbn.Vdata["V1_0"]["cprob"] = np.array([0.5, 0.2, 0.3])
        tscbn.Vdata["V2_0"]["cprob"] = np.array([0.3, 0.2, 0.5])

        # wenn V11 auf an geht geht auch StaLicht1 auf ein
        tscbn.Vdata["V0_1"]["cprob"]["['o0_0']"] = np.array([0.2, 0.3, 0.5])
        tscbn.Vdata["V0_1"]["cprob"]["['o0_1']"] = np.array([0.6, 0.2, 0.2])
        tscbn.Vdata["V0_1"]["cprob"]["['o0_2']"] = np.array([0.2, 0.6, 0.2])

        tscbn.Vdata["V0_2"]["cprob"]["['o0_0', 'o2_0']"] = np.array([0.2, 0.3, 0.5])
        tscbn.Vdata["V0_2"]["cprob"]["['o0_0', 'o2_1']"] = np.array([0.2, 0.6, 0.2])
        tscbn.Vdata["V0_2"]["cprob"]["['o0_0', 'o2_2']"] = np.array([0.2, 0.6, 0.2])
        tscbn.Vdata["V0_2"]["cprob"]["['o0_1', 'o2_0']"] = np.array([0.1, 0.2, 0.7])
        tscbn.Vdata["V0_2"]["cprob"]["['o0_1', 'o2_1']"] = np.array([0.7, 0.2, 0.1])
        tscbn.Vdata["V0_2"]["cprob"]["['o0_1', 'o2_2']"] = np.array([0.2, 0.2, 0.6])
        tscbn.Vdata["V0_2"]["cprob"]["['o0_2', 'o2_0']"] = np.array([0.5, 0.3, 0.2])
        tscbn.Vdata["V0_2"]["cprob"]["['o0_2', 'o2_1']"] = np.array([0.6, 0.2, 0.2])
        tscbn.Vdata["V0_2"]["cprob"]["['o0_2', 'o2_2']"] = np.array([0.2, 0.6, 0.2])

        tscbn.Vdata["V1_1"]["cprob"]["['o1_0']"] = np.array([0.2, 0.2, 0.6])
        tscbn.Vdata["V1_1"]["cprob"]["['o1_1']"] = np.array([0.6, 0.2, 0.2])
        tscbn.Vdata["V1_1"]["cprob"]["['o1_2']"] = np.array([0.2, 0.6, 0.2])

        tscbn.Vdata["V1_2"]["cprob"]["['o1_0', 'o0_0']"] = np.array([0.2, 0.3, 0.5])
        tscbn.Vdata["V1_2"]["cprob"]["['o1_0', 'o0_1']"] = np.array([0.2, 0.6, 0.2])
        tscbn.Vdata["V1_2"]["cprob"]["['o1_0', 'o0_2']"] = np.array([0.2, 0.6, 0.2])
        tscbn.Vdata["V1_2"]["cprob"]["['o1_1', 'o0_0']"] = np.array([0.1, 0.2, 0.7])
        tscbn.Vdata["V1_2"]["cprob"]["['o1_1', 'o0_1']"] = np.array([0.6, 0.2, 0.2])
        tscbn.Vdata["V1_2"]["cprob"]["['o1_1', 'o0_2']"] = np.array([0.2, 0.2, 0.6])
        tscbn.Vdata["V1_2"]["cprob"]["['o1_2', 'o0_0']"] = np.array([0.5, 0.3, 0.2])
        tscbn.Vdata["V1_2"]["cprob"]["['o1_2', 'o0_1']"] = np.array([0.6, 0.2, 0.2])
        tscbn.Vdata["V1_2"]["cprob"]["['o1_2', 'o0_2']"] = np.array([0.2, 0.6, 0.2])

        tscbn.Vdata["V2_1"]["cprob"]["['o2_0']"] = np.array([0.1, 0.3, 0.6])
        tscbn.Vdata["V2_1"]["cprob"]["['o2_1']"] = np.array([0.7, 0.1, 0.2])
        tscbn.Vdata["V2_1"]["cprob"]["['o2_2']"] = np.array([0.2, 0.7, 0.1])

        tscbn.Vdata["V2_2"]["cprob"]["['o2_0']"] = np.array([0.1, 0.3, 0.6])
        tscbn.Vdata["V2_2"]["cprob"]["['o2_1']"] = np.array([0.7, 0.1, 0.2])
        tscbn.Vdata["V2_2"]["cprob"]["['o2_2']"] = np.array([0.2, 0.7, 0.1])

        tscbn.Vdata["V2_3"]["cprob"]["['o2_0']"] = np.array([0.2, 0.2, 0.6])
        tscbn.Vdata["V2_3"]["cprob"]["['o2_1']"] = np.array([0.6, 0.2, 0.2])
        tscbn.Vdata["V2_3"]["cprob"]["['o2_2']"] = np.array([0.2, 0.6, 0.2])

        return tscbn


    def _model_1(self):
        '''
        Optimalfall - Alle State Changes treten auf
        :return:
        '''
        # ------------------------------------------------------------------
        #                Define Basic Structure - Manually
        # ------------------------------------------------------------------

        # Define Vertices
        v, node_cpds = [], dict()
        defaults = {"BedLicht_0" : "Aus_bed", "StgLicht_0" : "Aus", "StaLicht_0" : "Aus"}

        # Initial nodes
        v += self._dynamic_node("BedLicht_0", "disc", ["Aus_bed", "An_bed"], node_cpds) # NEVER STATE IST SINNLOS
        v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An"], node_cpds) # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("StgLicht_0", "disc", ["Aus", "An"], node_cpds)

        # More nodes
        v += self._dynamic_node("BedLicht_1", "disc", ["Aus_bed", "An_bed"], node_cpds)
        v += self._dynamic_node("StaLicht_1", "disc", ["Aus", "An"], node_cpds)
        v += self._dynamic_node("StgLicht_1", "disc", ["Aus", "An"], node_cpds)

        # Succeeding
        v += self._dynamic_node("BedLicht_2", "disc", ["Aus_bed", "An_bed"], node_cpds)
        v += self._dynamic_node("StaLicht_2", "disc", ["Aus", "An"], node_cpds)
        v += self._dynamic_node("StgLicht_2", "disc", ["Aus", "An"], node_cpds)

        # Define Edges
        e = []
        e += self._self_dependencies(v, ["BedLicht", "StaLicht", "StgLicht"])
        e += [["BedLicht_1", "StgLicht_1"]]  # Bedienung passiert immer vor Steuerung
        e += [["StgLicht_1", "StaLicht_1"]] # Steuerung passiert immer vor Status

        # ------------------------------------------------------------------
        #                Define Temporal Information - Manually
        # ------------------------------------------------------------------
        dL_mean = 3
        dL_var = 1
        skel, node_cpds = self._add_temporal_basics(dL_mean, dL_var, node_cpds, v, e, defaults)

        # ------------------------------------------------------------------
        #                Create Network
        # ------------------------------------------------------------------
        tscbn = TSCBN("", skel, node_cpds, unempty = True, forbid_never = False, discrete_only = True, default_states = defaults, default_is_distributed = True)  # Discrete case - later continuous nodes
        tscbn.draw("ext")

        # set some probabilities that I am sure about
        tscbn.Vdata["StgLicht_0"]["cprob"] = np.array([0.5, 0.5])
        tscbn.Vdata["BedLicht_0"]["cprob"] = np.array([0.5, 0.5])
        tscbn.Vdata["StaLicht_0"]["cprob"] = np.array([0.5, 0.5])

        # wenn StgLicht1 auf an geht geht auch StaLicht1 auf ein
        tscbn.Vdata["StgLicht_1"]["cprob"]["['Aus', 'Aus_bed']"] = np.array([0.0, 1.0])
        tscbn.Vdata["StgLicht_1"]["cprob"]["['Aus', 'An_bed']"] = np.array([0.0, 1.0]) # Halbfalsch
        tscbn.Vdata["StgLicht_1"]["cprob"]["['An', 'Aus_bed']"] = np.array([1.0, 0]) # Halbfalsch
        tscbn.Vdata["StgLicht_1"]["cprob"]["['An', 'An_bed']"] = np.array([1.0, 0.0]) # Muss an bleiben - weil ich würde hier das event An sehen - es würde auftreten -
                                                                                       # das würde man in dem Intervall nicht sehen!!!!!
                                                                                       # ABER: es würde ja StaLicht beeinflussen - weil es ja tatsächlich passiert ist
                                                                                       # d.h. zum Samplen muss ich das so machen
                                                                                       # Never würde hier Fehler bedeuten - also BedLicht ging an aber StgLicht hat Nie reagiert somit hat StaLicht nie reagiert
        tscbn.Vdata["StgLicht_2"]["cprob"]["['Aus']"] = np.array([0.0, 1.0]) # FALSCH NOCH
        tscbn.Vdata["StgLicht_2"]["cprob"]["['An']"] = np.array([1.0, 0.0]) # FALSCH NOCH

        tscbn.Vdata["BedLicht_1"]["cprob"]["['Aus_bed']"] = np.array([0.0, 1.0]) #
        tscbn.Vdata["BedLicht_1"]["cprob"]["['An_bed']"] = np.array([1.0, 0.0])

        tscbn.Vdata["BedLicht_2"]["cprob"]["['Aus_bed']"] = np.array([0.0, 1.0])
        tscbn.Vdata["BedLicht_2"]["cprob"]["['An_bed']"] = np.array([1.0, 0.0])

        tscbn.Vdata["StaLicht_1"]["cprob"]["['Aus', 'Aus']"] = np.array([0.0, 1.0])
        tscbn.Vdata["StaLicht_1"]["cprob"]["['Aus', 'An']"] = np.array([0.0, 1.0]) # FALSCH NOCH
        tscbn.Vdata["StaLicht_1"]["cprob"]["['An', 'Aus']"] = np.array([1.0, 0.0])  # FALSCH NOCH
        tscbn.Vdata["StaLicht_1"]["cprob"]["['An', 'An']"] = np.array([1.0, 0.0])

        tscbn.Vdata["StaLicht_2"]["cprob"]["['Aus']"] = np.array([0.0, 1.0]) # hier macht never sinn - weil ich habe keinen kausalen Einfluss der meinen Wert auf ein Signal setzen würde
                                                                              # stattdessen bleibe ich Aus einfach weil nix passiert ist und ich passiere somit auch nicht
                                                                              # Frage? Warum bin ich dann überhaupt da - weil kann sein das ich mit 0.9 Never bin aber mit 0.1 Aus weil ich
                                                                              # mich spontan gewechselt habe
        tscbn.Vdata["StaLicht_2"]["cprob"]["['An']"] = np.array([1.0, 0.0])

        return tscbn

    def _model_2(self):
        '''
        Wahrscheinlichkeit für einen Zustandswechsel ist immer größer als die für ein bleiben im selben
        Zustand
        '''

        # ------------------------------------------------------------------
        #                Define Basic Structure - Manually
        # ------------------------------------------------------------------

        # Define Vertices
        v, node_cpds = [], dict()
        defaults = {"BedLicht_0" : "Aus_bed", "StgLicht_0" : "Aus", "StaLicht_0" : "Aus"}

        # Initial nodes
        v += self._dynamic_node("BedLicht_0", "disc", ["Aus_bed", "An_bed"], node_cpds) # NEVER STATE IST SINNLOS
        v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An"], node_cpds) # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("StgLicht_0", "disc", ["Aus", "An"], node_cpds)

        # More nodes
        v += self._dynamic_node("BedLicht_1", "disc", ["Aus_bed", "An_bed"], node_cpds)
        v += self._dynamic_node("StaLicht_1", "disc", ["Aus", "An"], node_cpds)
        v += self._dynamic_node("StgLicht_1", "disc", ["Aus", "An"], node_cpds)

        # Succeeding
        v += self._dynamic_node("BedLicht_2", "disc", ["Aus_bed", "An_bed"], node_cpds)
        v += self._dynamic_node("StaLicht_2", "disc", ["Aus", "An"], node_cpds)
        v += self._dynamic_node("StgLicht_2", "disc", ["Aus", "An"], node_cpds)

        # Define Edges
        e = []
        e += self._self_dependencies(v, ["BedLicht", "StaLicht", "StgLicht"])
        e += [["BedLicht_1", "StgLicht_1"]]  # Bedienung passiert immer vor Steuerung
        e += [["StgLicht_1", "StaLicht_1"]] # Steuerung passiert immer vor Status

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
        tscbn.draw("ext")

        # set some probabilities that I am sure about
        tscbn.Vdata["StgLicht_0"]["cprob"] = np.array([0.5, 0.5])
        tscbn.Vdata["BedLicht_0"]["cprob"] = np.array([0.5, 0.5])
        tscbn.Vdata["StaLicht_0"]["cprob"] = np.array([0.5, 0.5])

        # wenn StgLicht1 auf an geht geht auch StaLicht1 auf ein
        tscbn.Vdata["StgLicht_1"]["cprob"]["['Aus', 'Aus_bed']"] = np.array([0.2, 0.8])
        tscbn.Vdata["StgLicht_1"]["cprob"]["['Aus', 'An_bed']"] = np.array([0.2, 0.8])  # Halbfalsch
        tscbn.Vdata["StgLicht_1"]["cprob"]["['An', 'Aus_bed']"] = np.array([0.8, 0.2])  # Halbfalsch
        tscbn.Vdata["StgLicht_1"]["cprob"]["['An', 'An_bed']"] = np.array( [0.8, 0.2])  # Muss an bleiben - weil ich würde hier das event An sehen - es würde auftreten -
        # das würde man in dem Intervall nicht sehen!!!!!
        # ABER: es würde ja StaLicht beeinflussen - weil es ja tatsächlich passiert ist
        # d.h. zum Samplen muss ich das so machen
        # Never würde hier Fehler bedeuten - also BedLicht ging an aber StgLicht hat Nie reagiert somit hat StaLicht nie reagiert
        tscbn.Vdata["StgLicht_2"]["cprob"]["['Aus']"] = np.array([0.2, 0.8])  # FALSCH NOCH
        tscbn.Vdata["StgLicht_2"]["cprob"]["['An']"] = np.array([0.8, 0.2])  # FALSCH NOCH

        tscbn.Vdata["BedLicht_1"]["cprob"]["['Aus_bed']"] = np.array([0.2, 0.8])  #
        tscbn.Vdata["BedLicht_1"]["cprob"]["['An_bed']"] = np.array([0.8, 0.2])

        tscbn.Vdata["BedLicht_2"]["cprob"]["['Aus_bed']"] = np.array([0.2, 0.8])
        tscbn.Vdata["BedLicht_2"]["cprob"]["['An_bed']"] = np.array([0.8, 0.2])

        tscbn.Vdata["StaLicht_1"]["cprob"]["['Aus', 'Aus']"] = np.array([0.2, 0.8])
        tscbn.Vdata["StaLicht_1"]["cprob"]["['Aus', 'An']"] = np.array([0.2, 0.8])  # FALSCH NOCH
        tscbn.Vdata["StaLicht_1"]["cprob"]["['An', 'Aus']"] = np.array([0.8, 0.2])  # FALSCH NOCH
        tscbn.Vdata["StaLicht_1"]["cprob"]["['An', 'An']"] = np.array([0.8, 0.2])

        tscbn.Vdata["StaLicht_2"]["cprob"]["['Aus']"] = np.array([0.2,
                                                                  0.8])  # hier macht never sinn - weil ich habe keinen kausalen Einfluss der meinen Wert auf ein Signal setzen würde
        # stattdessen bleibe ich Aus einfach weil nix passiert ist und ich passiere somit auch nicht
        # Frage? Warum bin ich dann überhaupt da - weil kann sein das ich mit 0.9 Never bin aber mit 0.1 Aus weil ich
        # mich spontan gewechselt habe
        tscbn.Vdata["StaLicht_2"]["cprob"]["['An']"] = np.array([0.8, 0.2])

        return tscbn

    def _model_3(self):
        '''
        Realistische Konstellation - Wahrscheinlichkeiten ab und zu auf Zustandswechsel ab und zu keiner

        Wobei es ja sinnlos ist zu sagen - ich mache ein Modell das SC aufzeichnet und dann habe ich aber Daten die
        nie State Changes haben -
        '''

        # ------------------------------------------------------------------
        #                Define Basic Structure - Manually
        # ------------------------------------------------------------------

        # Define Vertices
        v, node_cpds = [], dict()
        defaults = {"BedLicht_0" : "Aus_bed", "StgLicht_0" : "Aus", "StaLicht_0" : "Aus"}

        # Initial nodes
        v += self._dynamic_node("BedLicht_0", "disc", ["Aus_bed", "An_bed"], node_cpds) # NEVER STATE IST SINNLOS
        v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An"], node_cpds) # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("StgLicht_0", "disc", ["Aus", "An"], node_cpds)

        # More nodes
        v += self._dynamic_node("BedLicht_1", "disc", ["Aus_bed", "An_bed"], node_cpds)
        v += self._dynamic_node("StaLicht_1", "disc", ["Aus", "An"], node_cpds)
        v += self._dynamic_node("StgLicht_1", "disc", ["Aus", "An"], node_cpds)

        # Succeeding
        v += self._dynamic_node("BedLicht_2", "disc", ["Aus_bed", "An_bed"], node_cpds)
        v += self._dynamic_node("StaLicht_2", "disc", ["Aus", "An"], node_cpds)
        v += self._dynamic_node("StgLicht_2", "disc", ["Aus", "An"], node_cpds)

        # Define Edges
        e = []
        e += self._self_dependencies(v, ["BedLicht", "StaLicht", "StgLicht"])
        e += [["BedLicht_1", "StgLicht_1"]]  # Bedienung passiert immer vor Steuerung
        e += [["StgLicht_1", "StaLicht_1"]] # Steuerung passiert immer vor Status

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
        tscbn.draw("ext")

        # set some probabilities that I am sure about
        tscbn.Vdata["StgLicht_0"]["cprob"] = np.array([0.5, 0.5])
        tscbn.Vdata["BedLicht_0"]["cprob"] = np.array([0.5, 0.5])
        tscbn.Vdata["StaLicht_0"]["cprob"] = np.array([0.5, 0.5])

        # wenn StgLicht1 auf an geht geht auch StaLicht1 auf ein
        tscbn.Vdata["StgLicht_1"]["cprob"]["['Aus', 'Aus_bed']"] = np.array([0.8, 0.2])
        tscbn.Vdata["StgLicht_1"]["cprob"]["['Aus', 'An_bed']"] = np.array([0.2, 0.8])  # Halbfalsch
        tscbn.Vdata["StgLicht_1"]["cprob"]["['An', 'Aus_bed']"] = np.array([0.8, 0.2])  # Halbfalsch
        tscbn.Vdata["StgLicht_1"]["cprob"]["['An', 'An_bed']"] = np.array( [0.2, 0.8])  # Muss an bleiben - weil ich würde hier das event An sehen - es würde auftreten -
        # das würde man in dem Intervall nicht sehen!!!!!
        # ABER: es würde ja StaLicht beeinflussen - weil es ja tatsächlich passiert ist
        # d.h. zum Samplen muss ich das so machen
        # Never würde hier Fehler bedeuten - also BedLicht ging an aber StgLicht hat Nie reagiert somit hat StaLicht nie reagiert
        tscbn.Vdata["StgLicht_2"]["cprob"]["['Aus']"] = np.array([0.8, 0.2])  # FALSCH NOCH
        tscbn.Vdata["StgLicht_2"]["cprob"]["['An']"] = np.array([0.2, 0.8])  # FALSCH NOCH

        tscbn.Vdata["BedLicht_1"]["cprob"]["['Aus_bed']"] = np.array([0.1, 0.9])  #
        tscbn.Vdata["BedLicht_1"]["cprob"]["['An_bed']"] = np.array([0.9, 0.1])

        tscbn.Vdata["BedLicht_2"]["cprob"]["['Aus_bed']"] = np.array([0.9, 0.1])
        tscbn.Vdata["BedLicht_2"]["cprob"]["['An_bed']"] = np.array([0.1, 0.9])

        tscbn.Vdata["StaLicht_1"]["cprob"]["['Aus', 'Aus']"] = np.array([0.8, 0.2])
        tscbn.Vdata["StaLicht_1"]["cprob"]["['Aus', 'An']"] = np.array([0.2, 0.8])  # FALSCH NOCH
        tscbn.Vdata["StaLicht_1"]["cprob"]["['An', 'Aus']"] = np.array([0.8, 0.2])  # FALSCH NOCH
        tscbn.Vdata["StaLicht_1"]["cprob"]["['An', 'An']"] = np.array([0.2, 0.8])

        tscbn.Vdata["StaLicht_2"]["cprob"]["['Aus']"] = np.array([0.8, 0.2])  # hier macht never sinn - weil ich habe keinen kausalen Einfluss der meinen Wert auf ein Signal setzen würde
        # stattdessen bleibe ich Aus einfach weil nix passiert ist und ich passiere somit auch nicht
        # Frage? Warum bin ich dann überhaupt da - weil kann sein das ich mit 0.9 Never bin aber mit 0.1 Aus weil ich
        # mich spontan gewechselt habe
        tscbn.Vdata["StaLicht_2"]["cprob"]["['An']"] = np.array([0.2, 0.8])

        return tscbn

    def _model_4(self):
        '''
            Größeres Modell -
        '''

        # ------------------------------------------------------------------
        #                Define Basic Structure - Manually
        # ------------------------------------------------------------------

        # Define Vertices
        v, node_cpds = [], dict()
        defaults = {"BedLicht_0" : "Aus_bed", "StgLicht_0" : "Aus", "StaLicht_0" : "Aus"}

        # Initial nodes
        v += self._dynamic_node("BedLicht_0", "disc", ["Aus_bed", "An_bed"], node_cpds) # NEVER STATE IST SINNLOS
        v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An"], node_cpds) # v += self._dynamic_node("StaLicht_0", "disc", ["Aus", "An", "Never"], node_cpds)
        v += self._dynamic_node("StgLicht_0", "disc", ["Aus", "An"], node_cpds)

        # More nodes
        v += self._dynamic_node("BedLicht_1", "disc", ["Aus_bed", "An_bed"], node_cpds)
        v += self._dynamic_node("StaLicht_1", "disc", ["Aus", "An"], node_cpds)
        v += self._dynamic_node("StgLicht_1", "disc", ["Aus", "An"], node_cpds)

        # Succeeding
        v += self._dynamic_node("BedLicht_2", "disc", ["Aus_bed", "An_bed"], node_cpds)
        v += self._dynamic_node("StaLicht_2", "disc", ["Aus", "An"], node_cpds)
        v += self._dynamic_node("StgLicht_2", "disc", ["Aus", "An"], node_cpds)

        # Define Edges
        e = []
        e += self._self_dependencies(v, ["BedLicht", "StaLicht", "StgLicht"])
        e += [["BedLicht_1", "StgLicht_1"]]  # Bedienung passiert immer vor Steuerung
        e += [["StgLicht_1", "StaLicht_1"]] # Steuerung passiert immer vor Status


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
        tscbn.draw("ext")

        # set some probabilities that I am sure about
        tscbn.Vdata["StgLicht_0"]["cprob"] = np.array([0.5, 0.5])
        tscbn.Vdata["BedLicht_0"]["cprob"] = np.array([0.5, 0.5])
        tscbn.Vdata["StaLicht_0"]["cprob"] = np.array([0.5, 0.5])

        # wenn StgLicht1 auf an geht geht auch StaLicht1 auf ein
        tscbn.Vdata["StgLicht_1"]["cprob"]["['Aus', 'Aus_bed']"] = np.array([0.8, 0.2])
        tscbn.Vdata["StgLicht_1"]["cprob"]["['Aus', 'An_bed']"] = np.array([0.2, 0.8])  # Halbfalsch
        tscbn.Vdata["StgLicht_1"]["cprob"]["['An', 'Aus_bed']"] = np.array([0.8, 0.2])  # Halbfalsch
        tscbn.Vdata["StgLicht_1"]["cprob"]["['An', 'An_bed']"] = np.array( [0.2, 0.8])  # Muss an bleiben - weil ich würde hier das event An sehen - es würde auftreten -
        # das würde man in dem Intervall nicht sehen!!!!!
        # ABER: es würde ja StaLicht beeinflussen - weil es ja tatsächlich passiert ist
        # d.h. zum Samplen muss ich das so machen
        # Never würde hier Fehler bedeuten - also BedLicht ging an aber StgLicht hat Nie reagiert somit hat StaLicht nie reagiert
        tscbn.Vdata["StgLicht_2"]["cprob"]["['Aus']"] = np.array([0.8, 0.2])  # FALSCH NOCH
        tscbn.Vdata["StgLicht_2"]["cprob"]["['An']"] = np.array([0.2, 0.8])  # FALSCH NOCH

        tscbn.Vdata["BedLicht_1"]["cprob"]["['Aus_bed']"] = np.array([0.1, 0.9])  #
        tscbn.Vdata["BedLicht_1"]["cprob"]["['An_bed']"] = np.array([0.9, 0.1])

        tscbn.Vdata["BedLicht_2"]["cprob"]["['Aus_bed']"] = np.array([0.9, 0.1])
        tscbn.Vdata["BedLicht_2"]["cprob"]["['An_bed']"] = np.array([0.1, 0.9])

        tscbn.Vdata["StaLicht_1"]["cprob"]["['Aus', 'Aus']"] = np.array([0.8, 0.2])
        tscbn.Vdata["StaLicht_1"]["cprob"]["['Aus', 'An']"] = np.array([0.2, 0.8])  # FALSCH NOCH
        tscbn.Vdata["StaLicht_1"]["cprob"]["['An', 'Aus']"] = np.array([0.8, 0.2])  # FALSCH NOCH
        tscbn.Vdata["StaLicht_1"]["cprob"]["['An', 'An']"] = np.array([0.2, 0.8])

        tscbn.Vdata["StaLicht_2"]["cprob"]["['Aus']"] = np.array([0.8, 0.2])  # hier macht never sinn - weil ich habe keinen kausalen Einfluss der meinen Wert auf ein Signal setzen würde
        # stattdessen bleibe ich Aus einfach weil nix passiert ist und ich passiere somit auch nicht
        # Frage? Warum bin ich dann überhaupt da - weil kann sein das ich mit 0.9 Never bin aber mit 0.1 Aus weil ich
        # mich spontan gewechselt habe
        tscbn.Vdata["StaLicht_2"]["cprob"]["['An']"] = np.array([0.2, 0.8])

        return tscbn

    def _add_temporal_basics(self, dL_mean, dL_var, node_cpds, v, e, defaults):
        ''' basic part that is same for all cases

        Note to myself: Pro Discrete Parent Combination - habe eigene Mean, und habe eigene Variance
                        Pro Continuous Parent - habe keine distributions - weil sich einfach kombinieren lässt de Sach
        Type definition:  http://pythonhosted.org/libpgm/CPDtypes.html
            If i bim continuous:
                - Wenn ich nur continuous parents hob, oder goa koane parents - then type: lg
                - Wenn ich discrete parents hob oder discrete und continuous - type: lgandd

            If i bim discrete
                - if I have no parents or discrete parents - type is: discrete
                - if I have continuous parents - type is: "to be implemented" - do it yourself bro
        '''

        # 1. Require initial node, per temporal variable to be at t=0

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

    def _add_temporal_basics_dump(self, dL_mean, dL_var, node_cpds, v, e):
        ''' basic part that is same for all cases

        Note to myself: Pro Discrete Parent Combination - habe eigene Mean, und habe eigene Variance
                        Pro Continuous Parent - habe keine distributions - weil sich einfach kombinieren lässt de Sach
        Type definition:  http://pythonhosted.org/libpgm/CPDtypes.html
            If i bim continuous:
                - Wenn ich nur continuous parents hob, oder goa koane parents - then type: lg
                - Wenn ich discrete parents hob oder discrete und continuous - type: lgandd

            If i bim discrete
                - if I have no parents or discrete parents - type is: discrete
                - if I have continuous parents - type is: "to be implemented" - do it yourself bro
        '''

        # 1. Require initial node, per temporal variable to be at t=0

        # 1. tp and dL information needs to be given - is distributed in any case
        #    currently same for all nodes - in reality distinct
        for vert in v:
            node_cpds[vert]["dL_mean"] = dL_mean
            node_cpds[vert]["dL_var"] = dL_var

        # Generate Network from this information
        skel = GraphSkeleton()
        skel.V = v
        skel.E = e
        try:
            skel.toporder()
        except:
            print("Warning -> Graph has cycles -> may be irrelevant depending on use-case")

        return skel, node_cpds
