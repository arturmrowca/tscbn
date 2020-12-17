import copy
import math

from numpy import random
from pgmpy.estimators.base import BaseEstimator

from _include.evaluation.base_evaluator import BaseEvaluator
from sklearn.metrics import mean_squared_error
import numpy as np

class StructureEvaluator(BaseEvaluator):
    '''
    Based on given results this class is returns an evaluation of the
    model structure
    '''

    def __init__(self, append_csv):
        '''
        Possible Metrics are:
            - brier-score: Brier-score which evaluates the
            - struct-sim: Structural Similarity
            - add-edges: number of additional edges between result and reference model
            - del-edges: number of deleted edges between result and reference model

            - num-edges: total number of edges
            - num-nodes: totoal number of nodes
            - num-states: total number of states per node
            - num-cpds: total number of parameter entries
            - temp-rmse: drift of RMSE from interval changes - depending on variance when drawn and tolerance when resolution chosen
        '''
        super(StructureEvaluator, self).__init__(append_csv)
        self._specifications = None

    def evaluate(self, model_dict, reference=None, specifications=None, additional_infos=None):
        '''
        Main method used for evaluation
        '''
        # Result dictionary
        self._specifications = specifications
        eval_results = {}

        for model_name in model_dict:

            # Initialize Methods
            model = model_dict[model_name]
            if model_name not in eval_results: eval_results[model_name] = {}

            # Perform evaluation
            for metric in self._metrics:
                if additional_infos is None:
                    eval_results[model_name][metric] = self._compute_metric(model, reference, metric, None)
                else:
                    eval_results[model_name][metric] = self._compute_metric(model, reference, metric, additional_infos[model_name])

        # Return results
        self._last_eval_results = eval_results
        return eval_results

    def _readable_metric(self, metric):
        '''
        Translates a metric key to a readable version
        '''
        if metric == "num-edges": return "Number of edges"
        if metric == "num-nodes": return "Number of nodes"
        if metric == "num-states": return "Number of states"
        if metric == "num-cpds": return "Number CPD Entry" # ENTRIES
        if metric == "temp-rmse": return "Temporal RMSE"
        if metric == "num-add-edges": return "Number of additional edges"
        if metric == "num-add-edges-skel": return "Number of additional edges (skel)"
        if metric == "add-edges": return "Additional edges"
        if metric == "add-edges-skel": return "Additional edges (skel)"
        if metric == "num-del-edges": return "Number of missing edges"
        if metric == "num-del-edges-skel": return "Number of missing edges (skel)"
        if metric == "del-edges": return "Missing edges"
        if metric == "del-edges-skel": return "Missing edges (skel)"
        if metric == "shd": return "Structural Hamming distance"
        if metric == "shd-skel": return "Structural Hamming distance (skel)"
        if metric == "add-nodes": return "Additional nodes"
        if metric == "del-nodes": return "Missing nodes"
        if metric == "num-add-nodes": return "Number of additional nodes"
        if metric == "num-del-nodes": return "Number of missing nodes"
        if metric == "kld": return "Kullback-Leibler divergence"
        if metric == "execution-time": return "Execution time"
        if metric == "temp_threshold": return "Temporal threshold"
        if metric == "psi-execution-time": return "Parent set identification xecution time"
        if metric == "so-execution-time": return "Structure optimization execution time"
        return metric

    def _compute_metric(self, model, reference, metric, additional_infos=None):
        '''
        Given the resulting model this method computes the metrics specified
        '''
        if metric == "num-edges":
            return len(model.E)
        if metric == "num-nodes":
            return len(model.V)
        if metric == "num-states":
            return self._compute_state_number(model)
        if metric == "num-cpds":
            return self._compute_cpds_number(model)
        if metric == "temp-rmse":
            return self._compute_rmse(model)
        if metric == "num-add-edges":
            return self._compute_num_add_edges(model, reference)
        if metric == "num-add-edges-skel":
            return self._compute_num_add_edges_skel(model, reference)
        if metric == "num-del-edges":
            return self._compute_num_del_edges(model, reference)
        if metric == "num-del-edges-skel":
            return self._compute_num_del_edges_skel(model, reference)
        if metric == "add-edges":
            return self._compute_add_edges(model, reference)
        if metric == "del-edges":
            return self._compute_del_edges(model, reference)
        if metric == "add-edges-skel":
            return self._compute_add_edges_skel(model, reference)
        if metric == "del-edges-skel":
            return self._compute_del_edges_skel(model, reference)
        if metric == "shd":
            return self._compute_shd(model, reference)
        if metric == "shd-skel":
            return self._compute_shd_skel(model, reference)
        if metric == "add-nodes":
            return self._compute_add_nodes(model, reference)
        if metric == "del-nodes":
            return self._compute_del_nodes(model, reference)
        if metric == "num-add-nodes":
            return self._compute_num_add_nodes(model, reference)
        if metric == "num-del-nodes":
            return self._compute_num_del_nodes(model, reference)
        if metric == "kld":
            try:
                return self._compute_kld(model, reference, additional_infos['data'])
            except:
                return 0
        if metric == "execution-time":
            return additional_infos['execution_time']
        if metric == "psi-execution-time" and 'psi_execution_time' in additional_infos:
            return additional_infos['psi_execution_time']
        if metric == "so-execution-time" and 'so_execution_time' in additional_infos:
            return additional_infos['so_execution_time']
        if metric == "temp_threshold":
            return additional_infos['temp_threshold']

    def _compute_cpds_number(self, model):
        try:
            cnt = model.alldata["eval_cpd_entries_count"]
            return cnt
        except:
            pass

        cnt = 0
        for node_name in model.Vdata:
            if not str.startswith(node_name, "dL_"):
                if not isinstance(model.Vdata[node_name]["cprob"], dict):
                    cprob_len = 1
                else:
                    cprob_len = len(model.Vdata[node_name]["cprob"])

                # Anzahl states mal len(cprob)
                cnt += len(model.Vdata[node_name]["vals"]) * cprob_len

            else:
                cnt += 2 # assume two parameters per dL node

        return cnt

    def _compute_state_number(self, model):

        try:
            # means CTBN
            model.alldata["eval_cpd_entries_count"]
            return 0
        except:
            cnt = 0
            for node_name in model.Vdata:
                if not str.startswith(node_name, "dL_"):
                    cnt += len(model.Vdata[node_name]["vals"])

            return cnt


    def _compute_rmse(self, model):
        #print("Computing RMSE (independent of parameter estimation!)...")
        assert(isinstance(self._specifications, dict)), "Specification needs to be a dicitonary in evaluate - else remove temp-rmse from the evaluationen metrics"



        # convert sequences if needed
        if model.__class__.__name__ == "DBNDiscrete":
            '''
            quick calculation
            model_start_time + x * res < last_gap
            model_start_time + (x+1) * res > last_gap
            
            -> solved for x:  x = - ((2*start_t + res - 2*last_gap_abs)/2*res) 
            '''
            errors = []
            real_val =[]
            for obj_index in range(len(self._specifications["temp_gap_between_objects"])):

                t_abs = 0
                st = model.start_time
                res = model.resolution
                obj = self._specifications["object_names"][obj_index]

                for interval_gap in self._specifications["temp_gap_between_objects"][obj_index]:
                    t_abs += interval_gap

                    x = int(- ((2*st + res - 2*t_abs)/(2*res)))

                    error_idx = np.argmin([np.abs(t_abs-(st + res * x)), np.abs(t_abs-(st + res * (x+1)))])
                    if error_idx == 0: error_val = st + res * x
                    else: error_val = st + res * (x+1)

                    errors += [error_val]
                    real_val += [t_abs]

            rmse = math.sqrt(mean_squared_error(real_val, errors))  # over all state changes
            return rmse

        if model.__class__.__name__ == "TSCBN":
            self.rmse_tscb_variance = self._specifications["temporal_variance"]  # je nach Variance
            self.rmse_mean_range = 0.00001

            # per node check drift

            # go through all parents
            # - if parents given estimate child assuming given mean and variance
            # - if state is "never" - remove it and use "long path" with according mean and variance

            initial_set = copy.deepcopy([node for node in model.V if (
                not model.Vdata[node]["parents"] or model.Vdata[node]["parents"] is None)])

            i = 0
            node_set = initial_set
            done = []
            t_abs = {}
            gather = []
            while node_set:

                # Get next parent
                node_set = list(set(node_set))
                if i >= len(node_set): i = 0
                p = node_set[i]
                i += 1

                if not (model.Vdata[p]["parents"] is None or not model.Vdata[p]["parents"]):
                    mean = random.random() * self.rmse_mean_range
                    var = self.rmse_tscb_variance
                    drift = np.abs(np.random.normal(mean, np.sqrt(var), 1)[0])
                    gather += [drift]

                    #rmse = math.sqrt(mean_squared_error([0], [drift]))# bzw. besser sammeln über alle Knoten und dann gebündelt vergleichen

                    #rmses += [rmse]
                else:
                    t_abs[p] = 0.0

                # add children
                node_set += [c for c in model.Vdata[p]["children"] if
                             not c in done and not str.startswith(c, "dL_")]

                # drop parent
                node_set.remove(p)
                done += [p]
            rmse = math.sqrt(mean_squared_error([0]*len(gather), gather)) # over all state changes

        # compute average

        return rmse

    def _compute_add_edges(self, model, reference):
        add_edges = []
        for edge in model.E:
            if edge not in reference.E and not str.startswith(edge[1], "dL_"):
                add_edges.append(edge)
            pass
        return add_edges

    def _compute_del_edges(self, model, reference):
        del_edges = []
        for edge in reference.E:
            if edge not in model.E and not str.startswith(edge[1], "dL_"):
                del_edges.append(edge)
            pass
        return del_edges

    def _compute_add_edges_skel(self, model, reference):
        add_edges = []
        for i, j in model.E:
            if [i, j] not in reference.E and [j, i] not in reference.E and not str.startswith(j, "dL_"):
                add_edges.append([i, j])
            pass
        return add_edges

    def _compute_del_edges_skel(self, model, reference):
        del_edges = []
        for i, j in reference.E:
            if [i, j] not in model.E and [j, i] not in model.E and not str.startswith(j, "dL_"):
                del_edges.append([i, j])
            pass
        return del_edges

    def _compute_num_del_edges(self, model, reference):
        num_del_edges = 0
        for edge in reference.E:
            if edge not in model.E and not str.startswith(edge[1], "dL_"):
                num_del_edges += 1
            pass
        return num_del_edges

    def _compute_num_add_edges(self, model, reference):
        num_add_edges = 0
        for edge in model.E:
            if edge not in reference.E and not str.startswith(edge[1], "dL_"):
                num_add_edges += 1
            pass
        return num_add_edges

    def _compute_num_del_edges_skel(self, model, reference):
        num_del_edges = 0
        for i, j in reference.E:
            if [i, j] not in model.E and [j, i] not in model.E and not str.startswith(j, "dL_"):
                num_del_edges += 1
            pass
        return num_del_edges

    def _compute_num_add_edges_skel(self, model, reference):
        num_add_edges = 0
        for i, j in model.E:
            if [i, j] not in reference.E and [j, i] not in reference.E and not str.startswith(j, "dL_"):
                num_add_edges += 1
            pass
        return num_add_edges

    def _compute_shd(self, model, reference):
        return self._compute_num_add_edges(model, reference) + self._compute_num_del_edges(model, reference)

    def _compute_shd_skel(self, model, reference):
        return self._compute_num_add_edges_skel(model, reference) + self._compute_num_del_edges_skel(model, reference)

    def _compute_add_nodes(self, model, reference):
        add_nodes = []
        for node in model.V:
            if node not in reference.V:
                add_nodes.append(node)
            pass
        return add_nodes

    def _compute_del_nodes(self, model, reference):
        del_nodes = []
        for node in reference.V:
            if node not in model.V:
                del_nodes.append(node)
            pass
        return del_nodes

    def _compute_num_add_nodes(self, model, reference):
        num_add_nodes = 0
        for node in model.V:
            if node not in reference.V:
                num_add_nodes += 1
            pass
        return num_add_nodes

    def _compute_num_del_nodes(self, model, reference):
        num_del_nodes = 0
        for node in reference.V:
            if node not in model.V:
                num_del_nodes += 1
            pass
        return num_del_nodes

    # def _compute_kld(self, model, reference, data):  # empirical Kullback-Leibler divergence
    #     """
    #     Remark: computationally expensive, mathmatically not completely correct
    #     """
    #     if self._compute_num_add_nodes(model, reference) != 0 or \
    #             self._compute_num_del_nodes(model, reference) != 0 or data is None:
    #         # nodes in model and reference model have to be identical to compute KLD
    #         # data has to be passed to compute KLD
    #         return float('inf')
    #     model.skeleton.toporder()  # sort model nodes in topological order
    #     nodes = [node for node in model.skeleton.V if not str.startswith(node, "dL_")]
    #     node_values = [reference.Vdata[node]['vals'] for node in nodes]  # take values from reference model
    #     be = BaseEstimator(data, complete_samples_only=False)
    #     kld = 0
    #     for value_combination in product(*node_values):
    #         pt = 1  # probability in true network
    #         pl = 1  # probability in discovered network
    #         for value, node in zip(value_combination, nodes):  # calculate prob for each value in this combination
    #             # reference model
    #             cprob = reference.Vdata[node]['cprob']  # probability table
    #             value_index = reference.Vdata[node]['vals'].index(value)
    #             parents = reference.Vdata[node]['parents']  # condition set
    #             if not parents:
    #                 pt *= cprob[value_index]  # unconditioned probability for value
    #             else:
    #                 parents_state = str([value_combination[nodes.index(parent)] for parent in parents])
    #                 pt *= cprob[parents_state][value_index]
    #             pass
    #
    #             # model discovered by structure discoverer
    #             parents = model.skeleton.getparents(node)  # get parents from skeleton as Vdata is from reference
    #             counts = be.state_counts(node, parents)
    #             if len(parents) == 0:
    #                 parents_state = node
    #             elif len(parents) == 1:
    #                 parents_state = value_combination[nodes.index(parents[0])]
    #             else:
    #                 parents_state = tuple(value_combination[nodes.index(parent)] for parent in parents)
    #             conditional_sample_size = sum(counts[parents_state])
    #             sample_size = counts[parents_state][value]
    #             pl *= sample_size / conditional_sample_size
    #             if sample_size == 0:  # this value combination was never observed
    #                 break  # this contradicts the definition of KLD
    #             pass
    #         pass
    #         if pl > 0:
    #             kld += pt * math.log(pt/pl)
    #         pass
    #     return kld

    def _compute_kld(self, model, reference, data):
        """
        Empirical Kullback-Leibler divergence calculated using Mutual-Information-Test as described in
        # Tsamardinos, I. ; Brown, L. E. ; Aliferis, C. F.: The Max-Min Hill-Climbing Bayesian Network Structure
        # Learning Algorithm. In: Machine Learning Bd. 65(1), Springer, 2006, S. 31-78
        """
        if self._compute_num_add_nodes(model, reference) != 0 or \
                self._compute_num_del_nodes(model, reference) != 0 or data is None:
            # nodes in model and reference model have to be identical to compute KLD
            # data has to be passed to compute KLD
            return float('inf')
        be = BaseEstimator(data, complete_samples_only=False)
        kld = 0
        nodes = [node for node in model.skeleton.V if not str.startswith(node, "dL_")]
        for node in nodes:  # iterate over all nodes in learned model
            parents = model.skeleton.getparents(node)  # get parents of node
            if not parents:  # nodes without parents are not relevant
                continue
            counts = be.state_counts(node, parents)  # get counts
            # compute mutual information between node and its parents
            node_values = counts.index.values
            parents_values = counts.columns.values
            total_counts = counts.values.sum()
            for value in node_values:  # iterate over all values of node
                node_counts = counts.loc[value].sum()
                for parents_state in parents_values:  # iterate over all combinations of parent values
                    parents_counts = counts[parents_state].sum()
                    joint_counts = counts[parents_state][value].sum()
                    if node_counts == 0 or parents_counts == 0 or joint_counts == 0:
                        continue  # if one of the counts was zero then do not increase the KLD
                    p_node = node_counts / total_counts
                    p_parents = parents_counts / total_counts
                    p_joint = joint_counts / total_counts
                    kld += p_joint * math.log(p_joint / (p_node * p_parents), 2)
                pass
            pass
        return kld
