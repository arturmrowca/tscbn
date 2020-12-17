'''
Created on 23.12.2017

@author: q416435
'''
import copy
import numpy as np
from _include.estimator.em_algorithm_tscbn_estimator import EMAlgorithmParameterEstimator
from _include.evaluation.parameter_evaluator import ParameterEvaluator


class CSParameterEvaluator(ParameterEvaluator):

    def __init__(self, append_csv):
        super(CSParameterEvaluator, self).__init__(append_csv)

    def _compute_jpd_n_log(self, model, dummy, test_sequence = None):
        '''
        Sample aus interval und schaue wie wahrscheinlich das produziert wird
        '''
        print("Computing jpd and log-likelihood // temp-jpd and temp-log-likelihood...")

        # per sample
        a,b,c,d,=[],[],[],[]
        for test_seq in test_sequence:
            tscbn_sequence = self._get_sequence([test_seq], model, number_samples=10)

            aa,bb,cc,dd, ee =  super(CSParameterEvaluator, self)._compute_jpd_n_log(model, tscbn_sequence, test_sequence = None)
            a += [aa]
            b += [bb]
            c += [cc]
            d += [dd]

        return np.mean(a), np.mean(b), np.mean(c), np.mean(d), "N.A."

    def _get_sequence(self, sequences, model, number_samples = 1):
        ''' Sample sequences from the set '''
        #
        samples = []
        em = EMAlgorithmParameterEstimator()
        em.tbn = model
        per_seq_trees, per_seq_initial_states = em._extract_sample_trees(sequences)
        trees = per_seq_trees[0]
        initial_states = per_seq_initial_states[0]


        # sample from this
        initial_set = [n for n in model.nodes.keys() if model.Vdata[n]["parents"] == None]
        pars = {}

        for tz in range(number_samples):
            # initial states
            tscbn_sample = {} # RESULT
            for i in initial_states:
                tscbn_sample[i+"_0"] = initial_states[i][0]
                tscbn_sample["dL_" + i + "_0"] = 0.0


            for t in trees: trees[t].reset(initial_states)
            node_set = copy.deepcopy(initial_set)
            parents_set, set_values, i, current_sample_initial = [], {}, 0, []
            current_sample, sample_legid, t_abs, t_abs_end = [], True, {}, {}


            done = []
            while node_set:

                # 1. next node
                i, n = em._next_node(node_set, i)

                # 2. copy parent information - to omit parallel access
                if n not in pars:
                    par = {}
                    par["parents"] = copy.deepcopy(model.Vdata[n]["parents"])
                    par["dL_parents"] = copy.deepcopy(model.Vdata["dL_" + n]["parents"])
                    par["tbn_vals"] = copy.deepcopy(model.Vdata[n]["vals"])
                    par["children"] = copy.deepcopy(model.Vdata[n]["children"])
                    par["cprob"] = copy.deepcopy(model.Vdata[n]["cprob"])
                    pars[n] = par

                # 3. if initial states - draw it from there
                if n.split("_")[1] == "0":
                    # DRAW LEAF NODE INITIAL SAMPLE
                    val = initial_states[n.split("_")[0]][0]  # L().log.debug("%s - I return: %s " % (str(n), str(val)))
                    current_sample_initial.append([n, pars[n]["tbn_vals"].index(val)])  # info, info
                    set_values[n]=val
                    t_abs[n] = 0
                else:

                    # 4. if not initial states - draw conditioned on parents
                    # check if all parents given - else continue
                    if not set(pars[n]["parents"]).issubset(parents_set):
                        i += 1
                        continue

                    # get conditions
                    cond = [set_values[k] for k in pars[n]["parents"]]

                    # DRAW AND STORE NEXT SYMBOL
                    parent_starts = [[em._is_never(k, set_values), t_abs[k]] for k in pars[n]["parents"]]
                    # parent_ends = [ [self._is_never(k, set_values), t_abs_end[k]] for k in pars[n]["parents"]]
                    val = trees[n.split("_")[0]].get_next_symbol(parent_starts, em._parent_outcome(n, set_values),
                                                                 cond)
                    if val is None:
                        print("Sample not legit")
                        break
                    tscbn_sample[n] = val[0]

                    set_values[n] = val[0]
                    t_abs[n] = val[1]
                    t_abs_end[n] = val[2]

                    # IF DRAWN SAMPLE LEGIT RECORD IT
                    current_sample.append([n, str(cond), pars[n]["tbn_vals"].index(val[0])])

                    # RECORD DELTA T DISTRIBUTION
                    cond_dL = [set_values[k] for k in
                               pars[n]["dL_parents"]]  # [set_values[k] for k in modelVdata["dL_" + n]["parents"]]
                    if "dL_" + n not in tscbn_sample:
                        tscbn_sample["dL_" + n] = t_abs[n] - max([t_abs[k] for k in pars[n]["parents"]])

                # GET NEXT NODES
                parents_set.append(n)
                node_set.remove(n)
                done += [n]
                node_set += [o for o in pars[n]["children"] if not o in done and not str.startswith(o, "dL_")]
                node_set = list(set(node_set))

            samples.append(tscbn_sample)


        return samples
