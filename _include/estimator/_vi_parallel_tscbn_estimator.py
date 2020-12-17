#!/usr/bin/env python 
# -*- coding: utf-8 -*-
from itertools import product
import time
from _include.estimator.parameter_estimator import ParameterEstimator
from _include.toolkit import parallelize_stuff
import numpy as np
import copy
from general.log import Log as L
from scipy.integrate import quad

def iter_vi(vi_est):
    vi_est.perform_iteration()
    return vi_est

class VIElement(object):
    '''
    This element has a group of sequences it iterates over and keeps its local
    distributions
    '''

    def __init__(self, seq_group, q_latent, q_latent_t, tbn, rel_outcome_combos):
        # all
        self.q_latent = q_latent
        self.q_latent_t = q_latent_t
        self.tbn = tbn

        # all sequences this element iterates over
        self.observations = seq_group
        self.temporal_condition = True
        self.include_temporal_estimate_to_cat = True
        self.relevant_outcome_combos = rel_outcome_combos

        # helpers for improved performance
        self._direct_q_nodes = dict()
        self._number_actual = dict()
        self._relevant_qs = dict()
        self._store_temporal_approximation = dict()

    def dt_integrand(self, t, mu, var):
        '''
        Assuming a Gaussian per temporal node this integrand has to be computed
        :param t: delta time as variable to integrate over
        :param mu: constant mu
        :param var: consant var
        :return: integral over a*b as shown below
        '''
        a = (-(t*t)) + 2*mu*t - mu*mu
        b = (1/np.sqrt(2*np.pi*var)) * np.exp(-( ((t-mu)*(t-mu))/(2*var)))

        return a*b

    def perform_iteration(self):
        self.seq_ids = sorted(list(self.q_latent.keys()))
        i = -1
        for seq_id in self.seq_ids:
            i += 1
            self._variational_iteration(self.observations[i], seq_id)

        # summarize parameters
        self._prepare_cpds()
        self._prepare_times()

    def _prepare_cpds(self):
        self.resulting = dict()


        # 1. Iterate nodes
        for node in self.tbn.V:
            if str.startswith(node, "dL_"): continue
            self.resulting[node] = dict()

            new_cprob = dict()

            # 2. No parents given: estimate is average of all local estimates
            if self.tbn.Vdata[node]["parents"] is None:
                self.resulting[node] = np.mean(np.array([n[node] for n in self.q_latent.values()]), axis=0)
                continue

            # 3. Iterate outcomes
            for k in self.seq_ids:
                for cond in self.tbn.Vdata[node]["cprob"]:
                    if not cond in new_cprob: new_cprob[cond] = np.zeros(np.array(self.tbn.Vdata[node]["vals"]).shape)

                    # 5. Compute p(node|parents) = q(node)*q(parent_1)*q(parent_2)...
                    p = copy.deepcopy(self.q_latent[k][node]) # node estimate
                    i = -1
                    for pa in self.tbn.Vdata[node]["parents"]: # parent estimates
                        i += 1
                        p *= self.q_latent[k][pa][self.tbn.Vdata[pa]["vals"].index(eval(cond)[i])]
                    new_cprob[cond] += p


            # 7. Set updated distributions
            self.resulting[node] = new_cprob

    def _prepare_times(self):
        '''
        Based on latent variable estimates q(z) compute new CPDs and set the according parameters
        p(A|B,C) = q(A)*q(B)*q(C)
        mu = weighted mean of q_all(outcomes) * mu over all sequences = q1*m1 + q2*m2 + q3*m3 / q1 + q2 + q3
        :param sequences: List of sequences in format: listeelement is dict. of key: TV name value: [state, start_time_of_state, end_time_of_state]
        '''
        self.resulting_mu = dict()
        self.resulting_mu_q = dict()

        # 1. Iterate temporal nodes and update them (e.g. p(dT_A|B,C))
        kk = 0
        for node in self.tbn.V:

            kk += 1
            #print("Update t %s / %s" % (str(kk), str(len(self.tbn.V))))
            if not str.startswith(node, "dL_"): continue
            self.resulting_mu[node] = dict()
            self.resulting_mu_q[node] = dict()



            # 2. Iterate all outcome combinations
            for cond in self.tbn.Vdata[node]["hybcprob"]:
                if not eval(cond): continue

                # Initialize Values
                sum_mu_times_q = 0.0
                sum_q = 0.0

                # 3. Iterate local estimates to compute global parameters

                for k in self.seq_ids:

                    # Compute q(parent_1=this_outcome)*q(parent_2=this_outcome)*...
                    i, q = -1, 1.0
                    for pa in self.tbn.Vdata[node]["parents"]:
                        i += 1
                        q *= self.q_latent[k][pa][self.tbn.Vdata[pa]["vals"].index(eval(cond)[i])] # this seq: q(pa) * mu(pa)

                    # Add outcomes to sum
                    mu = self.q_latent_t[k][node]["mean_base"]
                    sum_mu_times_q += mu * q
                    sum_q += q

                self.resulting_mu[node][cond] = sum_mu_times_q
                self.resulting_mu_q[node][cond] = sum_q

    def _variational_iteration(self, observation, seq_id):
        '''
        Per sequence a set of local estimates is found in this step and iterated
        :param observation: current observed sequence
        :param seq_id: Identifier of current sequence
        '''


        # 1. Iterate over all nodes and update them
        for node in self.tbn.V:

            retry, temporal_cond = True, self.temporal_condition
            oo = 0
            while retry:

                # 2. Compute update for continuous temporal node
                if str.startswith(node, "dL_"):
                    self._update_latent_time(node, seq_id, observation)

                    retry = False
                    continue

                else:
                    retry, temporal_cond = self._update_latent_node(seq_id, observation, node, retry, temporal_cond)

                if oo > 1:
                    print("retry %s" % str(oo))
                oo += 1

    def _temporal_approximation(self, outcome, seq_id, iter, observ):
        '''
        When streching a sequence (e.g. AB) to a longer size (e.g. AABB) each node needs to have
        assigned a start and end time. This method computes a linear interpolation per outcome as follows
            Starttimes
            - Nevers between state changes are linearly interpolated e.g. A=1, B=2 for AAAB gives A=1, A=1.33,A=1.66,B=2.0
            - If the last short sequence element is followed by further elements the distance between this element
              and its preceeding element is used for further steps e.g. seen A=1, B=2 then, AABBB would be A=1,A=1.5, B=2,B=2.5,B=3
              -> This upstep time is determined as
                  - last_time minus time before last seen time
                  - if not available: total time of sequence divided by number of nodes
            Endtimes:
            - Endtimes are always the start times of the preceeding element
            - if the last element was extended the end time is the extended value + the equidistance to the preceding element

        :param outcome: Outcome to be temporally approximated i.e. a dictionary of node_name:node_value
        :param seq_id: Id of current sequence - used to store the result to avoid recomputation
        :param iter: Id of current iteration - used to store the result to avoid recomputation
        :param observ: one sequence containing key: TV name and value: list of [state, start_time, end_time]
        :return outcome: Corrected outcome with temporally approximated times
        '''

        # 1. Avoid recomputation of information by storing in dictionary
        if seq_id in self._store_temporal_approximation and iter in self._store_temporal_approximation[seq_id]:
            return self._store_temporal_approximation[seq_id][iter]

        # 2. Determine temporal variables
        temporal_vars = set(list([n.split("_")[0] for n in outcome if not str.startswith(n, "dL_")]))

        # 3. Approximate per TV
        for tv in temporal_vars:

            # 4. Initialize times of long sequences e.g. AABB [A_time, A_time, B_time, B_time]
            all_times = [outcome[k][1] for k in outcome if k.split("_")[0]== tv and not str.startswith(k, "dL_")]
            times = sorted(list(set(all_times))) # times of short sequence i.e. e.g. AB
            all_times = np.array(all_times)

            # 5. Determine upstep size
            try: upstep_size = times[-1] - times[-2]
            except:
                end_time = sorted(list(set([outcome[k][2] for k in outcome if k.split("_")[0]==tv and not str.startswith(k, "dL_")])))[-1]
                upstep_size = end_time/float(len(all_times)+1)
            if len(all_times) == len(times): continue

            # 6. Linear Interpolation
            for t in times:
                if times[-1] == t:
                    kko = all_times[all_times==t]
                    l_steps = len(kko)
                    all_times[all_times==t] = np.linspace(t, t+(l_steps)*upstep_size, len(kko)+1)[:len(kko)]
                else:
                    kko = all_times[all_times==t]
                    all_times[all_times==t] = np.linspace(t, all_times[all_times>t][0], len(kko)+1)[:len(kko)]

            # 7. Assign result to outcomes and compute end times
            for q in range(len(all_times)):
                try: last_time = all_times[q+1]
                except: last_time = observ[tv][-1][-1]
                outcome[tv+"_"+str(q)] = (outcome[tv+"_"+str(q)][0], all_times[q], last_time)
                if outcome[tv+"_"+str(q)][1] > outcome[tv+"_"+str(q)][2]:
                    outcome[tv+"_"+str(q)] = (outcome[tv+"_"+str(q)][0], outcome[tv+"_"+str(q)][1], outcome[tv+"_"+str(q)][1]  + upstep_size)

        # 8. Avoid recomputation of information by storing in dictionary
        if not seq_id in self._store_temporal_approximation: self._store_temporal_approximation[seq_id] = dict()
        if not iter in self._store_temporal_approximation[seq_id]: self._store_temporal_approximation[seq_id][iter] = outcome

        return outcome

    def _update_latent_time(self, node, seq_id, observation):
        '''
        For the current sequence the latent local estimate for the temporal node dt is q(dt; mu, sigma) and is assumed Gaussian and thus, is governed
        by mu and sigma, sigma is assumed constant in our case, mu is recomputed from the observed data
        :param node: name of temporal node to be updated
        :param seq_id: id of sequence to be updated
        :param observation: one sequence containing key: TV name and value: list of [state, start_time, end_time]
        :return:
        '''

        # 1. Two strategies: 0: repeat last time, 1: linear interpolation
        strategy = 1
        mu_numerator, mu_denominator, iter = 0.0, 0.0, 0

        # 2. iterate all combinations of outcomes /for me as node only my intranodes and inter nodes are relevant not all!
        for outcome in self.relevant_outcome_combos[seq_id]:
            iter += 1

            # 3. per output combination e.g. ABBA BBC assign all absolute times to nodes
            if strategy == 1: outcome = self._temporal_approximation(outcome, seq_id, iter, observation)

            # 4. Compute dT = myTime - earliestParentTime
            parent_times = [outcome[p][1] for p in self.tbn.Vdata[node]["parents"] if "dL_"+p != node]
            my_outcome = outcome[node[3:]][0]
            my_time = outcome[node[3:]][1]
            if parent_times: max_time = np.max(parent_times)
            else: max_time = 0.0

            # 5. Current Mue
            mu = my_time - max_time
            var = 0.01

            # 6. Compute q_all = q1(this_outcome) q2(this_outocme)
            q_all = self.q_latent[seq_id][node[3:]][self.tbn.Vdata[node[3:]]["vals"].index(my_outcome)]
            q_all *= np.prod([self.q_latent[seq_id][k][self.tbn.Vdata[k]["vals"].index(outcome[k][0])] for k in self.tbn.Vdata[node]["parents"]])

            # 7. Compute update for mu
            mu_numerator = (mu/var) * q_all
            mu_denominator = (1/var) * q_all

        # 8. Update mu
        new_est = mu_numerator / mu_denominator
        if new_est<0: new_est = 0.0
        self.q_latent_t[seq_id][node]["mean_base"] = new_est

    def _update_latent_node(self, seq_id, observation, node, retry, temporal_cond):

        # if observation of this node not latent - q(z) is eindeutig - egal was parents sind!

         # no update needed - only one observation possible
        if seq_id in self._direct_q_nodes and node in self._direct_q_nodes[seq_id]:
            retry = False
            return retry, temporal_cond
        else:
            tv = node.split("_")[0]
            number_observed = len(observation[node.split("_")[0]])
            if tv not in self._number_actual:
                self._number_actual[tv] = len([t for t in self.tbn.V if t.split("_")[0] == tv])
            if self._number_actual[tv] == number_observed:
                self.q_latent[seq_id][node] = np.zeros(self.q_latent[seq_id][node].shape)

                # observation an stelle int(node.split("_")[-1])
                node_id = int(node.split("_")[-1])
                node_val = observation[tv][node_id][0]
                idx_one = self.tbn.Vdata[node]["vals"].index(node_val)
                self.q_latent[seq_id][node][idx_one] = 1.0
                retry = False
                if not seq_id in self._direct_q_nodes:
                    self._direct_q_nodes[seq_id] = dict()
                if not node in self._direct_q_nodes[seq_id]:
                    self._direct_q_nodes[seq_id][node] = True
                return retry, temporal_cond


        # 1. Determine all valid outcomes
        #self._log(5, (node, seq_id, observation))
        #self._determine_relevant(seq_id, node, observation)

        # 2. store relevant qs to avoid recomputation
        q_compute, lst, gen = False, [], False
        if not seq_id in self._relevant_qs: self._relevant_qs[seq_id] = dict()
        if not node in self._relevant_qs[seq_id]: self._relevant_qs[seq_id][node], q_compute = [], True
        if isinstance(self.relevant_outcome_combos[seq_id], list): gen = True

        # 3. Compute expectation: Iterate over all allowed observations
        result_expectation = np.zeros(len(self.tbn.Vdata[node]["vals"]))
        for outcome in self.relevant_outcome_combos[seq_id]:

            # 4. Store for next iteration
            if not gen: lst += [outcome]
            #L().log.debug("Outcome: %s" % str(outcome))

            # 5. p(node|parents_node_alle_versionen)
            current_outcome_idx = self.tbn.Vdata[node]["vals"].index(outcome[node][0])
            if not self.tbn.Vdata[node]["parents"] is None:

                # 6. Check temporal conditions
                parent_start_times = np.array([outcome[p][1] for p in self.tbn.Vdata[node]["parents"]])
                parent_end_times = np.array([outcome[p][2] for p in self.tbn.Vdata[node]["parents"]])
                my_time = outcome[node][1]
                my_end_time = outcome[node][2]
                if temporal_cond and not self._temporal_conditions_given(node, outcome, parent_start_times, parent_end_times, self.tbn.Vdata[node]["parents"], my_time, my_end_time):
                    continue

                # 7. Determine p(node|parents_this_outcome) = p_me_parents
                parent_outcomes = str([outcome[p][0] for p in self.tbn.Vdata[node]["parents"]])
                p_me_parents = self.tbn.Vdata[node]["cprob"][parent_outcomes][current_outcome_idx]

                # 8 . compute q for all parents
                if q_compute: self._relevant_qs[seq_id][node] += self.tbn.Vdata[node]["parents"]

            else:
                # 9. If no parents given
                p_me_parents = self.tbn.Vdata[node]["cprob"][current_outcome_idx]

            ''' ------------ CATEGORICAL PART ------------ '''
            # 10. Compute p of child nodes for this outcome combo - CATEGORICAL PART
            # p(child_1|parents_und_node) p(child_2|parents_und_node) # all child and parent versions
            p_child_parents = 1.0
            for child in self.tbn.Vdata[node]["children"]:
                if str.startswith(child, "dL_"): continue
                child_outcome_idx = self.tbn.Vdata[child]["vals"].index(outcome[child][0])
                parent_outcomes = str([outcome[p][0] for p in self.tbn.Vdata[child]["parents"]])
                p_child_parents *= self.tbn.Vdata[child]["cprob"][parent_outcomes][child_outcome_idx]

                if q_compute: self._relevant_qs[seq_id][node] += self.tbn.Vdata[child]["parents"]
                if q_compute: self._relevant_qs[seq_id][node] += [child]

            # 11. Compute result of categorical part
            # compute q_total = q = q(z1)*q(z2)*...
            if q_compute:
                self._relevant_qs[seq_id][node] = list(set(self._relevant_qs[seq_id][node]))
                if node in self._relevant_qs[seq_id][node]: self._relevant_qs[seq_id][node].remove(node) # NOT q(me) else I miss the point of this!!!
            q_all = np.prod(np.array([self.q_latent[seq_id][p] [self.tbn.Vdata[p]["vals"].index(outcome[p][0])] for p in self._relevant_qs[seq_id][node]]))

            # 12. Add result to final expectation
            result_expectation[current_outcome_idx] += np.log(p_me_parents * p_child_parents) * q_all

            ''' ------------ TEMPORAL PART ------------ '''
            # 13. Compute p of child nodes for this outcome combo - CONTINUOUS PART
            # q(all_except_t) * (-(1/(2sg^2)) * integral( -(t-mu)^2 * q(t))dt
            if self.include_temporal_estimate_to_cat:
                mu = self.q_latent_t[seq_id]["dL_"+node]["mean_base"]
                var = self.q_latent_t[seq_id]["dL_"+node]["variance"]
                fst = -(1/2*var)
                integral = quad(self.dt_integrand, mu-5*var, mu+5*var, args=(mu, var))[0]
                second = fst*integral*q_all
                result_expectation[current_outcome_idx] += second # 4 mit time, 4 ohne time
        if not gen: self._relevant_outcome_combos[seq_id] = lst

        # 14. ALL OUTCOME COMBOS DONE: UPDATE q(node)
        result_expectation[result_expectation == 0.0] = -np.inf # 0 means not seen -> which means prob. of 0 -> i.e. -inf!
        result_expectation = np.nan_to_num(result_expectation)

        q_est = np.exp(result_expectation)

        # OVERFLOW - Number to big -> nan
        if np.sum(q_est) == 0:
            if not np.all(result_expectation == result_expectation[0]):
                result_expectation /= np.max(np.abs(result_expectation))
                q_est = np.exp(result_expectation)
            else:
                q_est = np.ones(q_est.shape)
        self.q_latent[seq_id][node] = q_est/ np.sum(q_est)

        # 15. If temporal condition failed
        #     -> recompute without temporal condition
        if np.isnan(self.q_latent[seq_id][node][0]):
            temporal_cond = False
            #L().log.debug("Require to retry without temporal aspect")
            print("Require to retry without temporal aspect %s and %s" % (str(result_expectation), str(q_est)))
        else:
            retry = False

        # 16. Logging
        #L().log.debug("-------------> Update Latent Variable %s at Seq %s" % (str(self._q_latent[seq_id][node]), str(seq_id)))
        #L().log.debug("\n")

        return retry, temporal_cond

    def _temporal_conditions_given(self, node, outcome, parent_start_times, parent_end_times, parents, my_start_time, my_end_time):
        ''' This method checks if the given outcome combination with given node with start time my_start_time and ending time my_end_time,
            and parent nodes with parent_start_times and parent_end_times are logical and therefore legid sequence outcomes
            The following conditions need to hold:
                - condition 1: wenn node==Never, dann muss Endzeit_node> all parents_end_zeiten
                - condition 2: wenn node==occurring, a parent_Never - this_parent_start_times < node_end_time
                - condition 3: wenn node==occ, parent== occ - parent_start_time<node_start_time
            :param node: Name of current node
            :param outcome: Values of the current outcome
            :param parent_start_times: List of start times of all parents
            :param parent_end_times: List of end times of all parents
            :param my_start_time: Start time of currents node outcome
            :param my_end_time: End time of currents node outcome
        '''

        # 1. Check current node is never
        my_prev_node = "_".join(node.split("_")[:-1]) +"_"+ str(int(node.split("_")[-1])-1)
        me_is_never =  outcome[my_prev_node] == outcome[node]

        # 2. Iterate parents
        for par in range(len(parent_end_times)):
            this_parent = parents[par]
            this_parent_end_time = parent_end_times[par]
            this_parent_start_time = parent_start_times[par]

             # 3. Check parent is never i.e. parent_of_parent == parent
            prev_prev_node = "_".join(this_parent.split("_")[:-1]) +"_"+ str(int(this_parent.split("_")[-1])-2)
            prev_node = "_".join(this_parent.split("_")[:-1]) +"_"+ str(int(this_parent.split("_")[-1])-1)
            try: par_is_never = outcome[prev_node] == outcome[prev_prev_node]
            except: par_is_never = False

            # Condition 1: if node==Never -> End_time_node> all parents_end_times
            if me_is_never:
                if my_end_time < this_parent_end_time:
                    return False
            else:
                # Condition 2: if node==occ., a parent_Never - this_parent_start_times < node_end_time
                if par_is_never and this_parent_start_time > my_end_time:
                    return False

                # Condition 3: if node==occ., parent== occ - parent_start_time<node_start_time
                if (not par_is_never) and this_parent_start_time > my_start_time:
                    return False
        return True


class VariationalInferenceParallelParameterEstimator(ParameterEstimator):
    '''
    Variational Inference Algorithm
    '''

    def __init__(self):
        super(VariationalInferenceParallelParameterEstimator, self).__init__()

        # public properties
        self.iteration_frequency = 1 # Number of Iterations
        self.temporal_condition = True # If False temporal condition is never checked
        self.include_temporal_estimate_to_cat = True # Continuous node included in computation of categorical
        self.sub_groups_seq_combos = 50 # batch size -> split data into this much batches

        # information
        self.sub_groups = 100
        self._show_runs = False
        self._first_iteration = True
        self._tree_node_number = dict()
        self._relevant_iter_nodes = dict()
        self._relevant_outcomes_nodes = dict()
        self._relevant_outcome_combos = dict()
        self._relevant_qs = dict()
        self._store_temporal_approximation = dict()
        self._number_actual = dict()
        self._direct_q_nodes = dict()

    def estimateParameter(self, sequences, model, debug = False, evaluator = False, reference = None ):
        '''
        Entry point for parameter estimation
        :param sequences: List of sequences in format: listeelement is dict. of key: TV name value: [state, start_time_of_state, end_time_of_state]
        :param model: string indicating the model used e.g. TSCBNStructureModel or DBNModel
        :param debug: boolean - True if debug mode is on
        :param evaluator: Parameter Evaluation Engine used to compute statistics on the fly (e.g. KL Divergence)
        :param reference: The reference model which contains the True model structure and parameters
        :return:
        '''
        self._evaluator = evaluator
        self._reference = reference

        if model == "TSCBNStructureModel" or model == "TSCBNSimpleStructureModel":
            return self._estimate_tscbn(sequences, debug)

    def _set_uniform_prior(self):
        '''
        This method sets an uniform prior at each node and parameter of the given TSCBN
        :return: Nothing
        '''

        # Iterate all nodes
        L().log.debug("Set unfiform priors: ")
        for n in self.tbn.Vdata:
            if str.startswith(n, "dL_"): continue

            # If thid node has parents
            if isinstance(self.tbn.Vdata[n]["cprob"], dict):
                for k in self.tbn.Vdata[n]["cprob"]:
                    self.tbn.Vdata[n]["cprob"][k]  = np.array([1.0 / float(len(self.tbn.Vdata[n]["cprob"][k]))]*len(self.tbn.Vdata[n]["cprob"][k]))
                    L().log.debug("%s | %s = %s" % (n, k, str(self.tbn.Vdata[n]["cprob"][k])))

            # No parents of this node
            else:
                self.tbn.Vdata[n]["cprob"] = np.array([1.0 / float(len(self.tbn.Vdata[n]["cprob"]))]*len(self.tbn.Vdata[n]["cprob"]))
                L().log.debug("%s = %s" % (n, str(self.tbn.Vdata[n]["cprob"])))

    def _drop_duplicate_list(self, seq):
        '''
        Takes a list and removes all identical subsequent elements in the list
        e.g. AAACCDDD will be ACD
        :param seq: List of states e.g. ["A", "B", "B"]
        :return: Short version of this list
        '''
        prev = None
        res = []
        for x in seq:
            if x != prev:
                res += [x]
            prev = x
        return res

    def _copy_q_latent_from_tree(self, sequence_length):
        '''
        Per sequence copy the tree of all possible latent variable estimates
        i.e. q(z) is an estimate of each node of each sequence!
        :param sequence_length: Length of the sequence thus, number of local q s to create
        :return _q_latent: Contains a dictionary of local estimates of all categorical variables in format keys: [sequence_id][node_name] and value: local distribution
        :return _q_latent_time: Contains a dictionary of local estimates of all continuous variables  in format keys: [sequence_id][node_name] and value: local distribution as mue and sigma
        '''

        # Initialize
        q_latent, q_latent_time = dict(), dict()

        for k in range(sequence_length):

            # Per sequence a node dictionary
            q_latent[k], q_latent_time[k] = dict(), dict()

            # Iterate all nodes
            for n in self.tbn.V:

                # Add distribution for temporal node
                if str.startswith(n, "dL_"):
                    q_latent_time[k][n] = copy.deepcopy(self.tbn.Vdata[n]["hybcprob"][list(self.tbn.Vdata[n]["hybcprob"].keys())[0]])
                    continue

                # Add distribution for categorical node (except if has no parents)
                try: q_latent[k][n] = copy.deepcopy(self.tbn.Vdata[n]["cprob"][list(self.tbn.Vdata[n]["cprob"].keys())[0]])
                except: q_latent[k][n] = copy.deepcopy(self.tbn.Vdata[n]["cprob"])

        return q_latent, q_latent_time

    def _log(self, idx, args=None):
        '''
        Contains the logging strings
        :param idx: Index of log to print
        :param args: Arguments for the log to print
        '''
        if idx == 0:
            L().log.info("------------------------------------------------------------> VI Iteration: %s ------------------------------------------------------------" % args)
        if idx == 1:
            L().log.debug("\n\n\n")
            L().log.debug("----------------------------")
            L().log.debug("Sequence: %s" % args)
            if int(args) % 25 == 0:
                L().log.info("Sequence: %s" % args)
            L().log.debug("----------------------------");L().log.debug("\n")
        if idx == 2:
            L().log.debug("---------------------------------------------------------------------------------------------------------------------------------------------------")
            L().log.debug("     Histogram Update")
            L().log.debug("---------------------------------------------------------------------------------------------------------------------------------------------------")
            self._log_cpds(normalize=False)
        if idx == 3:
            if self.tbn.show_plot_generated:
                self._visual.plot_histograms_from_bn(self.tbn, self.original_tbn)
            L().log.info( "------------------------------------------------------------> EM Finished ------------------------------------------------------------")
            self._log_cpds()

        if idx == 4:
            L().log.info("\n")
            L().log.info("Start VI Iterations")

        if idx == 5:
                L().log.debug("\n\n")
                L().log.debug("----------------------- UPDATE latent Node q(%s) o Seq: %s -----------------------" % (args[0], args[1]))
                L().log.debug("Observe: %s" % str(args[2]))

    def _print_kl_divergence(self):
        '''
        Prints the KL Divergence of the given reference model against the estimated parameters
        '''
        try:
            kl_div = self._evaluator._compute_kl_divergence(self.tbn, self._reference, print_it = False)

            if np.isnan(kl_div):
                self._evaluator._compute_kl_divergence(self.tbn, self._reference, print_it = False)

            L().log.info("--- Current KL Divergence: %s" % str(kl_div))
            print("--- Current KL Divergence: %s" % str(kl_div))
        except:
            L().log.error("No Evaluation for kl per iteration set")

    def _initialize_vi_iteration(self, vi_iteration):
        '''
        Initialize values for the VI iterations
        :param vi_iteration: index of current iteration
        :return start_time: Start time used to compute elapsed time
        :return k: Index of the sequence
        '''

        start_time = time.time()
        k = -1;self._log(0, str(vi_iteration+1))

        print("New Iteration %s" % (str(vi_iteration+1)))
        self._log_cpds(normalize=False)

        return start_time, k,

    def _update_sequence(self, k, sequence):
        '''
        Logs the state after iterating one sequence,
        :param k: Iteration index of previous sequence
        :return sequence:  Current sequence
        '''

        # Update Sequence index
        k += 1;self._log(1, str(k))

        # Logging
        L().log.debug(str(sequence))
        L().log.debug("----------------------------")

        return k

    def _recompute_global_parameters(self, iter_out, start_time):
        '''
        From local extimates compute new CPDs p(x|y,x) = average of sum over all sequnces q(x)*q(y)*q(z)
        :param sequences: List of sequenes
        :param start_time: Start time of current VI Iteration
        :return: 
        '''

        # Update estimates
        L().log.info("----> Recomputing categorical CPDs")
        self._update_cpd(iter_out)
        L().log.info("----> Recomputing continuous CPDs")
        self._update_times(iter_out)
        L().log.info("----> CPD Update done")

        # Print logging
        print("Elapsed Time: %s" % str(time.time() - start_time))
        self._print_kl_divergence()

    def iter_det_relevant(self, nodes, observations, seq_ids):
        res_dict = dict()
        k = -1
        for observation in observations:
            k += 1
            seq_id = seq_ids[k]
            res_dict[seq_id] = self._determine_relevant_light(seq_id, observation)
        return res_dict

    def _estimate_tscbn(self, sequences, debug):
        '''
        Estimates the parameters of a TSCBN given its input sequences
        For this distinguish
            1) Parameters: p(z|bla) == Ziel                 => stored in self.tbn.Vdata
            2) Estimation of z_i: q(z_i) one per sequence   => stored in self._q_latent[sequence_id] and _q_latent_t
        :param sequences: List of sequences in format: listeelement is dict. of key: TV name value: [state, start_time_of_state, end_time_of_state]
        :param debug: True if debug mode is active
        '''

        # 1. Set uniform prior and print KL Divergence
        self._log_reference_cpds()
        self._set_uniform_prior();self._log(4)
        self._print_kl_divergence()

        # 2. Init. local estimate per latent variable and sequence
        self._q_latent, self._q_latent_t = self._copy_q_latent_from_tree(len(sequences))

        print("Determine valid sequence combinations") #sub_groups_seq_combos = 50 # smaller batches needed for combo determination

        sub_groups = self.sub_groups_seq_combos
        prev = 0
        inputs = []
        nodes = copy.deepcopy(self.tbn.V)
        for i in range(0, len(sequences), sub_groups):
            if i == 0: continue
            rel_indices = list(range(prev, i))
            seq_group = sequences[prev:i]
            inputs += [[nodes, seq_group, rel_indices]]
            prev = i
        rel_indices = list(range(prev, len(sequences)))
        seq_group = sequences[prev:]
        inputs += [[nodes, seq_group, rel_indices]]
        outputs = parallelize_stuff(inputs, self.iter_det_relevant, simultaneous_processes = 30, print_all = True)
        self._relevant_outcome_combos = dict()
        for output in outputs:
            self._relevant_outcome_combos = {**self._relevant_outcome_combos, **output}
        print("Generate iterators")


        #self._relevant_iter_nodes[node] - unnoetig
        #self._relevant_outcomes_nodes[seq_id] - unnoetig
        #self._relevant_outcome_combos[seq_id]

        # set sequences per VI element
        vi_elements = []
        sub_groups = self.sub_groups
        prev = 0
        for i in range(0, len(sequences), sub_groups):
            if i == 0: continue
            rel_indices = list(range(prev, i))
            q_latent = dict((k, self._q_latent[k]) for k in rel_indices)
            q_latent_t = dict((k, self._q_latent_t[k]) for k in rel_indices)
            relevant_outcome_combos = dict((k, self._relevant_outcome_combos[k]) for k in rel_indices)
            seq_group = sequences[prev:i]
            vi_element = VIElement(seq_group, q_latent, q_latent_t, copy.deepcopy(self.tbn), relevant_outcome_combos)
            vi_element.temporal_condition = self.temporal_condition
            vi_element.include_temporal_estimate_to_cat = self.include_temporal_estimate_to_cat
            vi_elements += [[vi_element]]
            prev = i
        rel_indices = list(range(prev, len(sequences)))
        q_latent = dict((k, self._q_latent[k]) for k in rel_indices)
        q_latent_t = dict((k, self._q_latent_t[k]) for k in rel_indices)
        seq_group = sequences[prev:]
        relevant_outcome_combos = dict((k, self._relevant_outcome_combos[k]) for k in rel_indices)
        vi_element = VIElement(seq_group, q_latent, q_latent_t, copy.deepcopy(self.tbn), relevant_outcome_combos)
        vi_element.temporal_condition = self.temporal_condition
        vi_element.include_temporal_estimate_to_cat = self.include_temporal_estimate_to_cat
        vi_elements += [[vi_element]]
        print("Start parallel execution")

        # 3. VI Iteration
        self._observations = sequences
        for vi_iteration in range(self.iteration_frequency):

            start_time, k = self._initialize_vi_iteration(vi_iteration)

            debug = False # True
            if debug:
                out = []
                for vi_element in vi_elements:
                    out += [iter_vi(vi_element[0])]

                # shuffle order of outcomes this is reality!
                from random import shuffle
                shuffle(out)
                print(str(out))
            else:
                out = parallelize_stuff(vi_elements, iter_vi, simultaneous_processes = self._parallel_processes)

            #Compute global parameters
            print("Done Running")
            print("Globals update...")
            self._recompute_global_parameters(out, start_time)
            print("copy update...")
            for vi_element in vi_elements:
                vi_element[0].tbn = copy.deepcopy(self.tbn)
            print("done updating...")
        self._log(3)

    def _update_times(self, iter_outs):
        '''
        Based on latent variable estimates q(z) compute new CPDs and set the according parameters
        p(A|B,C) = q(A)*q(B)*q(C)
        mu = weighted mean of q_all(outcomes) * mu over all sequences = q1*m1 + q2*m2 + q3*m3 / q1 + q2 + q3
        :param sequences: List of sequences in format: listeelement is dict. of key: TV name value: [state, start_time_of_state, end_time_of_state]
        '''

        # 1. Iterate temporal nodes and update them (e.g. p(dT_A|B,C))
        kk = 0
        for node in self.tbn.V:
            kk += 1
            #print("Update t %s / %s" % (str(kk), str(len(self.tbn.V))))
            if not str.startswith(node, "dL_"): continue
            #print("\n\n------> NODE: %s" % str(node))
            rem = []

            # 2. Iterate all outcome combinations
            for cond in self.tbn.Vdata[node]["hybcprob"]:

                # Remove failed entries
                if not eval(cond):
                    rem += [cond]
                    continue

                # Initialize Values
                sum_mu_times_q = 0.0
                sum_q = 0.0

                # 3. Iterate local estimates to compute global parameters
                for iter_out in iter_outs:
                    sum_mu_times_q += iter_out.resulting_mu[node][cond]
                    sum_q += iter_out.resulting_mu_q[node][cond]

                # 4. Compute new estimate as mu = q(p1) * mu1 + q(p2)*mu2 ... / sum of q(p1) ... q(px)
                self.tbn.Vdata[node]["hybcprob"][cond]["mean_base"] = sum_mu_times_q / sum_q

                #print("--> parents: %s" % str(cond))
                #print("--> dist: %s" % str(self.tbn.Vdata[node]["hybcprob"][cond]["mean_base"]))


                if np.isnan(self.tbn.Vdata[node]["hybcprob"][cond]["mean_base"]):
                    self.tbn.Vdata[node]["hybcprob"][cond]["mean_base"] = 0.0

            # 5. Remove failed entries (exception)
            for r in rem: del self.tbn.Vdata[node]["hybcprob"][r]

    def _update_cpd(self, iter_outs):
        '''
        Based on latent variable estimates q(z) compute new CPDs
        Assume random variables to be independent  p(A|B,C) = q(A)*q(B)*q(C)
        :param sequences: Sequences to iterate over
        '''

        # 1. Iterate nodes
        kk = 0
        for node in self.tbn.V:
            kk += 1
            #print("Update %s / %s" % (str(kk), str(len(self.tbn.V))))
            if str.startswith(node, "dL_"): continue
            #print("\n\n------> NODE: %s" % str(node))
            new_cprob = dict()

            # 2. No parents given: estimate is average of all local estimates
            if self.tbn.Vdata[node]["parents"] is None:
                L().log.info("Updated CPD - %s: %s" % (node, str(self.tbn.Vdata[node]["cprob"])))
                self.tbn.Vdata[node]["cprob"] = np.zeros(np.array(self.tbn.Vdata[node]["vals"]).shape)
                for iter_out in iter_outs:
                    self.tbn.Vdata[node]["cprob"] += iter_out.resulting[node]
                self.tbn.Vdata[node]["cprob"] /= np.sum(self.tbn.Vdata[node]["cprob"])

                #print("--> parents: %s" % str("None"))
                #print("--> dist: %s" % str(self.tbn.Vdata[node]["cprob"]))
                continue

            # 4. Iterate outcomes
            for iter_out in iter_outs:
                for cond in self.tbn.Vdata[node]["cprob"]:
                    if not cond in new_cprob: new_cprob[cond] = np.zeros(np.array(self.tbn.Vdata[node]["vals"]).shape)
                    new_cprob[cond] += iter_out.resulting[node][cond]

                    # 6. Normalize distribution on last iteration
                    if iter_out == iter_outs[-1]:
                        if np.sum(new_cprob[cond]) != 0: new_cprob[cond] /= np.sum(new_cprob[cond])
                        else:
                            ono = np.ones(np.array(self.tbn.Vdata[node]["vals"]).shape)
                            new_cprob[cond] = ono/np.sum(ono)


            #for cond in self.tbn.Vdata[node]["cprob"]:
            #    print("--> parents: %s" % str(cond))
            #    print("--> dist: %s" % str(self.tbn.Vdata[node]["cprob"][cond]))

            # 7. Set updated distributions
            self.tbn.Vdata[node]["cprob"] = new_cprob

    def _compute_relevant_iter_nodes(self, node):
        '''
        For a given node only the markov blanket is relevant when computing the CAVI Update. Thus, here all nodes
        of the Markov Blanket of a node are returned - Here only categorical elements of the Markov Blanket are
        returned
        :param node: Node to find the Markov Blanket to
        :return: list of relevant categorical nodes i.e. parents, children and co-parents
        '''

        # 1. Initialize
        relevant_nodes = []

        # 2. Node's parents
        if not self.tbn.Vdata[node]["parents"] is None:
            relevant_nodes += self.tbn.Vdata[node]["parents"]

        # 3. Node's children and co-parents
        for child in self.tbn.Vdata[node]["children"]:
            if str.startswith(child, "dL_"): continue
            relevant_nodes += [child]
            relevant_nodes += self.tbn.Vdata[child]["parents"]

        # 4. drop duplicates
        relevant_nodes = set(list(relevant_nodes))
        return relevant_nodes

    def _only_valid(self, nodes, dest_size, node_names, res_dict):
        '''
        This method computes, for a series of given input sequences, which combinations of output sequences are allowed
        when streching this series to given nodes. Then, per node all allowed outcomes are returned. However, a
        second method .... is required to check if combinations of node outcomes yield the node sequence given here
        e.g. ABCDE might result in dict - node_name[0]:A, node_name[1]:A,B ...
        if nodes is ["A", "B", "C", "D", "E"] and dest is 7 nodes
        0: A   1: A,B    2: A,B,C    3: A,B,C    4: A,B,C    5: B,C      6:  C

        ATTENTION: the result is passed by reference in this method via the argument res_dict!
        :param nodes: list of outcomes to stretch to the destination size dest
        :param dest_size: Destination size of the target sequence
        :param node_names: List of node names to assign to the streched sequence  e.g. [V0_0, V0_1, ...]
        :param res_dict: Dictionary containing result of streched sequence, with key: node_name_at_index and value: list of possible outcomes
        :return: The output is returned by reference via the dictionary res_dict
        '''

        # Maximum size of any node is number of possible nevers (nevs) + 1
        # when reached add 1 per iteration
        nevs = dest_size - len(nodes)

        # Approach left side: Fill A A,B A,B,C until number of nevers reached then, shift right and remove
        shift_idx = 0
        for i in range(1, dest_size+1):
            k = i
            if k > len(nodes): # k - shift_idx is size of result
                k = len(nodes) # remove the first
                if dest_size - i+1 <  k- shift_idx:  # number of remaining spots, here remove one
                    shift_idx += 1
            size = k - shift_idx
            if size > nevs+1: # then shift index
                shift_idx += 1
            if not i in res_dict:
                res_dict[node_names[i-1]] = nodes[shift_idx:k]

    def _compute_relevant_outcomes_nodes(self, observation):
        '''
        Given the observation per subset of nodes only certain outcomes are allowed per node i.e. the ones that
        are legid permutations
        This method returns a dictionary of node names as keys and according lists of values that are possible for
        that node, where a value containts [outcome, start_time, end_time] of the state

        e.g. V0_1 value: [A, C, F] - i.e.. for sequence with sequence_id, the node V0_1 can only have values A, C or F
        :param observation: one sequence containing key: TV name and value: list of [state, start_time, end_time]
        :return relevant: allowed outcomes per node - key: node_name and value: list of possible outcomes as [outcome, start_time, end_time]
        '''
        relevant = dict()
        for temp_var_name in observation:

            # extract observation elements for this TV
            obs_sequence = [(t[0], t[1], t[2]) for t in observation[temp_var_name]]
            node_names = [t for t in self.tbn.V if t.split("_")[0] == temp_var_name]
            number_nodes = len(node_names)

            # Determine legid mappings
            self._only_valid(obs_sequence, number_nodes, node_names, relevant)

        return relevant

    def _dict_of_lists_product(self, dicts):
        '''
        For a given set of dictionaries determines the cross product of all
        outcomes
        :param dicts: list of dictionaries
        :return: cross product of all dictionaries
        '''
        return (dict(zip(dicts, x)) for x in product(*dicts.values()))

    def _temporal_approximation(self, outcome, seq_id, iter, observ):
        '''
        When streching a sequence (e.g. AB) to a longer size (e.g. AABB) each node needs to have
        assigned a start and end time. This method computes a linear interpolation per outcome as follows
            Starttimes
            - Nevers between state changes are linearly interpolated e.g. A=1, B=2 for AAAB gives A=1, A=1.33,A=1.66,B=2.0
            - If the last short sequence element is followed by further elements the distance between this element
              and its preceeding element is used for further steps e.g. seen A=1, B=2 then, AABBB would be A=1,A=1.5, B=2,B=2.5,B=3
              -> This upstep time is determined as
                  - last_time minus time before last seen time
                  - if not available: total time of sequence divided by number of nodes
            Endtimes:
            - Endtimes are always the start times of the preceeding element
            - if the last element was extended the end time is the extended value + the equidistance to the preceding element

        :param outcome: Outcome to be temporally approximated i.e. a dictionary of node_name:node_value
        :param seq_id: Id of current sequence - used to store the result to avoid recomputation
        :param iter: Id of current iteration - used to store the result to avoid recomputation
        :param observ: one sequence containing key: TV name and value: list of [state, start_time, end_time]
        :return outcome: Corrected outcome with temporally approximated times
        '''

        # 1. Avoid recomputation of information by storing in dictionary
        if seq_id in self._store_temporal_approximation and iter in self._store_temporal_approximation[seq_id]:
            return self._store_temporal_approximation[seq_id][iter]

        # 2. Determine temporal variables
        temporal_vars = set(list([n.split("_")[0] for n in outcome if not str.startswith(n, "dL_")]))

        # 3. Approximate per TV
        for tv in temporal_vars:

            # 4. Initialize times of long sequences e.g. AABB [A_time, A_time, B_time, B_time]
            all_times = [outcome[k][1] for k in outcome if k.split("_")[0]== tv and not str.startswith(k, "dL_")]
            times = sorted(list(set(all_times))) # times of short sequence i.e. e.g. AB
            all_times = np.array(all_times)

            # 5. Determine upstep size
            try: upstep_size = times[-1] - times[-2]
            except:
                end_time = sorted(list(set([outcome[k][2] for k in outcome if k.split("_")[0]==tv and not str.startswith(k, "dL_")])))[-1]
                upstep_size = end_time/float(len(all_times)+1)
            if len(all_times) == len(times): continue

            # 6. Linear Interpolation
            for t in times:
                if times[-1] == t:
                    kko = all_times[all_times==t]
                    l_steps = len(kko)
                    all_times[all_times==t] = np.linspace(t, t+(l_steps)*upstep_size, len(kko)+1)[:len(kko)]
                else:
                    kko = all_times[all_times==t]
                    all_times[all_times==t] = np.linspace(t, all_times[all_times>t][0], len(kko)+1)[:len(kko)]

            # 7. Assign result to outcomes and compute end times
            for q in range(len(all_times)):
                try: last_time = all_times[q+1]
                except: last_time = observ[tv][-1][-1]
                outcome[tv+"_"+str(q)] = (outcome[tv+"_"+str(q)][0], all_times[q], last_time)
                if outcome[tv+"_"+str(q)][1] > outcome[tv+"_"+str(q)][2]:
                    outcome[tv+"_"+str(q)] = (outcome[tv+"_"+str(q)][0], outcome[tv+"_"+str(q)][1], outcome[tv+"_"+str(q)][1]  + upstep_size)

        # 8. Avoid recomputation of information by storing in dictionary
        if not seq_id in self._store_temporal_approximation: self._store_temporal_approximation[seq_id] = dict()
        if not iter in self._store_temporal_approximation[seq_id]: self._store_temporal_approximation[seq_id][iter] = outcome

        return outcome

    def _update_latent_time(self, node, seq_id, observation):
        '''
        For the current sequence the latent local estimate for the temporal node dt is q(dt; mu, sigma) and is assumed Gaussian and thus, is governed
        by mu and sigma, sigma is assumed constant in our case, mu is recomputed from the observed data
        :param node: name of temporal node to be updated
        :param seq_id: id of sequence to be updated
        :param observation: one sequence containing key: TV name and value: list of [state, start_time, end_time]
        :return:
        '''

        # 1. Two strategies: 0: repeat last time, 1: linear interpolation
        strategy = 1
        self._determine_relevant(seq_id, node[3:], observation)
        mu_numerator, mu_denominator, iter = 0.0, 0.0, 0

        # 2. iterate all combinations of outcomes /for me as node only my intranodes and inter nodes are relevant not all!
        for outcome in self._relevant_outcome_combos[seq_id]:
            iter += 1

            # 3. per output combination e.g. ABBA BBC assign all absolute times to nodes
            if strategy == 1: outcome = self._temporal_approximation(outcome, seq_id, iter, observation)

            # 4. Compute dT = myTime - earliestParentTime
            parent_times = [outcome[p][1] for p in self.tbn.Vdata[node]["parents"] if "dL_"+p != node]
            my_outcome = outcome[node[3:]][0]
            my_time = outcome[node[3:]][1]
            if parent_times: max_time = np.max(parent_times)
            else: max_time = 0.0

            # 5. Current Mue
            mu = my_time - max_time
            var = 0.01

            # 6. Compute q_all = q1(this_outcome) q2(this_outocme)
            q_all = self._q_latent[seq_id][node[3:]][self.tbn.Vdata[node[3:]]["vals"].index(my_outcome)]
            q_all *= np.prod([self._q_latent[seq_id][k][self.tbn.Vdata[k]["vals"].index(outcome[k][0])] for k in self.tbn.Vdata[node]["parents"]])

            # 7. Compute update for mu
            mu_numerator = (mu/var) * q_all
            mu_denominator = (1/var) * q_all

        # 8. Update mu
        new_est = mu_numerator / mu_denominator
        if new_est<0: new_est = 0.0
        self._q_latent_t[seq_id][node]["mean_base"] = new_est

    def _determine_relevant(self, seq_id, node, observation):
        '''
        Given constraints only a subset of nodes and outcomes is relevant, which is stored here
        to iterate over
        :param seq_id: Id of current sequence - used to store information to avoid recomputation
        :param node: current node - used to determine and store information to avoid recomputation
        :param observation: current observation sequence  - used to determine and store information to avoid recomputation
        '''

        # 1. Determine relevant nodes for this node
        if not node in self._relevant_iter_nodes:
            self._relevant_iter_nodes[node] = self._compute_relevant_iter_nodes(node)

        # 2. Determine relevant outcomes per node
        if not seq_id in self._relevant_outcomes_nodes:
            self._relevant_outcomes_nodes[seq_id] = self._compute_relevant_outcomes_nodes(observation)

        # 3. Find allowed outcome combinations only
        if not seq_id in self._relevant_outcome_combos:
            # per node allowed combinations
            extended_combos = self._dict_of_lists_product(self._relevant_outcomes_nodes[seq_id])
            # only sequences that result in original allowed e.g. AAABC -> ABC
            self._relevant_outcome_combos[seq_id] = self._only_short_is_original(extended_combos, observation)

    def _determine_relevant_light(self, seq_id, observation):
        '''
        Given constraints only a subset of nodes and outcomes is relevant, which is stored here
        to iterate over
        :param seq_id: Id of current sequence - used to store information to avoid recomputation
        :param node: current node - used to determine and store information to avoid recomputation
        :param observation: current observation sequence  - used to determine and store information to avoid recomputation
        '''

        # 1. Determine relevant nodes for this node
        relevant_outcomes_nodes = self._compute_relevant_outcomes_nodes(observation)
        extended_combos = self._dict_of_lists_product(relevant_outcomes_nodes)
        # relevant_outcome_combos[seq_id] = self._only_short_is_original(extended_combos, observation)
        rel_combos = self._only_short_is_original(extended_combos, observation)
        return rel_combos

    def _only_short_is_original(self, extended_combos, observation):
        '''
        For a given set of extended sequences e.g. AABBDD, ABDC deterine the ones
        that are consistent with the observed e.g. ABC observed, then AB not possible
        :return: list of short and consistent outcome combinations
        '''
        res_lst = []
        for outc in extended_combos:
            worked = True
            for n in observation:
                obs = [a[0] for a in observation[n]]
                k = -1
                t = []
                while True:
                    k += 1
                    try: t += [outc[n +"_"+ str(k)][0]]
                    except: break
                short = self._drop_duplicate_list(t)

                # check if short equals observation
                if not np.array_equal(short, obs):
                    # drop this entry
                    worked = False
                    break
            if worked:
                res_lst += [outc]
        return res_lst

    def dt_integrand(self, t, mu, var):
        '''
        Assuming a Gaussian per temporal node this integrand has to be computed
        :param t: delta time as variable to integrate over
        :param mu: constant mu
        :param var: consant var
        :return: integral over a*b as shown below
        '''
        a = (-(t*t)) + 2*mu*t - mu*mu
        b = (1/np.sqrt(2*np.pi*var)) * np.exp(-( ((t-mu)*(t-mu))/(2*var)))

        return a*b

    def _variational_iteration(self, observation, seq_id):
        '''
        Per sequence a set of local estimates is found in this step and iterated
        :param observation: current observed sequence
        :param seq_id: Identifier of current sequence
        '''

        # 1. Iterate over all nodes and update them
        for node in self.tbn.V:

            retry, temporal_cond = True, self.temporal_condition
            oo = 0
            while retry:

                # 2. Compute update for continuous temporal node
                if str.startswith(node, "dL_"):
                    self._update_latent_time(node, seq_id, observation)

                    # Update CPD
                    '''for cond in self.tbn.Vdata[node]["hybcprob"]:
                        i, q = -1, 1.0
                        for pa in self.tbn.Vdata[node]["parents"]:
                            i += 1
                            q *= self._q_latent[seq_id][pa][self.tbn.Vdata[pa]["vals"].index(eval(cond)[i])] # this seq: q(pa) * mu(pa)
                        mu = self._q_latent_t[seq_id][node]["mean_base"]
                        if not node in self._new_mu_times_q:
                            self._new_mu_times_q[node] = dict()
                            self._new_mu_times_q[node][cond] =0.0
                            self._new_sum_q[node] = dict()
                            self._new_sum_q[node][cond] = 0.0

                        self._new_mu_times_q[node][cond] += mu * q
                        self._new_sum_q[node][cond] += q'''

                    retry = False
                    continue

                else:
                    retry, temporal_cond = self._update_latent_node(seq_id, observation, node, retry, temporal_cond)

                    '''if not retry:
                        # add to global estimation
                        if not node in self._new_global_cpd:
                            self._new_global_cpd[node] = dict()

                        if not self.tbn.Vdata[node]["parents"] is None:
                            for cond in self.tbn.Vdata[node]["cprob"]:
                                if not cond in self._new_global_cpd[node]:
                                    self._new_global_cpd[node][cond] = np.zeros(np.array(self.tbn.Vdata[node]["vals"]).shape)
                                p = copy.deepcopy(self._q_latent[seq_id][node]) # node estimate
                                i = -1
                                for pa in self.tbn.Vdata[node]["parents"]: # parent estimates
                                    i += 1
                                    p *= self._q_latent[seq_id][pa][self.tbn.Vdata[pa]["vals"].index(eval(cond)[i])]
                                self._new_global_cpd[node][cond] += p

                                # 6. Normalize distribution on last iteration
                                if seq_id+1 == len(self._observations):
                                    if np.sum(self._new_global_cpd[node][cond]) != 0: self._new_global_cpd[node][cond] /= np.sum(self._new_global_cpd[node][cond])
                                    else:
                                        ono = np.ones(np.array(self.tbn.Vdata[node]["vals"]).shape)
                                        self._new_global_cpd[node][cond] = ono/np.sum(ono)'''

                if oo > 1:
                    print("retry %s" % str(oo))
                oo += 1

    def _update_latent_node(self, seq_id, observation, node, retry, temporal_cond):

        # if observation of this node not latent - q(z) is eindeutig - egal was parents sind!

         # no update needed - only one observation possible
        if seq_id in self._direct_q_nodes and node in self._direct_q_nodes[seq_id]:
            retry = False
            return  retry, temporal_cond
        else:
            tv = node.split("_")[0]
            number_observed = len(observation[node.split("_")[0]])
            if tv not in self._number_actual:
                self._number_actual[tv] = len([t for t in self.tbn.V if t.split("_")[0] == tv])
            if self._number_actual[tv] == number_observed:
                self._q_latent[seq_id][node] = np.zeros(self._q_latent[seq_id][node].shape)

                # observation an stelle int(node.split("_")[-1])
                node_id = int(node.split("_")[-1])
                node_val = observation[tv][node_id][0]
                idx_one = self.tbn.Vdata[node]["vals"].index(node_val)
                self._q_latent[seq_id][node][idx_one] = 1.0
                retry = False
                if not seq_id in self._direct_q_nodes:
                    self._direct_q_nodes[seq_id] = dict()
                if not node in self._direct_q_nodes[seq_id]:
                    self._direct_q_nodes[seq_id][node] = True
                return retry, temporal_cond


        # 1. Determine all valid outcomes
        #self._log(5, (node, seq_id, observation))
        self._determine_relevant(seq_id, node, observation)

        # 2. store relevant qs to avoid recomputation
        q_compute, lst, gen = False, [], False
        if not seq_id in self._relevant_qs: self._relevant_qs[seq_id] = dict()
        if not node in self._relevant_qs[seq_id]: self._relevant_qs[seq_id][node], q_compute = [], True
        if isinstance(self._relevant_outcome_combos[seq_id], list): gen = True

        # 3. Compute expectation: Iterate over all allowed observations
        result_expectation = np.zeros(len(self.tbn.Vdata[node]["vals"]))
        for outcome in self._relevant_outcome_combos[seq_id]:

            # 4. Store for next iteration
            if not gen: lst += [outcome]
            #L().log.debug("Outcome: %s" % str(outcome))

            # 5. p(node|parents_node_alle_versionen)
            current_outcome_idx = self.tbn.Vdata[node]["vals"].index(outcome[node][0])
            if not self.tbn.Vdata[node]["parents"] is None:

                # 6. Check temporal conditions
                parent_start_times = np.array([outcome[p][1] for p in self.tbn.Vdata[node]["parents"]])
                parent_end_times = np.array([outcome[p][2] for p in self.tbn.Vdata[node]["parents"]])
                my_time = outcome[node][1]
                my_end_time = outcome[node][2]
                if temporal_cond and not self._temporal_conditions_given(node, outcome, parent_start_times, parent_end_times, self.tbn.Vdata[node]["parents"], my_time, my_end_time):
                    continue

                # 7. Determine p(node|parents_this_outcome) = p_me_parents
                parent_outcomes = str([outcome[p][0] for p in self.tbn.Vdata[node]["parents"]])
                p_me_parents = self.tbn.Vdata[node]["cprob"][parent_outcomes][current_outcome_idx]

                # 8 . compute q for all parents
                if q_compute: self._relevant_qs[seq_id][node] += self.tbn.Vdata[node]["parents"]

            else:
                # 9. If no parents given
                p_me_parents = self.tbn.Vdata[node]["cprob"][current_outcome_idx]

            ''' ------------ CATEGORICAL PART ------------ '''
            # 10. Compute p of child nodes for this outcome combo - CATEGORICAL PART
            # p(child_1|parents_und_node) p(child_2|parents_und_node) # all child and parent versions
            p_child_parents = 1.0
            for child in self.tbn.Vdata[node]["children"]:
                if str.startswith(child, "dL_"): continue
                child_outcome_idx = self.tbn.Vdata[child]["vals"].index(outcome[child][0])
                parent_outcomes = str([outcome[p][0] for p in self.tbn.Vdata[child]["parents"]])
                p_child_parents *= self.tbn.Vdata[child]["cprob"][parent_outcomes][child_outcome_idx]

                if q_compute: self._relevant_qs[seq_id][node] += self.tbn.Vdata[child]["parents"]
                if q_compute: self._relevant_qs[seq_id][node] += [child]

            # 11. Compute result of categorical part
            # compute q_total = q = q(z1)*q(z2)*...
            if q_compute:
                self._relevant_qs[seq_id][node] = list(set(self._relevant_qs[seq_id][node]))
                if node in self._relevant_qs[seq_id][node]: self._relevant_qs[seq_id][node].remove(node) # NOT q(me) else I miss the point of this!!!
            q_all = np.prod(np.array([self._q_latent[seq_id][p] [self.tbn.Vdata[p]["vals"].index(outcome[p][0])] for p in self._relevant_qs[seq_id][node]]))

            # 12. Add result to final expectation
            result_expectation[current_outcome_idx] += np.log(p_me_parents * p_child_parents) * q_all

            ''' ------------ TEMPORAL PART ------------ '''
            # 13. Compute p of child nodes for this outcome combo - CONTINUOUS PART
            # q(all_except_t) * (-(1/(2sg^2)) * integral( -(t-mu)^2 * q(t))dt
            if self.include_temporal_estimate_to_cat:
                mu = self._q_latent_t[seq_id]["dL_"+node]["mean_base"]
                var = self._q_latent_t[seq_id]["dL_"+node]["variance"]
                fst = -(1/2*var)
                integral = quad(self.dt_integrand, mu-5*var, mu+5*var, args=(mu, var))[0]
                second = fst*integral*q_all
                result_expectation[current_outcome_idx] += second # 4 mit time, 4 ohne time
        if not gen: self._relevant_outcome_combos[seq_id] = lst

        # 14. ALL OUTCOME COMBOS DONE: UPDATE q(node)
        result_expectation[result_expectation == 0.0] = -np.inf # 0 means not seen -> which means prob. of 0 -> i.e. -inf!
        result_expectation = np.nan_to_num(result_expectation)

        q_est = np.exp(result_expectation)

        # OVERFLOW - Number to big -> nan
        if np.sum(q_est) == 0:
            if not np.all(result_expectation == result_expectation[0]):
                result_expectation /= np.max(np.abs(result_expectation))
                q_est = np.exp(result_expectation)
            else:
                q_est = np.ones(q_est.shape)
        self._q_latent[seq_id][node] = q_est/ np.sum(q_est)

        # 15. If temporal condition failed
        #     -> recompute without temporal condition
        if np.isnan(self._q_latent[seq_id][node][0]):
            temporal_cond = False
            #L().log.debug("Require to retry without temporal aspect")
            print("Require to retry without temporal aspect %s and %s" % (str(result_expectation), str(q_est)))
        else:
            retry = False

        # 16. Logging
        #L().log.debug("-------------> Update Latent Variable %s at Seq %s" % (str(self._q_latent[seq_id][node]), str(seq_id)))
        #L().log.debug("\n")

        return retry, temporal_cond

    def _temporal_conditions_given(self, node, outcome, parent_start_times, parent_end_times, parents, my_start_time, my_end_time):
        ''' This method checks if the given outcome combination with given node with start time my_start_time and ending time my_end_time,
            and parent nodes with parent_start_times and parent_end_times are logical and therefore legid sequence outcomes
            The following conditions need to hold:
                - condition 1: wenn node==Never, dann muss Endzeit_node> all parents_end_zeiten
                - condition 2: wenn node==occurring, a parent_Never - this_parent_start_times < node_end_time
                - condition 3: wenn node==occ, parent== occ - parent_start_time<node_start_time
            :param node: Name of current node
            :param outcome: Values of the current outcome
            :param parent_start_times: List of start times of all parents
            :param parent_end_times: List of end times of all parents
            :param my_start_time: Start time of currents node outcome
            :param my_end_time: End time of currents node outcome
        '''

        # 1. Check current node is never
        my_prev_node = "_".join(node.split("_")[:-1]) +"_"+ str(int(node.split("_")[-1])-1)
        me_is_never =  outcome[my_prev_node] == outcome[node]

        # 2. Iterate parents
        for par in range(len(parent_end_times)):
            this_parent = parents[par]
            this_parent_end_time = parent_end_times[par]
            this_parent_start_time = parent_start_times[par]

             # 3. Check parent is never i.e. parent_of_parent == parent
            prev_prev_node = "_".join(this_parent.split("_")[:-1]) +"_"+ str(int(this_parent.split("_")[-1])-2)
            prev_node = "_".join(this_parent.split("_")[:-1]) +"_"+ str(int(this_parent.split("_")[-1])-1)
            try: par_is_never = outcome[prev_node] == outcome[prev_prev_node]
            except: par_is_never = False

            # Condition 1: if node==Never -> End_time_node> all parents_end_times
            if me_is_never:
                if my_end_time < this_parent_end_time:
                    return False
            else:
                # Condition 2: if node==occ., a parent_Never - this_parent_start_times < node_end_time
                if par_is_never and this_parent_start_time > my_end_time:
                    return False

                # Condition 3: if node==occ., parent== occ - parent_start_time<node_start_time
                if (not par_is_never) and this_parent_start_time > my_start_time:
                    return False
        return True

    def _log_cpds(self, normalize=True, show_latent=True):
        '''
        Logs the current CPDs of all Trees and writes it to file
        :param normalize: if True normalizes all distributions
        :param show_latent: if True prints the latent local estimates q(z) as well
        '''
        L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
        L().log.info("     New CPDs")
        L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
        for n in self.tbn.Vdata:
            if str.startswith(n, "dL_"):
                if isinstance(self.tbn.Vdata[n]["hybcprob"], dict):
                    for k in self.tbn.Vdata[n]["hybcprob"]:
                        L().log.info("%s | %s = %s" % (n, k, str(self.tbn.Vdata[n]["hybcprob"][k])))
                else:
                    L().log.info("%s = %s" % (n, str(self.tbn.Vdata[n]["hybcprob"])))


                continue
            if isinstance(self.tbn.Vdata[n]["cprob"], dict):
                for k in self.tbn.Vdata[n]["cprob"]:
                    if normalize:
                        self.tbn.Vdata[n]["cprob"][k] /= np.sum(self.tbn.Vdata[n]["cprob"][k])

                    L().log.info("%s | %s = %s" % (n, k, str(self.tbn.Vdata[n]["cprob"][k])))

            else:
                if normalize:
                    self.tbn.Vdata[n]["cprob"] /= np.sum(self.tbn.Vdata[n]["cprob"])
                L().log.info("%s = %s" % (n, str(self.tbn.Vdata[n]["cprob"])))
        L().log.info("\n")
        if show_latent:
            L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
            L().log.info("     Latent Kolegas")
            L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
            for sequence in self._q_latent:
                if sequence>20: break
                L().log.info( "------------------------- Seq %s -----------------------" % (str(sequence)))
                L().log.info("Raw: %s" % str(self._observations[sequence]))
                for node in self._q_latent[sequence]:
                    L().log.info( "%s = %s" % (node, str(self._q_latent[sequence][node])))

                for node in self._q_latent_t[sequence]:
                    L().log.info( "%s = %s" % (node, str(self._q_latent_t[sequence][node])))

        L().log.info("\n\n")

    def _log_reference_cpds(self):
        '''
        Logs the current CPDs of all Trees and writes it to file
        :param normalize: if True normalizes all distributions
        :param show_latent: if True prints the latent local estimates q(z) as well
        '''
        L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
        L().log.info("     Original CPDs")
        L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
        for n in self._reference.Vdata:
            if str.startswith(n, "dL_"):
                if isinstance(self._reference.Vdata[n]["hybcprob"], dict):
                    for k in self._reference.Vdata[n]["hybcprob"]:
                        L().log.info("%s | %s = %s" % (n, k, str(self._reference.Vdata[n]["hybcprob"][k])))
                else:
                    L().log.info("%s = %s" % (n, str(self._reference.Vdata[n]["hybcprob"])))

                continue
            if isinstance(self._reference.Vdata[n]["cprob"], dict):
                for k in self._reference.Vdata[n]["cprob"]:
                    L().log.info("%s | %s = %s" % (n, k, str(self._reference.Vdata[n]["cprob"][k])))

            else:
                L().log.info("%s = %s" % (n, str(self._reference.Vdata[n]["cprob"])))
        L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
        L().log.info("\n\n")
