import copy
import random
import traceback
import math
from scipy.stats import norm

from _include.estimator.ctbn_estimator import CTBNEstimator
from network.dbndiscrete import DBNDiscrete
from _include.estimator.em_algorithm_tscbn_estimator import EMAlgorithmParameterEstimator
from _include.toolkit import PNT
from _include.evaluation.base_evaluator import BaseEvaluator
import numpy as np
from sklearn.metrics import mean_squared_error

class ParameterEvaluator(BaseEvaluator):
    '''
    Based on given results this class returns an evaluation of the
    parameter learning
    '''

    def test(self):
        import json
        test_sequences = json.load(open('seq.txt'))
        [start_time, end_time, resolution] = json.load(open('det.txt'))
        sequences = self.convert_intervals_to_dbn_sequence(test_sequences, start_time, end_time, resolution)


    def __init__(self, append_csv):
        '''
        Possible Metrics are: 
            - relative-entropy: Relative Entropy zwischen learned P and real P
        '''
        super(ParameterEvaluator, self).__init__(append_csv)

        self.tmp = {}

        # parameters for evaluation
        self.rmse_tscb_variance = 0.2 # variance assumed per node
        self.rmse_mean_range = 0.1 # drift of mean will be within this range e.g. 0.1 means it will be drawn from correct +- drift*correct

    def evaluate(self, model_dict, reference = None, test_sequences= None, test_tscbn_sequences = None):
        '''
        Main method used for evaluation
        '''

        # Result dictionary
        eval_results = {}
        
        for model_name in model_dict:
            print("\nEvaluate: %s %s %s" % (PNT.BOLD, str(model_name), PNT.END))
            
            # Initialize Methods
            model = model_dict[model_name]
            if model_name not in eval_results: eval_results[model_name] = {}
            if model == None:
                eval_results[model_name] = {}
                for metric in self._metrics:eval_results[model_name][metric] = "N.A."
                continue
            
            # Perform evaluation
            for metric in self._metrics:
                try:
                    eval_results[model_name][metric] = self._compute_metric(model, reference, metric, test_sequences = test_sequences, test_tscbn_sequences = test_tscbn_sequences)
                except:
                    eval_results[model_name][metric] = "N.A."

        # Return results
        self._last_eval_results = eval_results
        return eval_results

    def evaluate_direct(self, model, reference=None, test_sequences=None, test_tscbn_sequences=None):
        '''
        Main method used for evaluation
        '''

        # Result dictionary
        eval_results = {}

        # Initialize Methods
        model_name = model.__class__.__name__
        print("\nEvaluate: %s %s %s" % (PNT.BOLD, str(model_name), PNT.END))
        if model_name not in eval_results: eval_results[model_name] = {}

        # Perform evaluation
        for metric in self._metrics:
            eval_results[model_name][metric] = self._compute_metric(model, reference, metric,
                                                                        test_sequences=test_sequences,
                                                                        test_tscbn_sequences=test_tscbn_sequences)

        # Return results
        self._last_eval_results = eval_results
        return eval_results

    def _readable_metric(self, metric):
        ''' 
        Translates a metric key to a readable version
        '''
        if metric == "temp-rmse": return "Temporal RMSE"
        if metric == "temp-rte": return "Rel. Temporal Error"
        if metric == "relative-entropy": return "Relative Entropy"
        if metric == "jpd": return "Joint PD"
        if metric == "log-likelihood": return "Log-Likelihood"
        if metric == "temp-jpd": return "Time Joint PD"
        if metric == "temp-log-likelihood": return "Time Log-Likelihood"
        if metric == "runtime": return "Runtime"
        if metric == "accuracy": return "Accuracy"

        return metric

    def _compute_metric(self, model, reference, metric, test_sequences = [], test_tscbn_sequences = []):
        '''
        Given the resulting model this method computes the metrics specified
        '''
        # convert sequence to appropriate format
        try:
            sequences = None
            m_name = model.__class__.__name__
            sequences, interval_sequence = self._convert_sequences(model, test_sequences, test_tscbn_sequences)
        except:
            #traceback.print_exc()
            pass#print("No conversion possible")

        # Compute
        if (metric == "jpd" or metric == "log-likelihood" or metric == "accuracy") and not self._given_in(self.tmp, m_name, "jpd"):
            self.tmp[m_name]["jpd"], self.tmp[m_name]["log"], self.tmp[m_name]["temp-jpd"], self.tmp[m_name]["temp-log"], self.tmp[m_name]["accuracy"] = self._compute_jpd_n_log(model, sequences, test_sequences)

        if metric == "temp-rmse" or metric == "temp-rte":
            self.tmp[m_name]["temp-rmse"], self.tmp[m_name]["temp-rte"] = self._compute_rmse_rte(model, sequences, test_sequences, reference)

        # Return
        if metric == "runtime":
            return "\t\t\t" + self._compute_runtime(model) # OK

        if metric == "accuracy":
            return "\t\t\t" + str(self.tmp[m_name]["accuracy"])

        if metric == "jpd":
            return "\t\t\t" + str(self.tmp[m_name]["jpd"]) # OK

        if metric == "log-likelihood":
            return "\t" + str(self.tmp[m_name]["log"]) # OK

        if metric == "temp-jpd":
            return "\t\t" + str(self.tmp[m_name]["temp-jpd"]) # t.b.d.

        if metric == "temp-log-likelihood":
            return "" + str(self.tmp[m_name]["temp-log"]) # t.b.d.

        if metric == "relative-entropy":
            return "\t" + str(self._compute_kl_divergence(model, reference))

        if metric == "temp-rmse" and self._given_in(self.tmp, m_name, "temp-rmse"):
            return "\t\t"+ str(self.tmp[m_name]["temp-rmse"])

        if metric == "temp-rte"  and self._given_in(self.tmp, m_name, "temp-rte"):
            return "" + str(self.tmp[m_name]["temp-rte"])

    def _compute_runtime(self, model):
        if model.__class__.__name__ == "dict": return "N.A."
        print("Computing runtime...")
        try:
            t = str(model.parameter_execution_time).split(".")
            return t[0] + "." + t[1][:2] + " sec"
        except:
            return "N.A."

    def _compute_kl_divergence(self, model, reference, print_it = True):
        if model.__class__.__name__ == "dict": return -1
        if print_it: print("Computing Relative entropy...")

        #print(str(model.Vdata))
        #print(str(reference.Vdata))
        try:
            kl_divs = []
            kl_div_cnt = 0
            kl_div_sum = 0
            for node_name in model.Vdata:
                if str.startswith(node_name, "dL_"):
                    for cond in model.Vdata[node_name]["hybcprob"]:
                        real_mu = model.Vdata[node_name]["hybcprob"][cond]["mean_base"]
                        real_sg = 0.1#np.sqrt(model.Vdata[node_name]["hybcprob"][cond]["variance"])
                        ref_mu = reference.Vdata[node_name]["hybcprob"][cond]["mean_base"]
                        ref_sg = 0.1 # np.sqrt(reference.Vdata[node_name]["hybcprob"][cond]["variance"])
                        kl_div = np.log(ref_sg/real_sg) + (real_sg*real_sg+(real_mu-ref_mu)*(real_mu-ref_mu))/(2*ref_sg*ref_sg) - 0.5
                        kl_divs += [kl_div]
                        kl_div_sum += kl_div
                        kl_div_cnt += 1.0
                    continue

                if isinstance(model.Vdata[node_name]["cprob"], dict):
                    for cond in model.Vdata[node_name]["cprob"]:
                        dist = model.Vdata[node_name]["cprob"][cond]
                        ref_dist = reference.Vdata[node_name]["cprob"][cond]
                        kl_div = self._kl_divergence(dist, ref_dist, eps = 0.001)
                        if np.isnan(kl_div):
                            self._kl_divergence(dist, ref_dist, eps = 0.001)
                            print("NaN")
                        kl_divs += [kl_div]
                        kl_div_sum += kl_div
                        kl_div_cnt += 1.0
                else:
                    dist = model.Vdata[node_name]["cprob"]
                    ref_dist = reference.Vdata[node_name]["cprob"]

                    kl_div = self._kl_divergence(dist, ref_dist, eps=0.001)
                    if np.isnan(kl_div):
                        self._kl_divergence(dist, ref_dist, eps = 0.001)
                        print("NaN")

                    kl_divs += [kl_div]
                    kl_div_sum += kl_div
                    kl_div_cnt += 1.0


        except:
            if not isinstance(model, DBNDiscrete):
                return EMAlgorithmParameterEstimator.LAST_KL_DIVERGENCE

            #    #self._compute_kl_divergence(model, reference, print_it = True)
            return "N.A."

        return  kl_div_sum/kl_div_cnt

    def _compute_rmse_rte(self, model, sequences, whole_sequences, reference):
        print("Computing RMSE and RTE (independent of parameter estimation!)...")

        self._var_assumed = 0.2 # je nach Variance
        t_rmse, t_rte = 0,0

        # convert sequences if needed
        if model.__class__.__name__ == "DBNDiscrete":

            # initial
            set_length = len([node for node in model.V if str.startswith(node, "V0_")])
            initial_set = copy.deepcopy([node for node in model.V if (
            not model.Vdata[node]["parents"] or model.Vdata[node]["parents"] is None) and not str.startswith(node, "dL_")])
            rtes, rmses = [], []  # geht ja weil es ja die selben State Changes sind die ich predicten will d.h. pro Temporale Variable SC schaue ob ich ihn predicted hätte und mit welcher Zeit
            tvs = [i.split("_")[0] + "_" for i in initial_set]

            for seq_idx in range(len(sequences)):
                seq = sequences[seq_idx]
                real_seq = whole_sequences[seq_idx]

                last_state = np.array([seq[i] for i in initial_set])

                # ASSUME all state changes to occur - if they do determine precision - i.e. assume perfect param estimation - PROBLEM: DBN ÜBERSPRINGT AUCH SC WENN ZU GRO?E RESOLUTION
                t = model.start_time
                for i in range(1, set_length):
                    t += model.resolution
                    current_state = np.array([seq[tv+str(i)] for tv in tvs])
                    current_state_dict = {}
                    for tv in tvs:
                        current_state_dict[tv+str(i)] = seq[tv+str(i)]

                    # check if state change
                    if not np.all(last_state == current_state):
                        model_sc = t # recorded at t but really occurred at expected_sc

                        rel_change = np.where(last_state != current_state)[0]
                        change = []
                        change_time = []
                        change_range = []
                        tk = -1
                        for cur_tv in list(set([tv.replace("_", "") for tv in tvs])):
                            idddx = -1
                            tk += 1
                            if not tk in rel_change: continue

                            for el in real_seq[cur_tv]:
                                idddx += 1
                                state = el[0]
                                start_t = el[1]
                                end_t = el[2]

                                # require: all in current state, all within correct time
                                if current_state_dict[cur_tv+"_"+str(i)] == state and start_t < model_sc and end_t > model_sc:
                                    change.append(idddx)
                                    change_time.append(start_t)
                                    change_range.append([start_t, end_t])
                                    break

                        # EVALUATE ALL STATE CHANGES - V.A. tue das ja auf identischen Daten den Vergleich
                        rt = -1
                        for expected_sc in change_time:
                            rt += 1
                            rmse = math.sqrt(mean_squared_error([expected_sc], [model_sc]))
                            rte = np.abs(expected_sc-model_sc)/(change_range[rt][1] - change_range[rt][0])# bezogen auf Intervallänge des vorgängers im Original
                             # hier zwei möglichkeiten - erstes oder letztes // oder mean von beiden!
                            rmses += [rmse]
                            rtes += [rte]

                    last_state = current_state

        if model.__class__.__name__ == "TSCBN":

            # per node check drift

            # go through all parents
            # - if parents given estimate child assuming given mean and variance
            # - if state is "never" - remove it and use "long path" with according mean and variance

            rtes, rmses = [], []  # geht ja weil es ja die selben State Changes sind die ich predicten will d.h. pro Temporale Variable SC schaue ob ich ihn predicted hätte und mit welcher Zeit

            initial_set = copy.deepcopy([node for node in model.V if (
                not model.Vdata[node]["parents"] or model.Vdata[node]["parents"] is None)])

            for seq_idx in range(len(sequences)):  # Probability of time comes in - distance to previous
                seq = sequences[seq_idx]
                real_seq = whole_sequences[seq_idx]
                i = 0
                jpd = 1.0
                log_likelihood = 0.0
                t_jpd = 1.0
                t_log_likelihood = 0.0
                node_set = initial_set
                done = []
                t_abs = {}
                while node_set:

                    # Get next parent
                    node_set = list(set(node_set))
                    if i >= len(node_set): i = 0
                    p = node_set[i]
                    i += 1

                    # append jpd
                    val = seq[p]
                    if not  (model.Vdata[p]["parents"] is None or not model.Vdata[p]["parents"]):
                        # drift is anyway -> real - val from (drawn_mean, variance) i.e. draw from distribution and drift

                        mean = random.random() * self.rmse_mean_range
                        var = self.rmse_tscb_variance
                        drift = np.abs(np.random.normal(mean, np.sqrt(var), 1)[0])
                        try:
                            parent_t_abs = [t_abs[par] for par in model.Vdata[p]["parents"]]
                        except:
                            continue # until all parents given

                        t_abs[p] = max(parent_t_abs) + seq["dL_"+p] # use only my own preceding temporal variable

                        # get interval to previous event
                        for el in real_seq[p.split("_")[0]]:
                            state = el[0]
                            start_t = el[1]
                            end_t = el[2]

                            # require: all in current state, all within correct time
                            if seq[p] == state and start_t < t_abs[p]+0.0001 and end_t > t_abs[p]+0.0001:
                                interval = t_abs[p] - t_abs[p.split("_")[0] +"_"+ str(int(p.split("_")[1]) - 1)]
                                break


                        rmse = math.sqrt(mean_squared_error([0], [drift]))
                        rte = np.abs(0 - drift) / interval  # bezogen auf Intervallänge des vorgängers im Original

                        rmses += [rmse]
                        rtes += [rte]
                    else:
                        t_abs[p] = 0.0

                    # add children
                    node_set += [c for c in model.Vdata[p]["children"] if
                                 not c in done and not str.startswith(c, "dL_")]

                    # drop parent
                    node_set.remove(p)
                    done += [p]

        # compute average
        return np.mean(rmses), np.mean(rtes)

    def _convert_sequences(self, model, test_sequences, test_tscbn_sequences):
        if model.__class__.__name__ == "DBNDiscrete":
            #try:
            sequences = self.convert_intervals_to_dbn_sequence(test_sequences, model.start_time, model.end_time, model.resolution)
            '''except:
                print("yes it happened")
                import json
                with open('seq.txt', 'w') as outfile:
                    json.dump(test_sequences, outfile)
                with open('det.txt', 'w') as outfile:
                    json.dump([model.start_time, model.end_time, model.resolution], outfile)
                import sys
                print("I love to exit")
                sys.exit()'''


        if model.__class__.__name__ == "TSCBN":
            sequences = test_tscbn_sequences

        return sequences, test_sequences

    def _compute_jpd_n_log(self, model, sequences, test_sequence = None):
        '''
        Insert all testsequences to the model and compute the JPD and log-likelihood
        '''
        if model.__class__.__name__ != "dict":
            print("Computing jpd and log-likelihood // temp-jpd and temp-log-likelihood...")
        else:
            print("Computing log-likelihood...")
        # convert sequences if needed
        if model.__class__.__name__ == "DBNDiscrete":

            # compute jpd of sequencez
            # P(X,Y) = P(X|Y)P(Y)
            initial_set = copy.deepcopy([node for node in model.V if (not model.Vdata[node]["parents"] or model.Vdata[node]["parents"] is None) and not str.startswith(node, "dL_")])
            t_jpds, t_log_likelihoods = [], []
            jpds, log_likelihoods = [], []
            accuracies = []
            nr_total = float(len(model.V))
            nr_total_o = float(len([v for v in model.V if not str.endswith(v, "_0")])) # excludes parents

            for seq in sequences:
                last_state = np.array([seq[i] for i in initial_set])
                i = 0
                jpd = 1.0
                log_likelihood = 0.0
                node_set = initial_set
                done = []
                nr_correct = 0.0
                while node_set:

                    # Get next parent
                    node_set = list(set(node_set))
                    if i >= len(node_set): i = 0
                    p = node_set[i]
                    i += 1

                    # append jpd
                    val = seq[p]
                    if model.Vdata[p]["parents"] is None or not model.Vdata[p]["parents"]:
                        dist = self._smooth_distribution(model.Vdata[p]["cprob"])
                        cpd = dist[model.Vdata[p]["vals"].index(val)]
                        max_idx = False
                    else:
                        condition = str([seq[kk] for kk in model.Vdata[p]["parents"]])
                        dist = self._smooth_distribution(model.Vdata[p]["cprob"][condition])
                        cpd = dist[model.Vdata[p]["vals"].index(val)]
                        max_idx = np.argmax(model.Vdata[p]["cprob"][condition])
                    jpd *= cpd
                    log_likelihood += np.log(cpd)

                    # point estimate
                    if max_idx:
                        point_estimate = model.Vdata[p]["vals"][max_idx]
                        if point_estimate == val: nr_correct += 1

                    # add children
                    node_set += [c for c in model.Vdata[p]["children"] if not c in done]

                    # drop parent
                    node_set.remove(p)
                    done += [p]
                jpds += [jpd]
                log_likelihoods += [log_likelihood/nr_total] # average likelihood over all nodes
                accuracies += [nr_correct/nr_total_o]

        # CTBN WRAPPER
        if model.__class__.__name__ == "dict":
            # average likelihood over all nodes
            # transition matrix: model[nodename]["transition"][condition] -> yields transition matrix
            # transition matrix: model[nodename]["intensity"][condition] -> yields intensity matrix

            # Initialize
            log_likelihoods = []
            rename_dict = model["rename_dict"]

            for seq in test_sequence:
                # Probability of time comes in - distance to previous
                df = CTBNEstimator().sequence_to_dataframe(seq, rename_dict)

                # iterate
                # keeps track of previous state
                last = dict()
                for idx in range(len(df)):
                    cur = dict(df.iloc[0][list(seq.keys())])

                    # per entry compute probabilities -> if no condition given
                    if not last:
                        for node in cur: #initial run assume other nodes keep same state
                            if model[node]["parents"]:
                                condition = str(["%s=%s" % (str(p), str(int(cur[p]))) for p in model[node]["parents"]])
                                log_lh = np.log(model[node]["transition"][condition][int(cur[node])])
                            else:
                                log_lh = np.log(model[node]["transition"][int(cur[node])])
                            log_likelihoods += [log_lh]
                    else:
                        for node in cur: # condition is previous state
                            if model[node]["parents"]:
                                condition = str(["%s=%s" % (str(p), str(int(last[p]))) for p in model[node]["parents"]])
                                log_lh = np.log(model[node]["transition"][condition][int(cur[node])])
                            else:
                                log_lh = np.log(model[node]["transition"][int(cur[node])])
                            log_likelihoods += [log_lh]
                    last = cur
            # smooth to -2 to avoid -inf
            lhs = np.array(log_likelihoods)
            lhs[np.where(lhs == -np.inf)] = -2

            return -1, np.mean(lhs), -1, -1, -1


        if model.__class__.__name__ == "TSCBN":

            initial_set = copy.deepcopy([node for node in model.V if (
            not model.Vdata[node]["parents"] or model.Vdata[node]["parents"] is None)])
            jpds, log_likelihoods, t_jpds, t_log_likelihoods = [], [], [], []
            #cur_times = {} # time of current parent
            accuracies = []
            nr_total = float(len(model.V))
            nr_total_o = float(len([v for v in model.V if not str.endswith(v, "_0")])) # excludes parents
            for seq in sequences: # Probability of time comes in - distance to previous
                i = 0
                jpd = 1.0
                log_likelihood = 0.0
                t_jpd = 1.0
                t_log_likelihood = 0.0
                node_set = initial_set
                done = []
                nr_correct = 0.0
                while node_set:

                    # Get next parent
                    node_set = list(set(node_set))
                    if i >= len(node_set): i = 0
                    p = node_set[i]
                    i += 1

                    # append jpd
                    val = seq[p]
                    if model.Vdata[p]["parents"] is None or not model.Vdata[p]["parents"]:
                        dist = self._smooth_distribution(model.Vdata[p]["cprob"])
                        cpd = dist[model.Vdata[p]["vals"].index(val)]

                        dL_cond = str([seq[kk] for kk in model.Vdata["dL_" + p]["parents"] ])
                        mean, var = model.Vdata["dL_" + p]["hybcprob"][dL_cond]["mean_base"], model.Vdata["dL_" + p]["hybcprob"][dL_cond]["variance"]
                        actual_dL = seq["dL_" + p]
                        dT_cpd = norm(mean, np.sqrt(var)).pdf(actual_dL)/norm(mean, np.sqrt(var)).pdf(mean)
                        max_idx = False
                    else:
                        condition = str([seq[kk] for kk in model.Vdata[p]["parents"] ])
                        dist = self._smooth_distribution(model.Vdata[p]["cprob"][condition])
                        cpd = dist[model.Vdata[p]["vals"].index(val)]
                        dL_cond = str([seq[kk] for kk in model.Vdata["dL_" + p]["parents"] ])
                        mean, var = model.Vdata["dL_" + p]["hybcprob"][dL_cond]["mean_base"], model.Vdata["dL_" + p]["hybcprob"][dL_cond]["variance"]
                        #print(str(mean))
                        actual_dL = seq["dL_" + p]
                        dT_cpd = norm(mean, np.sqrt(var)).pdf(actual_dL)/norm(mean, np.sqrt(var)).pdf(mean)
                        max_idx = np.argmax(model.Vdata[p]["cprob"][condition])

                    if math.isnan(dT_cpd) or math.isinf(dT_cpd) or dT_cpd < 0.0001: dT_cpd = 0.001
                    jpd *= cpd
                    log_likelihood += np.log(cpd)
                    t_jpd *= dT_cpd
                    t_log_likelihood += np.log(dT_cpd)


                    if max_idx:
                        point_estimate = model.Vdata[p]["vals"][max_idx]
                        if point_estimate == val: nr_correct += 1

                    # add children
                    node_set += [c for c in model.Vdata[p]["children"] if not c in done and not str.startswith(c, "dL_")]

                    # drop parent
                    node_set.remove(p)
                    done += [p]


                jpds += [jpd]
                log_likelihoods += [log_likelihood/nr_total]
                t_jpds += [t_jpd]
                t_log_likelihoods += [t_log_likelihood]
                accuracies += [nr_correct/nr_total_o]

        # compute average
        if not t_jpds: m_jpds = "N.A."
        else: m_jpds = np.mean(t_jpds)
        if not t_log_likelihoods: m_t_log_likelihoods = "N.A."
        else: m_t_log_likelihoods = np.mean(t_log_likelihoods)

        return np.mean(jpds), np.mean(log_likelihoods), m_jpds, m_t_log_likelihoods, np.mean(accuracies)

    def _kl_divergence(self, dist, ref_dist, eps = 0.001):

        if len(np.where(dist == 0)[0]) > 0:
            dist[np.where(dist != 0)] = dist[np.where(dist != 0)] - eps / float(
                len(np.where(dist != 0)[0]))
            dist[np.where(dist == 0)] = eps
        if len(np.where(ref_dist == 0)[0]) > 0:
            ref_dist[np.where(ref_dist != 0)] = ref_dist[np.where(ref_dist != 0)] - eps / float(
                len(np.where(ref_dist != 0)[0]))
            ref_dist[np.where(ref_dist == 0)] = eps

        # replace nan
        if np.any(np.isnan(np.log(ref_dist / dist))):
            print("Nan")
        log_out = np.nan_to_num(np.log(ref_dist / dist))
        kl_div = np.sum(ref_dist * log_out)

        return kl_div

    def _smooth_distribution(self, dist, eps = 0.001):
        if len(np.where(dist == 0)[0]) > 0:
            dist[np.where(dist != 0)] = dist[np.where(dist != 0)] - eps / float(
                len(np.where(dist != 0)[0]))
            dist[np.where(dist == 0)] = eps
        return dist

    def convert_intervals_to_dbn_sequence(self, sequences, start_time, end_time, resolution):
        input_sequences = [] # to use in parameter estimation
        i=0
        for sequence in sequences:
            i+=1

            input_sequence = {}
            for tv in sequence:

                cur_sequence = sequence[tv]

                cur_idx, kk = -1, 0

                for cur_time in np.arange(start_time, end_time + 0.00001, resolution):
                    cur_idx += 1
                    cur_node = tv + "_" + str(cur_idx)

                    if cur_time > cur_sequence[-1][-1]:
                        input_sequence[cur_node] = cur_sequence[-1][0]
                        continue

                    seq_part = cur_sequence[kk] # e.g. o1, 0, 10

                    while not cur_time < seq_part[2]:
                        kk += 1
                        seq_part = cur_sequence[kk]
                    input_sequence[cur_node] = seq_part[0]
            #print("%s" % (str(input_sequence)))
            input_sequences.append(input_sequence)
        return input_sequences

    def _given_in(self, in_dict, key_1, key_2):
        if key_1 not in self.tmp: self.tmp[key_1] = {}
        if key_1 in in_dict:
            if key_2 in in_dict[key_1]:
                return True
        return False
