#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import threading
import traceback

import time

from _include.estimator.parameter_estimator import ParameterEstimator
from _include.m_libpgm.graphskeleton import GraphSkeleton
from _include.m_libpgm.pgmlearner import PGMLearner
from scipy.stats import norm
from general.base import Base
import numpy as np
from enum import Enum
import copy
from general.log import Log as L

class DistributionsEnum(Enum):
    UNIFORM = 1

class Constant(object):
    LOCK = threading.Lock()
    LOCK2 = threading.Lock()
    JOE = L()
    Never = 0

class EMAlgorithmSampleTree(Base):


    def __init__(self, sequence, number_nevers, last_symbol, tv_name, len_nodes, histogram_smoothing):
        if number_nevers<0:
            number_nevers=0

        self.delta_t_for_debug = {}

        self._len_nodes = len_nodes
        self._id = str(tv_name+str("-".join([s[0] for s in sequence])))
        self._initial_state = last_symbol
        self.histogram_smoothing = histogram_smoothing

        try:
            self._full_sequence = [self._initial_state] + sequence # IF SEQUENCE only 1 SYMBOL CRASH
        except:
            self._full_sequence = [[self._initial_state, 0.0, 300000]] + sequence

        self.sequence = sequence
        self.number_nevers = number_nevers
        self._last_symbol = last_symbol
        self._symbol_idx = 0
        self._next_symbols = 0
        self._initial = True
        self.reset_data = copy.deepcopy([sequence, number_nevers, last_symbol])
        self._update_idx = 0
        self._prev_symbol = None
        self._overall_index = -1
        self._em_iteration = 0
        self._tv_name = tv_name

        self._seen = [] # all variants of this sequence that were drawn
        self._cur_seen = []

        self._symbol_histograms = [] # will be updated on reset
        self._update_symbol_histograms = []

        self.create_distributions( sequence)
        self.initial_number_nevers = number_nevers

        self.initial_sequence_length = len(self.sequence)

    def create_distributions(self, sequence):
        # self.number_nevers = Anszahl an maximal möglichen Elementen die ich verteilen kann
        # wenn einer auftrat lösche bei allen - 1 outcome

        '''
        symbol histogram is 2 D array of shape
            A [X  0.15  0.1] i.e. ignore X so really it is  A [0.15  0.1]
            B [X  0.1 0.15]                                 B [0.1 0.15]
            C [X  0.35 0.15]                                C [0.35 0.15]

        - draw from this which is the position of the never
        - e.g. draw once:
            A1 == means A=1

        - set all impossible outcomes to 0
            i.e.
            A [0.0   0.0]           A [0.0    0.0]
            B [0.1   0.0]           B [0.22   0.0]
            C [0.35  0.0]/sum   ==  C [0.78   0.0]
            draw B1
        then Sequence is: AABBC
        '''
        if self.number_nevers != 0:
            self._symbol_histograms = np.ones((len(self._full_sequence), self.number_nevers))
            self._symbol_histograms /= np.sum(self._symbol_histograms)

            #if np.any(np.isnan(np.array(self._symbol_histograms))):
            #    raise ArithmeticError("Symbol Histogram contains nan values " + str(self._symbol_histograms))

    def new_iteration(self, first, _debug_time):
        ''' New iteration of the EM Algorithm '''
        self._em_iteration += 1

        L().log.info("\n\nHistogram Updates \n\t\t\t\tTV %s \n\t\t\t\tsequence: %s" % (self._tv_name, str(self._full_sequence)))
        L().log.info("Anzahl Knoten: "+ str(self._len_nodes))

        # Reset histograms
        if len(self._symbol_histograms)==0:  L().log.info("No histograms - as no ambiguity\n")
        for k in range(len(self._symbol_histograms)):
            if self._em_iteration >1:
                self._symbol_histograms += self.histogram_smoothing
                self._symbol_histograms /= np.sum(self._symbol_histograms)
            try:
                if isinstance(self._full_sequence[k][0], list): ll = self._full_sequence[k][0][0]
                else: ll = self._full_sequence[k][0]
                L().log.info("Symbol %s - distribution: %s" % (str(ll), self._symbol_histograms[k]))
            except:
                L().log.error(traceback.format_exc())

        if _debug_time:L().log.info("Check Time out: \n%s" % str(self.delta_t_for_debug))

    def _likelihood(self, tscbn, seen, tv):
        '''
        Computes the likelihood of temporal variables that are within a
        range of index_range using tbn
        idx == indices for which this likelihood holds
        LIKELIHOOD OF WHOLE SEQUENCE!!!!
        '''
        L().log.debug("\n\n----------------------------------------------------------------------")
        idx = [] # list of lists: l[0] == symbol_idx // l[1] == symbol number
        p_tot = 1.0
        i = -1
        symbol_idx = 0
        symbol_nr = -2 #0 means 1, 1 means 2...
        prev_symbol = seen[0][1]
        first = True
        L().log.debug("\n\nSeen: %s" % (str(seen)))
        for s in seen:
            symbol_nr += 1
            i += 1
            cond = str(s[0])
            symbol = s[1]
            n = tv + "_" + str(i)
            L().log.debug("Symbol nr: %s, Symbol index: %s" % (str(symbol_nr), str(symbol_idx)))
            L().log.debug("\n\nprev_symbol: %s, i: %s, cond: %s, symbol %s n: %s" % (str(prev_symbol), str(i), str(cond), str(symbol), str(n)))
            if not s[0] is None:
                p_tot *= tscbn[n]["cprob"][str(cond)][tscbn[n]["vals"].index(symbol)]
                L().log.debug("\n\np_cond: %s" %(str(tscbn[n]["cprob"][str(cond)][tscbn[n]["vals"].index(symbol)])))
            else:
                p_tot *= tscbn[n]["cprob"][tscbn[n]["vals"].index(symbol)]
                L().log.debug("\n\np_: %s" % (str(tscbn[n]["cprob"][tscbn[n]["vals"].index(symbol)])))

            if not first and prev_symbol != symbol:
                idx.append([symbol_idx, symbol_nr])
                symbol_idx += 1
                symbol_nr = -1
                L().log.debug("idx: %s" % str(idx))

            prev_symbol = symbol
            first = False
        symbol_nr += 1
        idx.append([symbol_idx, symbol_nr])
        L().log.debug("idx: %s" % str(idx))
        '''
        for i in range(0, len(pars.keys())):

            n = tv + "_" + str(i)
            if n not in pars: break
            symbol = set_values[n]

            if pars[n]["parents"] is None:
                cond = []
            else:
                cond = [set_values[k] for k in pars[n]["parents"]]

            # get prob given cond
            if not cond:
                p = pars[n]["cprob"][pars[n]["tbn_vals"].index(symbol)]
            else:
                p = pars[n]["cprob"][str(cond)][pars[n]["tbn_vals"].index(symbol)]
            p_tot *= p

        '''
        L().log.debug("\n\n----------------------------------------------------------------------")

        return p_tot, idx

    def _return_border_case(self):
        '''
        Returns symbols in border case
        '''

        # BORDER CASE: Remain in initial state forever or no nevers given
        if len(self.sequence) == 0:
            self.initial_number_nevers = 0
            return True # no histogram needed

        # BORDER CASE: No nevers given
        if self.initial_number_nevers == 0:
            self._last_symbol = self.sequence[0]
            self.sequence = self.sequence[1:]
            return True # no histogram needed
        return False

    def _handle_initial_run(self):
        '''
        Draw initial sample
        '''
        # Get distribution
        fair_dist = self._symbol_histograms[self._symbol_idx]
        dist = fair_dist[:(self.number_nevers + 1)] / np.sum(fair_dist[:(self.number_nevers + 1)])

        # Draw for how long to hold initial state
        self._initial = False
        self._next_symbols = np.random.choice(np.arange(0, self.number_nevers+1), p=dist)# remaining nr of next symbols to hold AFTER THE CURRENT ONE!

        # If no further element
        #if self._next_symbols != 0:
        #    self._next_symbols -= 1
        #    self.number_nevers -= 1

    def _draw_next_elements(self):
        '''
        Draw next element from histogram
        '''
        self._symbol_idx += 1
        fair_dist = self._symbol_histograms[self._symbol_idx]
        dist = fair_dist[:(self.number_nevers + 1)] / np.sum(fair_dist[:(self.number_nevers + 1)])
        self._next_symbols = np.random.choice(np.arange(0, self.number_nevers+1), p=dist) # remaining nr of next symbols to hold AFTER THE CURRENT ONE
        self.number_nevers -= self._next_symbols

    def update_histograms(self, tv, pars, set_values):
        if not self._seen:
            print("Yes that exists")
            return

        #L().log.info("\nProcessing sequence: %s %s" % (self._tv_name, str(self._full_sequence)))

        # reset Array - row: symbol_idx  col: Anzahl nevers (idx 0 means 1 never here)
        self._symbol_histograms = np.zeros(np.shape(self._symbol_histograms))

        # compute likelihood of this outcome
        lhs_sum = 0.0
        to_add = []
        if len(self._symbol_histograms) > 0:
            for seen in self._seen:

                # FORMAL:
                # compute lh - add it to all symbol histogs
                # e.g. 0.03 computed - add it to AA, add it to BBB, add it to C
                # then in the end normalize with sum of lhs
                # 1. get likelihood per seen e.g. seen = cond, AABBBC => lh = 0.03 // seen = cond, ABBBCCCC => lh = 0.08  // sum lhs = 0.11
                #       -> then: P(A=1) = 0.08/0.11   P(A=2) = 0.03/0.11 // P(B=1)=0 P(B=2)=0 P(B=3)=1.0 // P(C=1) = 0.03/0.11 P(C=4) = 0.08/0.11
                lh, idx = self._likelihood(pars, seen, tv)
                lhs_sum += lh
                to_add += [[lh, idx]]

                for ika in idx:
                    if ika[1] > 0:
                        self._symbol_histograms[ika[0], ika[1]-1] += lh

            self._symbol_histograms /= np.sum(self._symbol_histograms)





    def _draw_whole_distribution(self, force = False):

        valid = False
        while not valid:
            histograms = copy.deepcopy(self._symbol_histograms)
            seq = copy.deepcopy(self._full_sequence)
            total, nevers = 0, np.shape(histograms)[1]
            runs = 0
            number_of_symbols = len(seq)

            while total != nevers:
                runs += 1

                row = np.sum(histograms, axis=1)
                col = np.sum(histograms, axis=0)

                row /= np.sum(row)
                col /= np.sum(col)

                i = np.random.choice(np.arange(0, len(row)), p=row)
                j = np.random.choice(np.arange(0, len(col)), p=col)
                """if force:
                    while j == 0:
                        j = np.random.choice(np.arange(0, len(col)), p=col)
                        print("repeat")"""

                # d.h. j+1 nevers sind nun vergeben - kuerze die Sache entspreche
                histograms = histograms[:, :len(col) - j - 1]
                total += (j + 1)

                # Reihe i ist nun verboten
                histograms[i, :] = 0.0
                histograms /= np.sum(histograms)

                seq[i] = [seq[i]] * (j + 2)

                if np.shape(histograms)[1] is 0: break

            # repeat invalid
            if runs > number_of_symbols: valid = False
            else: valid = True

        # make list normal
        res = []
        for k in seq:
            if isinstance(k[0], list): res += k
            else: res += [k]
        return res

    def get_next_symbol(self, parent_starts, parent_outcome, condition):
        '''
            each symbol is followed by a distribution which is optimized
            e.g.
                ABC would have dist[0] = [0,1,2] - depending on number of outcomes
                dist[0] = "Number of A occurrences"
                dist[1] = "Number of B occurrences"
                ...

            Then for each next symbol
                - draw number of next symbols and next symbol
                - until have next symbols return next symbols
                - optimize distributions: i.e. forbid permitted outcomes e.g. auf 4 Stellen - habe shcon AAB dann kann C nimmer 2 sein sondern nur 1
                - can count at the same time what I did draw - e.g. if AABBCCC know next_distribution dist[0] = [0,1,0]
        '''
        self._overall_index += 1


        # ------------------------------------------------------------------------------------
        #   BORDER CASES
        # ------------------------------------------------------------------------------------
        is_border = self._return_border_case()
        if is_border:
            self._cur_seen += [(condition, self._last_symbol[0])]
            return self._last_symbol

        # ------------------------------------------------------------------------------------
        #   Normal Run
        # ------------------------------------------------------------------------------------
        if self._initial:
            # draw whole distribution
            good = False
            while not good:
                try:
                    self._whole_sequence = self._draw_whole_distribution()
                    good = True
                except:
                    pass

            #self._TEST_ORIGINAL = self._whole_sequence
            #print("WHOLE: " + str(self._whole_sequence))
            self._initial = False
            # first element was already passed as initial - thus can remove it here
            self._whole_sequence = self._whole_sequence[1:]


        self._last_symbol = self._whole_sequence[0]
        self._whole_sequence = self._whole_sequence[1:]

        # Check sample invalid
        #try:
        #    self._last_symbol[1][0]
        #    print("\nACHTUNG __________________ " + str(self._last_symbol))
        #    print("ORIGIONAL: "+str(self._TEST_ORIGINAL))
        #    print("FULL: " + str(self._full_sequence))
        #    print("Histos: " + str(self._symbol_histograms))
        #except:
        #    pass
        if not self._satisfies_parent_conditions(parent_starts, parent_outcome):
            Constant.JOE.log.debug("INVALID SAMPLE %s" % str(self._full_sequence))
            return None

        self._cur_seen += [(condition, self._last_symbol[0])]
        return self._last_symbol




        # so hole ich die nächste Symbol

        # ------------------------------------------------------------------------------------
        #   Draw next until None left
        # ------------------------------------------------------------------------------------
        if not self._next_symbols == 0: # Still did not reach last
            self._next_symbols -= 1 # return until done
            self.number_nevers -= 1


            #L().log.debug("%s - I return: %s %s" % (str(self._id), self._next_symbols + 1, str(self._last_symbol)))
            # HERE check if my returned sample is legid!
            if not self._satisfies_parent_conditions(parent_starts, parent_outcome):
                L().log.debug("_______________ RETRY HARD _______________ ")
                # try again - then return not possible
                if len(self.sequence) == 0:  # reached last element - but there is no next element - so not possible
                    return None
                # if this will not work - then return invalid sample
                self._last_symbol = self.sequence[0]
                self.sequence = self.sequence[1:]
                if self.number_nevers > 0:
                    self._draw_next_elements()
                if not self._satisfies_parent_conditions(parent_starts, parent_outcome):
                    return None

            self._cur_seen += [(condition, self._last_symbol[0])]
            return self._last_symbol

        else: # get next
            self._last_symbol = self.sequence[0]
            self.sequence = self.sequence[1:]

        # ------------------------------------------------------------------------------------
        #   Draw next elements
        # ------------------------------------------------------------------------------------
        if self.number_nevers > 0:
            self._draw_next_elements()
        #L().log.debug("%s - I return: %s %s" % (str(self._id), self._next_symbols + 1, str(self._last_symbol)))

        if not self._satisfies_parent_conditions(parent_starts, parent_outcome):
            L().log.debug("_______________ RETRY HARD _______________ ")
            # try again - then return not possible
            if len(self.sequence) == 0:  # reached last element - but there is no next element - so not possible
                return None
            # if this will not work - then return invalid sample
            self._last_symbol = self.sequence[0]
            self.sequence = self.sequence[1:]
            if self.number_nevers > 0:
                self._draw_next_elements()
            if not self._satisfies_parent_conditions(parent_starts, parent_outcome):
                return None


        return self._last_symbol

    def _satisfies_parent_conditions(self, parent_starts, parent_outcome):
        ''' Satisfying means:
            if I am never and parent is not never - require
                - parent_start_time < my_end_time
            if I am not never and parent is not never - require
                -  parent_start_time < my_start_time

            in any other case - require
                - parent_start_time < my_end_time
        '''
        res = []
        for idx in range(len(parent_starts)):
            par_never = parent_starts[idx][0]
            par_start = int(parent_starts[idx][1])
            #par_end = parent_ends[idx][1]
            try:
                me_never = parent_outcome == self._last_symbol[0]
                me_start = float(self._last_symbol[1])
                me_end = float(self._last_symbol[2])
            except:
                raise AssertionError(str(self._last_symbol))

            if me_never and not par_never:
                if par_start < me_end:
                    res.append(True)
                continue

            if (not me_never) and (not par_never):
                if par_start < me_start:
                    res.append(True)
                continue

            if par_start < me_end:
                res.append(True)
                continue

        return np.all(np.array(res))


    def reset(self, initial_states):
        #print("HARDCORE - DISTRUCTION ANGER HATE")
        [self.sequence, self.number_nevers, self._last_symbol] = copy.deepcopy(self.reset_data)
        self._symbol_idx = 0
        self._next_symbols = 0
        self._initial = True

        if self._cur_seen and not tuple(self._cur_seen) in self._seen:
            self._seen += [tuple(self._cur_seen)]
        #self._seen = list(set(tuple(x) for x in self._seen if self._seen.count(x) > 1))
        self._cur_seen = [[None, initial_states[self._tv_name][0]]]

class MLECountingLocalParameterEstimator(ParameterEstimator):
    '''
    Idea:
        ○ Weise zufällige erlaubte Konstellation zu pro Sequence mache MLE
        ○ Weise neue zufällige erlaubte Konstellation zu und mache MLE  v.a. auch easy zu parallelisieren
        ○ Ist dass nicht eine klügere des Monte Carlo Sampling
        ○ Wenn ich also 1000 mal verschiedene Konstellationen mit MLE bestimmt habe bilde Mittelwert davon und fertig
        ○ Oder einfach pro Sequenz eine Konstellation und über die mittle ich - ist praxistauglicher

    '''

    LAST_KL_DIVERGENCE = 0.0

    def test(self):
        import json

        sequences = json.load(open('seq_err.txt'))
        [start_time, end_time, resolution] = json.load(open('seq_det_err.txt'))

        # translate input seqeucens
        input_sequences = self.convert_intervals_to_dbn_sequence(sequences, start_time, end_time, resolution)
        #print(input_sequences)

    def __init__(self):
        super(MLECountingLocalParameterEstimator, self).__init__()

        # MCMC frequency
        self.sampling_frequency = 750
        self.iteration_frequency = 1 # EM Iteration steps
        self._first_iteration = True
        self.histogram_smoothing = 0.01
        self.cpd_smoothing = 0.0
        self._debug_time = False

        # information
        self._show_runs = False

        # parallel
        self._parallel_processes = 30

    def set_parallel_processes(self, number):
        self._parallel_processes = number

    def estimateParameter(self, sequences, model, debug = False, evaluator= False, reference =None ):
        self._evaluator = evaluator
        self._reference = reference
        '''
        This approach uses MCMC simulation to fill the CPD tables for the model sequences[0]["V0"]
        ACHTUNG: Sample bis jetzt nicht mit Verteilung in Abhängigkeit von etwas sondern
        '''
        # 1. wenn keine nevers enthalten sind -> dann: zähle einfach hoch
        # 2. wenn nevers enthalten sind -> dann muss nevers verteilen

        if model == "TSCBNStructureModel" or model == "TSCBNSimpleStructureModel":
            return self._estimate_tscbn(sequences, debug)
        elif model == "DBNStructureModel":
            return  self._estimate_dbn(sequences, debug)

    def _estimate_dbn(self, sequences, debug):

        # translate input seqeucens
        input_sequences = self.convert_intervals_to_dbn_sequence(sequences, self.tbn.start_time, self.tbn.end_time, self.tbn.resolution)

        # learn parameters
        learner = PGMLearner()
        try:
            result = learner.discrete_mle_estimateparams(self.tbn.skeleton, input_sequences)
        except KeyError:
            traceback.print_exc()
            input_sequences = self.convert_intervals_to_dbn_sequence(sequences, self.tbn.start_time, self.tbn.end_time, self.tbn.resolution)
            result = learner.discrete_mle_estimateparams(self.tbn.skeleton, input_sequences)

            import sys
            import json
            print("Error - storing data")
            with open('seq_err.txt', 'w') as outfile:
                json.dump(sequences, outfile)
            with open('seq_det_err.txt', 'w') as outfile:
                json.dump([self.tbn.start_time, self.tbn.end_time, self.tbn.resolution], outfile)
            print("Store")
            traceback.print_exc()
            sys.exit(0)

        return result

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

    def _log_cpds(self):
        L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
        L().log.info("     New CPDs")
        L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
        for n in self.tbn.Vdata:
            if str.startswith(n, "dL_"): continue
            if isinstance(self.tbn.Vdata[n]["cprob"], dict):
                for k in self.tbn.Vdata[n]["cprob"]:
                    L().log.info("%s | %s = %s" % (n, k, str(self.tbn.Vdata[n]["cprob"][k])))
            else:
                L().log.info("%s = %s" % (n, str(self.tbn.Vdata[n]["cprob"])))
        L().log.info("\n\n")

    def _estimate_tscbn(self, sequences, debug):
        repeat_frequency = self.sampling_frequency

        # 1. per sequence draw 100 (?) valid combinations
        per_seq_trees, per_seq_initial_states = self._extract_sample_trees(sequences)

        # clean skeleton
        new_E = []
        for ed in self.tbn.skeleton.E:
            if not str.startswith(ed[0], "dL_") and not str.startswith(ed[1], "dL_"):
                new_E.append(ed)
        new_V = []
        for ver in self.tbn.skeleton.V:
            if not str.startswith(ver, "dL_"):
                new_V.append(ver)
        skel = GraphSkeleton()
        skel.E = new_E
        skel.V = new_V

        #print("todo: DO THIS ONLY IF SEQUENCE IS NOT COMPLETE")
        ping = time.clock()
        delta_t_distribution = {}
        for i in range(repeat_frequency):
            if i % 50 == 0:
                print(str(i) + "/" + str(self.sampling_frequency))

            # this can be made parallel
            for seq_count in range(len(sequences)):
                try:
                    trees = per_seq_trees[seq_count]
                except:
                    continue # is ok as this sequence is probably invalid
                initial_states = per_seq_initial_states[seq_count]

                # draw valid sample - output = V0_0 = ... dL_...
                seq, delta_t_distribution = self.draw_valid_sample(initial_states, trees, delta_t_distribution)

        print("Estimation %s" % str(time.clock() - ping))

        # normalize
        for v in self.tbn.V:
            if str.startswith(v, "dL_"): continue
            if isinstance(self.tbn.Vdata[v]["cprob"], dict):
                for k in self.tbn.Vdata[v]["cprob"]:
                    self.tbn.Vdata[v]["cprob"][k] /= np.sum(self.tbn.Vdata[v]["cprob"][k])
            else:
                s = np.sum(self.tbn.Vdata[v]["cprob"]) # uniform if not seen
                if s == 0: self.tbn.Vdata[v]["cprob"]  = np.ones(len(self.tbn.Vdata[v]["cprob"]))/float(len(self.tbn.Vdata[v]["cprob"]))
                else: self.tbn.Vdata[v]["cprob"] /= s

        # compute time
        # do norm fit on last run then simply aggregate all gaussians - HALT - das muss conditioned passieren
        for k in delta_t_distribution:
            for j in delta_t_distribution[k]:
                mean, std = norm.fit(delta_t_distribution[k][j])
                var = std * std
                if var == 0: var = 0.02 # else it makes no sense - as everything else then exact value is zero
                mean_scale = [1] * len(self.tbn.Vdata[k]["parents"])
                self.tbn.Vdata[k]["hybcprob"][j] = {'variance': var, 'mean_base': mean, 'mean_scal': mean_scale}

        return self.tbn

    def draw_valid_sample(self, initial_states, trees, delta_t_distribution):

        initial_set = [n for n in self.tbn.nodes.keys() if self.tbn.Vdata[n]["parents"] == None]
        for t in trees: trees[t].reset(initial_states)
        node_set = copy.deepcopy(initial_set)
        parents_set, i = [], 0
        current_sample, sample_legid, t_abs, t_abs_end = [], True, {}, {}
        pars = dict()
        done = []
        sequence = dict()
        #delta_t_distribution = {}

        while node_set:

            # 1. next node
            i, n = self._next_node(node_set, i)

            # 2. bla
            if n not in pars:
                par = {}
                par["parents"] = self.tbn.Vdata[n]["parents"]
                par["dL_parents"] = self.tbn.Vdata["dL_" + n]["parents"]
                par["tbn_vals"] = self.tbn.Vdata[n]["vals"]
                par["children"] = self.tbn.Vdata[n]["children"]
                par["cprob"] = self.tbn.Vdata[n]["cprob"]
                pars[n] = par

            # 3. if initial states - draw it from there
            if n.split("_")[-1] == "0":

                # DRAW LEAF NODE INITIAL SAMPLE "_".join(n.split("_")[:-1]
                "_".join(n.split("_")[:-1])
                val = initial_states["_".join(n.split("_")[:-1])][0]
                delta_t_distribution["dL_" + n] = {}
                t_abs[n], t_abs_end[n], sequence[n] = 0.0, initial_states["_".join(n.split("_")[:-1])][2], val
                delta_t_distribution["dL_" + n][str([val])] = [0.0]

                # record state
                idx = self.tbn.Vdata[n]["vals"].index(val)
                #print(self.tbn.Vdata[n]["cprob"])
                self.tbn.Vdata[n]["cprob"][idx] += 1.0


                #print("%s, %s: %s" % (str(n), str(str([val])), str(np.mean(delta_t_distribution["dL_" + n][str([val])]))))

            else:

                # 4. if not initial states - draw conditioned on parents
                # check if all parents given - else continue
                if not set(pars[n]["parents"]).issubset(parents_set):
                    i += 1
                    continue

                # get conditions
                cond = [sequence[k] for k in pars[n]["parents"]]

                # DRAW AND STORE NEXT SYMBOL
                parent_starts = [ [self._is_never(k, sequence), t_abs[k]] for k in pars[n]["parents"]]
                val = trees["_".join(n.split("_")[:-1])].get_next_symbol(parent_starts, self._parent_outcome(n, sequence), cond)
                if val is None:
                    print("Sample not legit")
                    break

                # direct recording at seen index
                idx = self.tbn.Vdata[n]["vals"].index(val[0])
                self.tbn.Vdata[n]["cprob"][str(cond)][idx] += 1.0

                sequence[n] = val[0]
                t_abs[n] = val[1]
                t_abs_end[n] = val[2]

                # add time
                cond_dL = [sequence[k] for k in pars[n]["dL_parents"]] # [set_values[k] for k in self.tbn.Vdata["dL_" + n]["parents"]]
                if "dL_" + n  not in delta_t_distribution:
                    delta_t_distribution["dL_" + n] = {}
                if not str(cond_dL) in delta_t_distribution["dL_" + n]:
                    delta_t_distribution["dL_" + n][str(cond_dL)] = []  # Summe, Anzahl
                delta_t_distribution["dL_" + n][str(cond_dL)] += [max([0.0, t_abs[n] - max([t_abs[k] for k in pars[n]["parents"]])])]

                print("%s, %s: %s" % (str(n), str(str(cond_dL)), str(np.mean(delta_t_distribution["dL_" + n][str(cond_dL)]))))

            # GET NEXT NODES
            parents_set.append(n)
            node_set.remove(n)
            done += [n]
            node_set += [o for o in pars[n]["children"] if not o in done and not str.startswith(o, "dL_")]
            node_set = list(set(node_set))

        return sequence, delta_t_distribution

    def _update_CPD(self, output_list, per_seq_trees, final = True):
        # set
        for output in output_list:
            results = output[0]
            delta_t_distribution = output[1]
            trees = output[2]
            seq_count = output[3]
            per_seq_trees[seq_count] = trees

            for r in results:
                for st in r[0]:
                    self.tbn.Vdata[st[0]]["cprob"][st[1]] += 1 # prev[st[0]]["cprob"][st[1]]
                for st in r[1]:
                    self.tbn.Vdata[st[0]]["cprob"][st[1]][st[2]] += 1 # prev[st[0]]["cprob"][st[1]][st[2]]

        # normalize
        if final:
            #print("Final run done!")
            for node in self.tbn.Vdata:
                if "hybcprob" in self.tbn.Vdata[node]:
                    for k in self.tbn.Vdata[node]["hybcprob"]:
                        all_means, all_vars = [], []
                        for output in output_list:
                            if k in output[1][node]:
                                all_means += [output[1][node][k]["mean_base"]]
                                all_vars += [output[1][node][k]["variance"]]

                        if not all_means: all_means = [0]
                        if not all_vars: all_vars = [0]
                        self.tbn.Vdata[node]["hybcprob"][k]["mean_base"] = np.mean(all_means)
                        self.tbn.Vdata[node]["hybcprob"][k]["variance"] = np.mean(all_vars)

                if "cprob" in self.tbn.Vdata[node]:
                    if isinstance(self.tbn.Vdata[node]["cprob"], dict):
                        for k in self.tbn.Vdata[node]["cprob"]:
                            self.tbn.Vdata[node]["cprob"][k] = self.tbn.Vdata[node]["cprob"][k]/ sum(self.tbn.Vdata[node]["cprob"][k])

                            # smoothing
                            self.tbn.Vdata[node]["cprob"][k] += self.cpd_smoothing
                            self.tbn.Vdata[node]["cprob"][k] = self.tbn.Vdata[node]["cprob"][k] / sum(self.tbn.Vdata[node]["cprob"][k])

                    else:
                        self.tbn.Vdata[node]["cprob"] = self.tbn.Vdata[node]["cprob"]/sum(self.tbn.Vdata[node]["cprob"])
                        self.tbn.Vdata[node]["cprob"] += self.cpd_smoothing
                        self.tbn.Vdata[node]["cprob"] = self.tbn.Vdata[node]["cprob"] / sum(self.tbn.Vdata[node]["cprob"])


            # per tree also update histograms - needs to be done seperately - but is parallelizable
            # output_list[2] are the trees - that need to update their histogram
            #print("Updating histograms ")
            for seq_count in per_seq_trees: # trees = output[2]
                trees = per_seq_trees[seq_count]

                # each output is one sequence !
                for tv in trees:
                    #[n, str(cond), pars[n]["tbn_vals"].index(val[0])]
                    trees[tv].update_histograms(tv, self.tbn.Vdata, results)

    def _update_trees(self, output_list, per_seq_trees):
        ''' class id sollte immernoch gleich sein'''
        for output in output_list:
            per_seq_trees[output[3]] = output[2]
        return  per_seq_trees

    def _extract_sample_trees(self, sequences):

        seq_count = -1
        per_seq_trees = {}
        per_seq_initial_states = {}
        #print("Generate Sample generator")
        invalid_sequences = []

        for tv_sequence in sequences:
            seq_count += 1

            # erzeuge pro Temporale Variable never Verteilung
            trees = {}
            initial_states = {}
            rest_sequence = {}
            rel_nodes = {}


            invalid = False
            for n in list(self.tbn.nodes.keys()):
                tv = "_".join(n.split("_")[:-1])

                if not tv in trees:
                    if str.startswith(tv, "dL"): continue
                    # get initial state
                    try:
                        initial_states[tv] = tv_sequence[tv][0]
                    except:
                        invalid_sequences.append(tv_sequence)
                        invalid = True
                        break


                    # get rest of sequence
                    rest_sequence[tv] = tv_sequence[tv][1:]
                    rel_nodes[tv] = [n for n in list(self.tbn.nodes.keys()) if "_".join(n.split("_")[:-1]) == tv][1:] # ACHTUNG V1 includes V11, V12 ... NO!

                    # create tree for remaining sequence EMWholeSeqAlgorithmSampleTree(rest_sequence[tv], len(rel_nodes[tv]) - len(rest_sequence[tv]), initial_states[tv], tv, len(rel_nodes[tv]))
                    try:
                        trees[tv] =  EMAlgorithmSampleTree(rest_sequence[tv], len(rel_nodes[tv]) - len(rest_sequence[tv]), initial_states[tv], tv, len(rel_nodes[tv]), self.histogram_smoothing)
                    except ValueError as r:
                        print("Error: %s" % str(r))

            if invalid: continue
            # store for all iterations
            per_seq_trees[seq_count] = trees
            per_seq_initial_states[seq_count] = initial_states
            #L().log.debug("DETERMINE INTIAL STATE FOR SEQUENCE %s - is %s" % (str(seq_count), str(initial_states)))

        # INVALID SEQUENCES REMOVED
        print("Found %s invalid sequences" % str(len(invalid_sequences)))


        return  per_seq_trees, per_seq_initial_states

    def _next_node(self, node_set, i):
        # Get next node
        if i >= len(node_set): i = 0
        try:
            n = node_set[i]
        except:
            i, n = 0, node_set[i]
            print("Potential Problem")
        return i, n

    def _single_run(self, initial_states, trees, seq_count, len_sequences, debug, disable_out = True):
        '''
            This function is used to process multiple sequences together
        '''

        # get last state
        #L().log.debug("-----------------> SEQUENCE %s of %s" % (str(seq_count + 1), str(len_sequences)))
        results = []
        delta_t_distribution = {} # save key: node - value: dict: key - condition  (inkl. myself) value: list of given delta t

        # --------- SAMPLING -----------
        pars = {}
        Constant.LOCK.acquire()
        initial_set = [n for n in self.tbn.nodes.keys() if self.tbn.Vdata[n]["parents"] == None]
        Constant.LOCK.release()
        for tz in range(self.sampling_frequency):

            # Initialize
            #if debug: L().log.debug("Sequence %s - Run %s/%s" % (str(seq_count), str(tz + 1), str(self.sampling_frequency)))
            for t in trees: trees[t].reset(initial_states)
            #Constant.LOCK.acquire()
            node_set = copy.deepcopy(initial_set)#[n for n in self.tbn.nodes.keys() if self.tbn.Vdata[n]["parents"] == None]
            #Constant.LOCK.release()
            parents_set, set_values, i, current_sample_initial = [], {}, 0, []
            current_sample, sample_legid, t_abs, t_abs_end = [], True, {}, {}

            # Iterate tree - starting from parent

            done = []
            while node_set:

                # 1. next node
                i, n = self._next_node(node_set, i)

                # 2. copy parent information - to omit parallel access
                if n not in pars:
                    Constant.LOCK.acquire()
                    par = {}
                    par["parents"] = copy.deepcopy(self.tbn.Vdata[n]["parents"])
                    par["dL_parents"] = copy.deepcopy(self.tbn.Vdata["dL_" + n]["parents"])
                    par["tbn_vals"] = copy.deepcopy(self.tbn.Vdata[n]["vals"])
                    par["children"] = copy.deepcopy(self.tbn.Vdata[n]["children"])
                    par["cprob"] = copy.deepcopy(self.tbn.Vdata[n]["cprob"])
                    pars[n] = par
                    Constant.LOCK.release()

                # 3. if initial states - draw it from there
                if n.split("_")[-1] == "0":

                    # DRAW LEAF NODE INITIAL SAMPLE
                    val = initial_states["_".join(n.split("_")[:-1])][0] #L().log.debug("%s - I return: %s " % (str(n), str(val)))
                    current_sample_initial.append([n, pars[n]["tbn_vals"].index(val)])  # info, info
                    delta_t_distribution["dL_" + n] = {}
                    if self._debug_time: trees["_".join(n.split("_")[:-1])].delta_t_for_debug["dL_" + n] = {}
                    if self._debug_time: trees["_".join(n.split("_")[:-1])].delta_t_for_debug["dL_" + n][str([val])] = 0

                    t_abs[n], t_abs_end[n], delta_t_distribution["dL_" + n][str([val])], set_values[n] = 0.0, initial_states["_".join(n.split("_")[:-1])][2], [0.0], val

                else:

                    # 4. if not initial states - draw conditioned on parents
                    # check if all parents given - else continue
                    if not set(pars[n]["parents"]).issubset(parents_set):
                        i += 1
                        continue

                    # get conditions
                    cond = [set_values[k] for k in pars[n]["parents"]]

                    # DRAW AND STORE NEXT SYMBOL
                    parent_starts = [ [self._is_never(k, set_values), t_abs[k]] for k in pars[n]["parents"]]
                    #parent_ends = [ [self._is_never(k, set_values), t_abs_end[k]] for k in pars[n]["parents"]]
                    val = trees["_".join(n.split("_")[:-1])].get_next_symbol(parent_starts, self._parent_outcome(n, set_values), cond)


                    if val is None:
                        if debug: L().log.debug("Sample NOT LEGID - None - BREAK")
                        print("Sample not legit")
                        break

                    set_values[n] = val[0]
                    t_abs[n] = val[1]
                    t_abs_end[n] = val[2]

                    # IF DRAWN SAMPLE LEGIT RECORD IT
                    current_sample.append([n, str(cond), pars[n]["tbn_vals"].index(val[0])])
                    if debug: L().log.debug("NEXT: %s = %s" % (str(n), val[0]))
                    if debug: L().log.debug("nodes: %s" % str(node_set))

                    # RECORD DELTA T DISTRIBUTION
                    cond_dL = [set_values[k] for k in pars[n]["dL_parents"]] # [set_values[k] for k in self.tbn.Vdata["dL_" + n]["parents"]]


                    # DEBUG - ONLY HERE!!!!!!!!!
                    if self._debug_time:
                        if "dL_" + n  not in trees["_".join(n.split("_")[:-1])].delta_t_for_debug:
                            trees["_".join(n.split("_")[:-1])].delta_t_for_debug["dL_" + n] = {}
                        if not str(cond_dL) in trees["_".join(n.split("_")[:-1])].delta_t_for_debug["dL_" + n]:
                            trees["_".join(n.split("_")[:-1])].delta_t_for_debug["dL_" + n][str(cond_dL)] = []  # Summe, Anzahl
                    #trees[n.split("_")[0]].delta_t_for_debug["dL_" + n][str(cond_dL)] += [t_abs[n] - max([t_abs[k] for k in pars[n]["parents"]])]

                    # END DEBUG

                    if "dL_" + n  not in delta_t_distribution:
                        delta_t_distribution["dL_" + n] = {}
                    if not str(cond_dL) in delta_t_distribution["dL_" + n]:
                        delta_t_distribution["dL_" + n][str(cond_dL)] = []  # Summe, Anzahl
                    delta_t_distribution["dL_" + n][str(cond_dL)] += [t_abs[n] - max([t_abs[k] for k in pars[n]["parents"]])]

                # GET NEXT NODES
                parents_set.append(n)
                node_set.remove(n)
                done += [n]
                node_set += [o for o in pars[n]["children"] if not o in done and not str.startswith(o, "dL_")]
                node_set = list(set(node_set))


            results.append([current_sample_initial, current_sample])

        # do norm fit on last run then simply aggregate all gaussians - HALT - das muss conditioned passieren
        for k in delta_t_distribution:
            for j in delta_t_distribution[k]:
                mean, std = norm.fit(delta_t_distribution[k][j])
                var = std * std
                if var == 0: var = 0.02 # else it makes no sense - as everything else then exact value is zero
                mean_scale = [1] * len(self.tbn.Vdata[k]["parents"])
                delta_t_distribution[k][j] = {'variance': var, 'mean_base': mean, 'mean_scal': mean_scale}

                if self._debug_time: trees["_".join(k.replace("dL_", "").split("_")[:-1])].delta_t_for_debug[k][j] = {'variance': var, 'mean_base': mean}


        return results, delta_t_distribution, trees, seq_count

    def _is_never(self, node_name, set_vals):
        parent_idx = int(node_name.split("_")[-1])-1
        if parent_idx < 0: return  False # then I am 0

        # check if identical to that of node
        if np.all(np.array(set_vals[node_name]) == np.array(set_vals["_".join(node_name.split("_")[:-1])+"_"+str(parent_idx)])):
            return True
        return False

    def _parent_outcome(self, node_name, set_vals):
        parent_idx = int(node_name.split("_")[-1]) - 1
        return  set_vals["_".join(node_name.split("_")[:-1])+"_"+str(parent_idx)]

    def _count_state_up(self, nodes, temporal_variable, initial_state, other_states):
        '''
        If number of state equal to number of intervals, simply count its value count up given its condition

        SHOOT - DAS MACHT NUR SINN PRO TEMPORALE VARIABLE DIE AUCH SELBE LÄNGE HAT
                -> BZW: eigentlich wenn ich MC Simulation mache und sample dann sind fixed Variablen ja immer fixed
                        d.h. die Wahrscheinlichkeit würde an den Punkten eh immer den gleichen Wert an der Stelle geben
                        e.g. wenn ich conditione auf A=a und ich mach MCMC dann würde ich ja immer A=a abfragen

                        - Als pot. Optimierung
        '''
        # set initial state
        idx = self.tbn.Vdata[nodes[0]]["vals"].index(initial_state[0])
        self.tbn.Vdata[nodes[0]]["cprob"][idx] += 1.0

        # get conditions per next state
        i = 0
        for s in other_states:
            i += 1
            node = nodes[i]

            # Problem: Was ist wenn ich auf etwas condition mache wo never drauf verteiltt wird -> muss hier auch MCMC
            #          Simulation machen

            # get parents and get values of those

            condition = [initial_state[0]] + [a[0] for a in other_states]
            for s in other_states:
                a = 0

        # add to the one that has those conditions

    def _data_from_interval(self, time_measured, node_name):
        '''
        Gedanke: Lösche die Liste sofort nachdem ich aufgetreten bin und zähle dafür bei der Verteilung "eins dazu" für
        diese conditoin
        '''
        try:
            self.mean_values[node_name].append(time_measured)
        except:
            self.mean_values[node_name] = [time_measured]

        #raise ArithmeticError("Interval out of range %s, %s"%(str(el), str(intervals)))

