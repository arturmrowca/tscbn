#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import threading
import traceback

from _include.m_libpgm.pgmlearner import PGMLearner
from scipy.stats import norm

from _include.toolkit import parallelize_stuff, PNT
from general.base import Base
from _include.estimator.parameter_estimator import ParameterEstimator

import numpy as np
from enum import Enum
import copy
from general.log import Log as L
from collections import Counter

class TrivialCounterForBNEstimator(ParameterEstimator):
    '''
    Trivial algorithm that simply counts occurrences
    '''

    def __init__(self):
        super(TrivialCounterForBNEstimator, self).__init__()

    def estimateParameter(self, sequences, model, debug = False, evaluator= False, reference =None, opt1=[], opt2 = 'Aktiv_Funktion_Fahrerassistenzsystem_LDM'):
        print("Chose %s as target Variable" % str(opt2))

        if model == "TSCBNStructureModel" or model == "TSCBNSimpleStructureModel":
            return self._estimate_tscbn(sequences, debug, opt1, target=opt2)
        elif model == "DBNStructureModel":
            return  self._estimate_dbn(sequences, debug)

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

    def _log_cpds_emph_given(self, leaves):
        L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
        L().log.info("     New CPDs")
        L().log.info( "---------------------------------------------------------------------------------------------------------------------------------------------------")
        for n in self.tbn.Vdata:
            if str.startswith(n, "dL_"): continue
            #print(str(leaves))
            if not n in leaves:
                #print("Ignoring non leaves! As paths are only valid if they end at a leave")
                continue
            if isinstance(self.tbn.Vdata[n]["cprob"], dict):
                L().log.info("\n\n")

                for k in self.tbn.Vdata[n]["cprob"]:
                    L().log.info("\n\n\t------- Case: %s = %s \n\t\t\tVals: %s------- \n\t\t\tConditions:" % (n, str(self.tbn.Vdata[n]["cprob"][k]), self.tbn.Vdata[n]["vals"]))
                    con = eval(k)
                    remember = [(n, str(list(np.array(self.tbn.Vdata[n]["vals"])[self.tbn.Vdata[n]["cprob"][k] != 0])))]
                    tmp = dict()
                    for i in range(len(self.tbn.Vdata[n]["parents"])):
                        if con[i] == "Never":continue
                        tmp[self.tbn.Vdata[n]["parents"][i]] = con[i]
                        if not str.endswith(self.tbn.Vdata[n]["parents"][i], "_0"):
                            remember+=[(self.tbn.Vdata[n]["parents"][i], con[i])]
                            continue
                        L().log.info("\t\t%s = %s" % (self.tbn.Vdata[n]["parents"][i], con[i]))
                    remember.sort()
                    L().log.info("\n\t\t\tWhat happened:")
                    for r in remember:
                        prev_tv = "_".join(r[0].split("_")[:-1] + [str(int(r[0].split("_")[-1])-1)])
                        if prev_tv[0] == "_":prev_tv=prev_tv[1:]
                        comes_from = tmp[prev_tv]

                        L().log.info("\t\t%s = %s (prev: %s)" % (r[0], r[1], comes_from))
            else:
                L().log.info("\n\n%s = %s" % (n, str(self.tbn.Vdata[n]["cprob"])))
        L().log.info("\n\n")



    def _estimate_tscbn(self, sequences, debug, leaves, target = 'Aktiv_Funktion_Fahrerassistenzsystem_LDM'):
        cnt_s = 0
        tot_s = len(sequences)

        for sequence in sequences:
            if cnt_s % 50 == 0:
                L().log.info("Processing %s / %s" % (str(cnt_s), str(tot_s)))
            cnt_s += 1
            cur_seq ={}
            # simply count that
            largest = None
            max_val = 0
            for tv in sequence:
                i = 0
                for lst in sequence[tv]:
                    [state, start, end] = lst
                    node_name = tv + "_" + str(i)
                    if start > max_val and tv != target:
                        max_val=start
                        largest = node_name
                    i += 1
                    cur_seq[node_name] = state
            # das älteste element in Sequence muss unterstrich kriegen _...
            if largest == None:largest=target+"_1"
            cur_seq["_"+largest] = cur_seq[largest]
            del cur_seq[largest]

            # count all up in tree
            for node in cur_seq:
                if not self.tbn.Vdata[node]["parents"] is None:
                    o = list(set(list(self.tbn.Vdata[node]["parents"])))
                    o.sort()
                    self.tbn.Vdata[node]["parents"] = o

                state = cur_seq[node]

                if self.tbn.Vdata[node]["parents"] is None:
                    idx = self.tbn.Vdata[node]["vals"].index(state)
                    if not "cprob" in self.tbn.Vdata[node]:
                        self.tbn.Vdata[node]["vals"] += ["Never"]
                        self.tbn.Vdata[node]["cprob"] = np.zeros(len(self.tbn.Vdata[node]["vals"]))
                    self.tbn.Vdata[node]["cprob"][idx] += 1.0

                else:
                    # get condition
                    cond = []
                    for p in self.tbn.Vdata[node]["parents"]:
                        if p not in cur_seq:
                            cond += ["Never"] # it did not occur
                        else:
                            cond += [cur_seq[p]]
                    idx = self.tbn.Vdata[node]["vals"].index(state)
                    
                    if not "cprob" in self.tbn.Vdata[node]:
                        self.tbn.Vdata[node]["vals"] += ["Never"]
                        self.tbn.Vdata[node]["cprob"] = dict()
                    if not str(cond) in self.tbn.Vdata[node]["cprob"]:
                        self.tbn.Vdata[node]["cprob"][str(cond)] = np.zeros(len(self.tbn.Vdata[node]["vals"]))
                    
                    self.tbn.Vdata[node]["cprob"][str(cond)][idx] += 1

        # drop not existing cpds:
        for node in self.tbn.Vdata:

            if not self.tbn.Vdata[node]["parents"] is None and not str.startswith(node, "dL_"):
                keep = dict()
                for cond in self.tbn.Vdata[node]["cprob"]:
                    if not np.all(self.tbn.Vdata[node]["cprob"][cond] == 0):
                        keep[cond] = self.tbn.Vdata[node]["cprob"][cond]
                self.tbn.Vdata[node]["cprob"] = keep

        # Plot all distributions
        if self.tbn.show_plot_generated:
            self._visual.plot_histograms_from_bn(self.tbn, self.tbn)
        self._log_cpds_emph_given(leaves)
