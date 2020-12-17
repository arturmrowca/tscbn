from _include.bayspec.spec_mining.mining.metric_based_miner import MBMiner
from _include.bayspec.spec_mining.mining_graph.max_avg_mg import MaxAverageMiningGraph
from _include.discoverer.pc_tree_discoverer import PCTreeDiscoverer
from _include.discoverer.tree_discoverer import structure_consistency, create_tscbn
from _include.estimator.em_algorithm_tscbn_estimator import EMAlgorithmParameterEstimator
from _include.evaluation.cs_parameter_evaluator import CSParameterEvaluator
from _include.structure.case_study.hb_model import HBModel
from sklearn.model_selection import train_test_split
import dill
import os
from _include.structure.case_study.ind_model import IndModel
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from general.setup import create_estimator

np.seterr(divide='ignore', invalid='ignore')

def sanitize_commas(tscbn):
    rep_char = "#"
    # sanitize network -> , forbidden here
    for n in tscbn.V:
        try:
            tscbn.Vdata[n]["vals"] = [v.replace(",", rep_char) for v in tscbn.Vdata[n]["vals"]]
        except:
            pass

        if "cprob" in tscbn.Vdata[n] and isinstance(tscbn.Vdata[n]["cprob"], dict):
            cprob_new = {}
            for c in tscbn.Vdata[n]["cprob"]:
                cprob_new[str([x.replace(",", rep_char) for x in eval(c)])] = tscbn.Vdata[n]["cprob"][c]
            tscbn.Vdata[n]["cprob"] = cprob_new

        if "hybcprob" in tscbn.Vdata[n] and isinstance(tscbn.Vdata[n]["hybcprob"], dict):
            cprob_new = {}
            for c in tscbn.Vdata[n]["hybcprob"]:
                cprob_new[str([x.replace(",", rep_char) for x in eval(c)])] = tscbn.Vdata[n]["hybcprob"][c]
            tscbn.Vdata[n]["hybcprob"] = cprob_new

    return tscbn


def filter_longer_one_n_shift(sequences, shift_map):
    """
    Removes sequences that have only length 0 as those are meaningless
    :param sequences:
    :return:
    """
    res_sequences = list()
    for seq in sequences:
        bad = True
        for tv in seq:
            # shift time
            if tv in shift_map:
                for s in seq[tv]:
                    if s[1] != 0.0:
                        s[1] += shift_map[tv]
                        s[2] += shift_map[tv]

            if len(seq[tv]) != 1:
                bad = False

        if not bad:
            res_sequences.append(seq)

    return res_sequences


def to_dataframe(sequences):
    """
    This method transforms the given sequences into a pandas Dataframe.
    Also sequences that do not start at zero are extended with its inverse state
    :param sequences: Input sequences for discovery algorithms
    :return: Dataframe with one row per sequence entry, states_dict: Dictionary of states per TV
    """
    rows = []
    idx = -1
    states_dict = dict()
    for sequence in sequences:
        idx += 1
        for tv in sequence:
            if not tv in states_dict:states_dict[tv] = []
            tv_sequence = sequence[tv]

            #normalize to have a first element
            if tv_sequence[0][1] != 0.0:
                if len(tv_sequence) > 1:
                    state = tv_sequence[1][0]
                else:
                    state = "def" # default state
                tv_sequence.insert(0, [state, 0.0, tv_sequence[0][1]])

            for state in tv_sequence:
                if not state[0] in states_dict[tv]: states_dict[tv].append(state[0])
                rows += [[idx, tv, state[0], float(state[1]), float(state[2])]]
    df = pd.DataFrame(rows, columns = ["index", "tv", "state", "start_time", "end_time"])

    return df, sequences, states_dict


def run_case_hb(parallel_proesses, plot_model, estimators, show_plot = False):

    # Discover structure
    hb = HBModel()
    tscbn = hb.generate_model()
    tscbn.show_plot_generated = show_plot
    if plot_model: tscbn.draw("ext") # show the model, else set false

    # Extract sequences
    input_csv = "store/case_study_data/sequences_hb.csv"
    sequences = hb.extract_signals(input_csv)

    all_res = {}
    all_res["jpd"] = []
    all_res["log-likelihood"] = []
    all_res["temp-log-likelihood"] = []
    all_res["temp-jpd"] = []
    print("This real world data set is not publicly available and thus, was pseudonymised in terms of names and values. \nIt shall only be used to verify the results of the evaluation.")

    for estimator_id in estimators:
        for _ in range(10): # cross validation 10

            # split
            train_sequence, test_sequence = train_test_split(sequences, test_size=0.2)

            # Learn parameters from data
            pe = create_estimator(estimator_id)
            pe.histogram_smoothing = 0.0
            pe.cpd_smoothing = 0.1
            pe.sampling_frequency = 1000  # sampling frequency for the MC MC Simulation - too low?
            pe.iteration_frequency = 5  # EM Iterations
            pe.set_parallel_processes(parallel_proesses)
            pe_debug_mode = False
            pe.tbn = tscbn
            pe.tbn.clear_parameters()
            pe.estimateParameter(train_sequence, "TSCBNStructureModel", pe_debug_mode)

            # Evaluate results
            ev = CSParameterEvaluator(False)
            ev.add_metric("jpd")
            ev.add_metric("log-likelihood")
            ev.add_metric("temp-log-likelihood")
            ev.add_metric("temp-jpd")
            eval_result = ev.evaluate_direct(tscbn, None, test_sequence, None)
            print(str(eval_result))

            all_res["jpd"].append(float(eval_result["TSCBN"]["jpd"].replace("\t", "")))
            all_res["log-likelihood"].append(float(eval_result["TSCBN"]["log-likelihood"].replace("\t", "")))
            all_res["temp-log-likelihood"].append(float(eval_result["TSCBN"]["temp-log-likelihood"].replace("\t", "")))
            all_res["temp-jpd"].append(float(eval_result["TSCBN"]["temp-jpd"].replace("\t", "")))

        # print tree
        if False:
            for n in pe.tbn.Vdata:
                try:
                    if isinstance(pe.tbn.Vdata[n]["cprob"], dict):
                        for k in pe.tbn.Vdata[n]["cprob"]:
                            print("%s | %s = %s" % (n, k, str(pe.tbn.Vdata[n]["cprob"][k])))
                    else:
                        print("%s = %s" % (n, str(pe.tbn.Vdata[n]["cprob"])))
                except:
                    for k in pe.tbn.Vdata[n]["hybcprob"]:
                        print("%s | %s = mean: %s var: %s" % (n, k, str(pe.tbn.Vdata[n]["hybcprob"][k]["mean_base"]), str(pe.tbn.Vdata[n]["hybcprob"][k]["variance"])))

            print("\n\n")

        # print result
        print("\n\nFinal results (average of iterations):")
        for k in all_res:
            print("%s = %s" % (str(k), str(np.mean(all_res[k]))))


def run_case_ind(parallel_proesses, plot_model, estimators, show_plot = False):
    # 1. Discover structure
    ind = IndModel()
    tscbn = ind.generate_model()
    tscbn.show_plot_generated = show_plot
    if plot_model: tscbn.draw("ext") # show the model, else set false

    # 2. Extract sequences
    input_csv = "store/case_study_data/sequences_ind.csv"



    sequences = ind.extract_signals(input_csv)

    all_res = {}
    all_res["jpd"] = []
    all_res["log-likelihood"] = []
    all_res["temp-log-likelihood"] = []
    all_res["temp-jpd"] = []

    for estimator_id in estimators:
        for _ in range(10):  # cross validation - draw 10 times
            # split
            train_sequence, test_sequence = train_test_split(sequences, test_size=0.2)

            # 3. Learn parameters from data
            pe = create_estimator(estimator_id)
            pe.histogram_smoothing = 0.0
            pe.cpd_smoothing = 0.1
            pe.sampling_frequency = 1000  # sampling frequency for the MC MC Simulation - too low?
            pe.iteration_frequency = 5  # EM Iterations
            pe.set_parallel_processes(parallel_proesses)
            pe_debug_mode = False
            pe.tbn = tscbn
            pe.tbn.clear_parameters()
            pe.estimateParameter(train_sequence, "TSCBNStructureModel", pe_debug_mode)

            # 4. Evaluate results
            ev = CSParameterEvaluator(False)
            ev.add_metric("jpd")
            ev.add_metric("log-likelihood")
            ev.add_metric("temp-log-likelihood")
            ev.add_metric("temp-jpd")
            eval_result = ev.evaluate_direct(tscbn, None, test_sequence, None)

            #print(str(eval_result))

            all_res["jpd"].append(float(eval_result["TSCBN"]["jpd"].replace("\t", "")))
            all_res["log-likelihood"].append(float(eval_result["TSCBN"]["log-likelihood"].replace("\t", "")))
            all_res["temp-log-likelihood"].append(float(eval_result["TSCBN"]["temp-log-likelihood"].replace("\t", "")))
            all_res["temp-jpd"].append(float(eval_result["TSCBN"]["temp-jpd"].replace("\t", "")))

        print(all_res)
        # print tree
        if False:
            for n in pe.tbn.Vdata:
                try:
                    if isinstance(pe.tbn.Vdata[n]["cprob"], dict):
                        for k in pe.tbn.Vdata[n]["cprob"]:
                            print("%s | %s = %s" % (n, k, str(pe.tbn.Vdata[n]["cprob"][k])))
                    else:
                        print("%s = %s" % (n, str(pe.tbn.Vdata[n]["cprob"])))
                except:
                    for k in pe.tbn.Vdata[n]["hybcprob"]:
                        print("%s | %s = mean: %s var: %s" % (n, k, str(pe.tbn.Vdata[n]["hybcprob"][k]["mean_base"]),
                                                              str(pe.tbn.Vdata[n]["hybcprob"][k]["variance"])))

        print("\n\n")

        # print result
        print("\n\nFinal results (average of iterations):")
        for k in all_res:
            print("%s = %s" % (str(k), str(np.mean(all_res[k]))))


def map_func(bn, destination_file):

    top_n = 500
    sample_nr = 100000
    output = destination_file

    samples = bn.randomsample(sample_nr)
    d = {}
    ma = dict()
    rev_ma = dict()
    t_ma = dict()
    idx = 0
    cnt = 0
    for s in samples:
        cnt+=1
        if cnt % 500 == 0:
            print("Processing %d" % cnt)
        t = [(k, s[k]) for k in s if str.startswith(k, "dL_")]
        s = [(k, s[k]) for k in s if not str.startswith(k, "dL_")]



        if not str(s) in ma:
            ma[str(s)] = idx
            rev_ma[idx] = str(s)
            t_ma[idx] = str(t)

            d[idx] = 0
            idx += 1
        d[ma[str(s)]] += 1

    # print only 20 most probable sorted -> 1

    n = dict()
    th_v = sorted(list(d.values()))[-top_n]
    sorted_k = dict( (k, v) for k, v in d.items() if v >= th_v )


    # generate nice result string
    res_s = "\n"
    res_s += "Inter-Edges: %s" % str([e for e in bn.E if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]])
    res_s += "\nIndices: %s " % str(d)
    res_s += "\n\n\n"

    tmp_ss = []
    for tt in sorted_k:
        s = rev_ma[tt] # state nodes
        t = dict(eval(t_ma[tt])) # temporal nodes
        frequency = d[tt]
        res_d = dict()
        res_t = dict()
        tmp_s = ""
        tmp_s += "\n\nFREQUENCY: %s" % str(frequency)
        ev = eval(s)
        ev = sorted(ev)
        for cont in ev:
            tv = "_".join(cont[0].split("_")[:-1])
            idx = int("_".join(cont[0].split("_")[-1]))
            dt = t["dL_" + cont[0]]

            if not tv in res_d:
                res_d[tv] = []
                res_t[tv] = []
            res_d[tv] += [(idx, cont[1])]
            res_t[tv] += [(idx, dt)]

        for tv in res_d:
            tmp_s += "\n%s " % tv
            tmp_lst = sorted(res_t[tv])
            i = 0
            for st in sorted(res_d[tv]):
                tmp_s += "- dt=%s - %s " % (np.max([0, tmp_lst[i][1]]), st[1])
                i += 1

        tmp_ss += [[frequency, tmp_s]]
    for ss in sorted(tmp_ss)[::-1]:
        res_s += ss[1]


    file2 = open(output,"w+")
    file2.write(res_s)
    file2.close()
    print("Written results to: %s " % str(output))

    #print("Inter-Edges: %s" % str([e for e in bn.E if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]]))
    plt.bar(d.keys(), d.values(), 1.0, color='g')
    plt.show()


def run_bayspec_approach(sequences, sb_max_time_difference, k, chi, output_folder, data, sd_hyperparameter_estimation_chi = False, sd_hyperparameter_estimation_t_th = False, sd_hyperparameter_estimation_k = False, learn_model = False, bayspec = False, map = False):

    evaluate_model = False # loads the model and computes the log-likelihoods of sequences on this model
    combined = False # generates heat map chi vs k
    df_sequences, sequences, states_dict = to_dataframe(sequences)

    # 1. Run Structure Discovery hyperparameter estimation
    if sd_hyperparameter_estimation_t_th:
        print("\n\n" + "-".join(20*[""]) + "Run HP estimation and discover a good maximal t_th of %s" % str(2.0 * 10**13) + "-".join(20*[""]))

        # find max time difference by plotting gaps distribution
        df_sequences = df_sequences.sort_values("start_time")
        df_sequences["gap"] = df_sequences["end_time"] - df_sequences["start_time"]
        sns.distplot(np.nan_to_num(df_sequences["gap"]), kde = False)

        ax = plt.gca()
        def get_hist(ax):
            n,bins = [],[]
            for rect in ax.patches:
                ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
                n.append(y1-y0)
                bins.append(x0) # left edge of each bin
            bins.append(x1) # also get right edge of last bin
            return n,bins
        n, bins = get_hist(ax)#",".join([str((bins[i], n[i])) for i in range(len(bins))])

        plt.xlabel("Time Gap between entries")
        plt.ylabel("Frequency")
        plt.show()
        # c1 ",".join([str((line.get_xdata()[i], line.get_ydata()[i])) for i in range(len(line.get_xdata()))])
        return

    if combined:
        inter_edge_numbers = []
        k_range = list(np.arange(0.05, 1.05, 0.05))
        params = []
        for k in k_range:
            chi_range = list(np.arange(0.05, 1.05, 0.1)) + list(np.arange(1.0, 5.0, 0.5)) + list(np.arange(5.0, 100.1, 5.0))
            for chi in chi_range:

                sd = PCTreeDiscoverer(min_out_degree=0.15, k_infrequent=k, parallel=False, alpha=0.5, chi_square_thresh=chi, optimization_chi_square=True, max_time_difference=sb_max_time_difference)

                print("\n\n" + "-".join(20*[""]) + "Starting Structure Discovery k=%s chi=%s" % (str(k), str(chi)) + "-".join(20*[""]))
                try:
                    structure = sd.discover_structure(sequences)
                    edges = structure[1]
                    inter_edge_numbers += [len([e for e in edges if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]])]
                    print("Nodes: %s" % str(len([s for s in structure[0] if not str.startswith(s, "dL_")])))
                    ie = len([e for e in edges if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]])
                    print("Inter-edges %d" % ie)
                except AssertionError:
                    ie= 0
                    inter_edge_numbers += [0]
                params += [[k, chi, ie]]
        print(str(params))
        #plt.plot(k_range, inter_edge_numbers)
        #plt.xlabel("k")
        #plt.ylabel("# Inter-Edges")
        #plt.show()
        return

    if sd_hyperparameter_estimation_chi:
        inter_edge_numbers = []
        chi_range = list(np.arange(0.05, 1.05, 0.1)) + list(np.arange(1.0, 5.0, 0.5)) + list(np.arange(5.0, 100.1, 5.0))
        for chi in chi_range:
            sd = PCTreeDiscoverer(min_out_degree=0.15, k_infrequent=k, parallel=False, alpha=0.5, chi_square_thresh=chi, optimization_chi_square=True, max_time_difference=sb_max_time_difference)

            print("\n\n" + "-".join(20*[""]) + "Starting Structure Discovery chi=%s" % str(chi) + "-".join(20*[""]))
            try:
                structure = sd.discover_structure(sequences)
                edges = structure[1]
                inter_edge_numbers += [len([e for e in edges if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]])]
                print("Nodes: %s" % str(len([s for s in structure[0] if not str.startswith(s, "dL_")])))
                print("Inter-edges %d" % len([e for e in edges if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]]))
            except AssertionError:
                inter_edge_numbers += [0]

    # 2. Run Structure Discovery hyperparameter estimation for k
    if sd_hyperparameter_estimation_k:
        inter_edge_numbers = []
        k_range = list(np.arange(0.05, 1.05, 0.05))
        for k in k_range:
            sd = PCTreeDiscoverer(min_out_degree=0.15, k_infrequent=k, parallel=False, alpha=0.1, chi_square_thresh=1, optimization_chi_square=True, max_time_difference=sb_max_time_difference)

            print("\n\n" + "-".join(20*[""]) + "Starting Structure Discovery k=%s" % str(k) + "-".join(20*[""]))
            try:
                structure = sd.discover_structure(sequences)
                edges = structure[1]
                inter_edge_numbers += [len([e for e in edges if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]])]
                print("Nodes: %s" % str(len([s for s in structure[0] if not str.startswith(s, "dL_")])))
                print("Inter-edges %d" % len([e for e in edges if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]]))
            except AssertionError:
                inter_edge_numbers += [0]
        plt.plot(k_range, inter_edge_numbers)
        plt.xlabel("k")
        plt.ylabel("# Inter-Edges")
        plt.show()
        return

    if sd_hyperparameter_estimation_chi:
        inter_edge_numbers = []
        chi_range = list(np.arange(0.05, 1.05, 0.1)) + list(np.arange(1.0, 5.0, 0.5)) + list(np.arange(5.0, 100.1, 5.0))
        for chi in chi_range:
            sd = PCTreeDiscoverer(min_out_degree=0.15, k_infrequent=k, parallel=False, alpha=0.5, chi_square_thresh=chi, optimization_chi_square=True, max_time_difference=sb_max_time_difference)

            print("\n\n" + "-".join(20*[""]) + "Starting Structure Discovery chi=%s" % str(chi) + "-".join(20*[""]))
            try:
                structure = sd.discover_structure(sequences)
                edges = structure[1]
                inter_edge_numbers += [len([e for e in edges if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]])]
                print("Nodes: %s" % str(len([s for s in structure[0] if not str.startswith(s, "dL_")])))
                print("Inter-edges %d" % len([e for e in edges if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]]))
            except AssertionError:
                inter_edge_numbers += [0]
        plt.plot(chi_range, inter_edge_numbers)
        plt.xlabel("chi")
        plt.ylabel("# Inter-Edges")
        plt.show()
        return

    # 3. Run SD and Parameter Estimation, with parameters found from evaluation
    if learn_model:

        print("\n\n" + "-".join(20*[""]) + "Starting Structure Discovery k=%s, t_th=%s" % (str(k), str(sb_max_time_difference)) + "-".join(20*[""]))
        #sd = SBTreeDiscoverer(min_out_degree=0.15, k_infrequent=k, approach='parent_graph', parallel=True, score="BIC", max_time_difference=sb_max_time_difference)
        sd = PCTreeDiscoverer(min_out_degree=0.15, k_infrequent=k, parallel=False, alpha=0.5, chi_square_thresh=chi, optimization_chi_square=True, max_time_difference=sb_max_time_difference)
        # PREFIX
        #sd = NovelStructureDiscoverer(filtering=False, k_infrequent=0.1, alpha=0.1 , draw=False, max_reach=2, min_out_degree=0.25,  draw_only_result=False)
        nodes, edges = sd.discover_structure(sequences)
        edges, nodes = structure_consistency(edges, nodes)
        tscbn = create_tscbn(states_dict, nodes, edges)
        print("Nodes: %s Inter-edges: %d" % (str(len(nodes)), len([e for e in edges if not str.startswith(e[1], "dL_") and e[0].split("_")[0] != e[1].split("_")[0]])))

        # run parameter estimation
        print("\n\n" + "-".join(20*[""]) + "Starting Parameter Estimation EM" + "-".join(20*[""]))
        pe = EMAlgorithmParameterEstimator()#MLECountingLocalParameterEstimator()
        pe.tbn = tscbn
        pe.iteration_frequency = 4
        tscbn = pe.estimateParameter(sequences, "TSCBNStructureModel")

        # Store result to file
        destination_file = os.path.join(output_folder, data.__class__.__name__ + ".tscbn")
        print("Found model - storing to %s" % str(destination_file))
        with open(destination_file, "wb") as dill_file:
            dill.dump(tscbn, dill_file, recurse=True)


    if evaluate_model:
        # evaluate results
        destination_file = os.path.join(output_folder, data.__class__.__name__ + ".tscbn")
        with open(destination_file, 'rb') as in_strm:
            tscbn = dill.load(in_strm)
        ev = CSParameterEvaluator(False)
        ev.add_metric("jpd")
        ev.add_metric("log-likelihood")
        ev.add_metric("temp-log-likelihood")
        ev.add_metric("temp-jpd")
        eval_result = ev.evaluate_direct(tscbn, None, sequences, None)


        log_likelihood = float(eval_result["TSCBN"]["log-likelihood"].replace("\t", ""))
        temp_log_likelihood = float(eval_result["TSCBN"]["temp-log-likelihood"].replace("\t", ""))
        print("LL %s, TLL %s" % (str(log_likelihood), str(temp_log_likelihood)))

        return

    if bayspec:
        min_prob_threshold = 0.6
        print("\n\n" + "-".join(20*[""]) + "Running Bayspec with p_min = %s" % str(min_prob_threshold) + "-".join(20*[""]))

        # Load file
        destination_file = os.path.join(output_folder, data.__class__.__name__ + ".tscbn")
        if not os.path.exists(destination_file):
            print("Please run SD and PE first before extraction of specifications. \nNo file found at %s" % str(destination_file))
            return
        with open(destination_file, 'rb') as in_strm:
            tscbn = dill.load(in_strm)

        # Run Bayspec
        tscbn = sanitize_commas(tscbn)
        bn1 = MaxAverageMiningGraph(tscbn)
        all_paths = bn1.path_computation(min_prob_threshold)
        paths = [p for p in all_paths if p["metric"] <= (1 - 0.7)] # metric hier - sagt bei welcher Levenshtein er nimmer merged d.h. strenger heisst weniger verwaschen!
        bn1.paths = paths
        metric_based_miner = MBMiner(bn1)
        prespecs = metric_based_miner.start()


        # Print Results
        print("\n\n" + "-".join(40*[""]) + "\n\tPaths\n"+ "-".join(40*[""]))
        for p in paths:
            print(p)

        print("\n\n" + "-".join(40*[""]) + "\n\tFound Specifications\n"+ "-".join(40*[""]))
        specs = []
        for spec in prespecs[::-1]:
            if spec[1] is not None:
                print(str(spec))
                specs += [spec]

        return

    if map:
        min_prob_threshold = 0.6
        print("\n\n" + "-".join(20*[""]) + "Running Bayspec with p_min = %s" % str(min_prob_threshold) + "-".join(20*[""]))

        # Load file
        destination_file = os.path.join(output_folder, data.__class__.__name__ + ".tscbn")
        if not os.path.exists(destination_file):
            print("Please run SD and PE first before extraction of specifications. \nNo file found at %s" % str(destination_file))
            return
        with open(destination_file, 'rb') as in_strm:
            tscbn = dill.load(in_strm)

        # Run MAP
        destination_file = os.path.join(output_folder, data.__class__.__name__ + str(chi).replace(".","_") + ".txt")
        map_func(tscbn, destination_file)


def run_case_hb_bayspec(sd_hyperparameter_estimation_t_th = False, sd_hyperparameter_estimation_k = False,sd_hyperparameter_estimation_chi = False, learn_model = False, bayspec = False, map = False):

    # Extract sequences
    data = HBModel()
    input_csv = "store/case_study_data/sequences_hb.csv"

    translate_signals = None 
    translate_states = None


    output_folder = os.path.split(input_csv)[0]
    sequences = data.extract_signals(input_csv, translate_signals, translate_states)

    sb_max_time_difference = 2 * 2.0 * 10**13 # found in previous step ( 2* to include two events before
    k = 0.05 # found in previous step
    chi = 5 # 0.8 free; 5 restrictive

    # Run Bayspec
    run_bayspec_approach(sequences, sb_max_time_difference, k, chi, output_folder, data, sd_hyperparameter_estimation_chi, sd_hyperparameter_estimation_t_th, sd_hyperparameter_estimation_k, learn_model, bayspec, map)


def run_case_ind_bayspec(sd_hyperparameter_estimation_t_th = False, sd_hyperparameter_estimation_k = False, sd_hyperparameter_estimation_chi = False, learn_model = False, bayspec = False, map = False):

    # 1. Discover structure
    data = IndModel()
    input_csv = "store/case_study_data/sequences_ind.csv"

    translate_signals = None 
    translate_states = None

    output_folder = os.path.split(input_csv)[0]
    sequences = data.extract_signals(input_csv, translate_signals, translate_states)# HIER KÖNNTE DAS PROBLEM LIEGEN WEIL ICH EINFACH EINEN SCHEISS DAZUPACKE

    shifto = dict()
    shifto["S-D"] = 0.001 # send at the same time -> thus, shift by dependency
    shifto["S-E"] = 0.002 # send at the same time -> thus, shift by dependency
    shifto["S-B"] = 0.003 # send at the same time -> thus, shift by dependency
    sequences = filter_longer_one_n_shift(sequences, shifto)


    sb_max_time_difference = 2* 4.0 * 10**9 # found in previous step - 2*
    k = 0.2 # found in previous step
    chi = 0.25 # found in previous step 0.25  & 0.85

    # Run Bayspec
    run_bayspec_approach(sequences, sb_max_time_difference, k, chi, output_folder, data, sd_hyperparameter_estimation_chi, sd_hyperparameter_estimation_t_th, sd_hyperparameter_estimation_k, learn_model, bayspec,map)
