from _include.estimator.ctbn_estimator import CTBNEstimator
from _include.estimator.em_algorithm_tscbn_estimator import EMAlgorithmParameterEstimator
from _include.evaluation.parameter_evaluator import ParameterEvaluator
from _include.structure.ctbn_structure_model import CTBNStructureModel
from _include.structure.dbn_structure_model import DBNStructureModel
from _include.structure.structure_generator import StructureGenerator
from _include.structure._tscbn_structure_model import TSCBNStructureModel
from general.log import Log as L
import datetime, os
import json, dill # https://pypi.org/project/dill/
import copy
from general.setup import Printos, create_estimator
import time


def run_vary_sc_probability(number_TVs, parallel_proesses, result_path, print_sequences, plot_model, print_true_distribution, estimators):

    if not number_TVs in [3,5,10]:
        print("No model is stored for this number_TV value. Please set number_TV to 3, 5 or 10")
        return

    L().log.info("Initializing Experiment: State Change Probability variation")
    append_csv = False
    id_time = datetime.datetime.now().strftime("%I_%M%p_%d_%B_%Y_%H_%M_%S")

    for estimator_id in estimators:
        for state_change_prob in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]: #  [0.65]

            # Settings
            pe_debug_mode = False
            cpd_smoothing = 0.1
            parallel_processes = parallel_proesses
            object_nr = number_TVs
            nodes_per_tv = 5
            states_per_tv = 4
            edges_per_tv = 2
            percentage_inter = 0.5
            per_object_gap = 0.5 # gap between two intra-nodes
            intra_gap_range = 0.1 # gap between two intra-nodes is drawn - kind of variance: lies within - per_object_gap and per_object_gap+intra_gap_range e.g. [0.5 to 0.5 + 0.1]
            t_variance_tscbn = 0.02 # Variance of resulting TSCBN (after Parameter estimation)
            dbn_tolerance = 0.02 # tolerance
            train_test_split = 0.9  # percentage of training data
            id = "_"+ "_".join([str(v) for v in [object_nr, nodes_per_tv, states_per_tv, edges_per_tv, percentage_inter, per_object_gap, intra_gap_range, t_variance_tscbn, dbn_tolerance, state_change_prob, train_test_split]])
            out_path = os.path.join(result_path, r"evaluation_%s.csv" % id_time)

            # Iteration options
            grid_sample_sequences_from_tscbn =  [2000] # unter 1000 macht hier gar keinen Sinn bei so vielen Daten e.g. hier 228 samples - bei 100 sequenzen sehe nichtmal bruchteil
            grid_em_sampling_frequency = [1000]
            grid_em_iterations = [5]

            # init
            sg = StructureGenerator(test_type = 1)
            sg.add_base_structure_models([TSCBNStructureModel, DBNStructureModel, CTBNStructureModel])  #  TNBNStructureModel, DBNStructureModel])
            sg.reference_model = TSCBNStructureModel # this model is used to generate sample data

            # Load sequences
            sequences_in = json.load(open('store/sequences%s.txt' % id))
            in_seq_in = json.load(open('store/in_seq%s.txt' % id))
            first = True

            if print_sequences:
                k = 0
                for sequence in sequences_in:
                    k += 1; print(sequence)
                    if k % 100 == 0:
                        r = input("To load more sequences type 'y' ")
                        if not r == "y": break

            # print True distribution of model
            if print_true_distribution:
                print("Actual distribution: ")
                with open('store/models%s.txt' % id, 'rb') as infile:
                    real_models = dill.load(infile)
                    infile.close()
                act_tscbn = real_models[sg.reference_model.__name__]

                for n in act_tscbn.Vdata:
                    try:
                        if isinstance(act_tscbn.Vdata[n]["cprob"], dict):
                            for k in act_tscbn.Vdata[n]["cprob"]:
                                print("%s | %s = %s" % (n, k, str(act_tscbn.Vdata[n]["cprob"][k])))
                        else:
                            print("%s = %s" % (n, str(act_tscbn.Vdata[n]["cprob"])))
                    except:
                        for k in act_tscbn.Vdata[n]["hybcprob"]:
                            print("%s | %s = mean: %s var: %s" % (n, k, str(act_tscbn.Vdata[n]["hybcprob"][k]["mean_base"]), str(act_tscbn.Vdata[n]["hybcprob"][k]["variance"])))
                print("\n\n")

            for sample_sequences_from_tscbn in grid_sample_sequences_from_tscbn:
                for em_sampling_frequency in grid_em_sampling_frequency:
                    for em_iterations in grid_em_iterations:
                        print("\n-------------------------------\nDo: "+str(object_nr) +" "+ str(nodes_per_tv) +" "+ str(states_per_tv) +" "+ str(edges_per_tv) +" "+ str(percentage_inter) +" "+ str(per_object_gap) +" "+ str(t_variance_tscbn) +" "+ str(dbn_tolerance) +" "+ str(state_change_prob) +" "+ str(sample_sequences_from_tscbn) +" "+ str(em_sampling_frequency) +" "+ str(em_iterations))

                        # Load reference model
                        with open('store/models%s.txt' % id, 'rb') as infile:
                            real_models = dill.load(infile)
                            infile.close()
                        with open('store/specifications%s.txt' % id, 'rb') as infile:
                            specifications = dill.load(infile)
                            infile.close()
                        models = copy.deepcopy(real_models)
                        models["CTBNStructureModel"] = CTBNStructureModel()

                        # Initialize Parameter Estimators
                        pe = create_estimator(estimator_id)
                        ctbn_estimator = CTBNEstimator()

                        pe.original_tbn = copy.deepcopy(models[sg.reference_model.__name__])
                        original_tbn = copy.deepcopy(models[sg.reference_model.__name__])
                        if plot_model and first:
                            pe.original_tbn.draw("ext")
                            first = False

                        # Initialize Estimator and Evaluator
                        ev = ParameterEvaluator(append_csv);append_csv = True
                        ev.add_setting("estimator", str(estimator_id))
                        ev.add_setting("object_nr", object_nr)
                        ev.add_setting("nodes_per_tv", nodes_per_tv)
                        ev.add_setting("states_per_tv", states_per_tv)
                        ev.add_setting("edges_per_tv", edges_per_tv)
                        ev.add_setting("percentage_inter", percentage_inter)
                        ev.add_setting("per_tv_gap", per_object_gap)
                        ev.add_setting("tscbn_variance", t_variance_tscbn)
                        ev.add_setting("dbn_tolerance", dbn_tolerance)
                        ev.add_setting("sc_probability", state_change_prob)
                        ev.add_setting("sample_sequences_from_tscbn", sample_sequences_from_tscbn)
                        ev.add_setting("em_sampling_frequency", em_sampling_frequency)
                        ev.add_setting("em_iterations", em_iterations)
                        ev.set_output_path(out_path)
                        ev.rmse_tscb_variance = 0.1  # variance assumed per node - does not require parameter estimation
                        ev.rmse_mean_range = 0.2  # drift of mean will be within this range e.g. 0.1 means it will be drawn from correct +- drift*correct
                        ev.add_metric("runtime")
                        ev.add_metric("log-likelihood")
                        ev.add_metric("relative-entropy")
                        ev.add_metric("temp-log-likelihood")
                        pe.cpd_smoothing = cpd_smoothing
                        pe.sampling_frequency = em_sampling_frequency # sampling frequency for the MC MC Simulation
                        pe.iteration_frequency = em_iterations # EM Iterations
                        pe.set_parallel_processes(parallel_processes)
                        evidence = {}  # evidence when sampling
                        sg.set_model_visualization(plot = False, console_out = False)
                        Printos.print_settings(sg, pe, ev, 1, train_test_split, sample_sequences_from_tscbn, evidence, [])

                        # --------------------------------------------------------------------------------------------
                        #       Run tests
                        # --------------------------------------------------------------------------------------------
                        L().log.info("------------------ Running Test ------------------" )
                        if not ev._append_csv: eval_result = ev.write_header(True)
                        sequences = sequences_in[:sample_sequences_from_tscbn + 1]
                        in_seq = in_seq_in[:sample_sequences_from_tscbn + 1]

                        # choose random train and test data
                        from sklearn.model_selection import train_test_split
                        train_sequences, test_sequences, train_tscbn_sequences, test_tscbn_sequences = train_test_split(sequences, in_seq, test_size=0.1, random_state=0)

                        # ----------------------------------------------------------------------------------------
                        #       ESTIMATE PARAMETERS
                        # ----------------------------------------------------------------------------------------
                        for m in list(set(models)):
                            print("\nEstimating: %s ---" % str(m))
                            L().log.info("Parameter Estimation %s..." %(str(m)))

                            # TESTING
                            #print("_-------___TEST")
                            #if m != 'TSCBNStructureModel':continue

                            # Estimation of CTBN
                            if m == 'CTBNStructureModel':
                                models[m] = ctbn_estimator.estimateStructureAndParameter(train_sequences,original_tbn)
                                continue

                            # Clear Models
                            pe.tbn = copy.deepcopy(models[m])
                            pe.original_tbn = copy.deepcopy(models[m])
                            if m == sg.reference_model.__name__:
                                pe.tbn.clear_parameters() # copy model structure only

                            # Estimate Parameters of DBN and TSCBN
                            ping = time.clock()
                            pe.estimateParameter(train_sequences, m, pe_debug_mode, ev, pe.original_tbn) # computes kl divergence per run
                            models[m] = pe.tbn
                            models[m].parameter_execution_time = time.clock() - ping # eceution time
                            print("Finished: %s ---" % str(m))

                        # ----------------------------------------------------------------------------------------
                        #       EVALUATION
                        # ----------------------------------------------------------------------------------------
                        eval_result = ev.evaluate(models, reference = pe._reference, test_sequences = test_sequences, test_tscbn_sequences = test_tscbn_sequences)
                        ev.print_eval_results(eval_results = eval_result, specs = specifications, to_csv = True)


def run_vary_sample_number(number_TVs, parallel_processes, result_path, print_sequences, plot_model, print_true_distribution, estimators):

        if not number_TVs in [3,5,10]:
            print("No model is stored for this number_TV value. Please set number_TV to 3, 5 or 10")
            return

        # Settings
        state_change_prob = 0.8
        pe_debug_mode = False
        cpd_smoothing = 0.1
        parallel_processes = parallel_processes
        object_nr = number_TVs
        nodes_per_tv = 5
        states_per_tv = 4
        edges_per_tv = 2
        percentage_inter = 0.5
        per_object_gap = 0.5 # gap between two intra-nodes
        intra_gap_range = 0.1 # gap between two intra-nodes is drawn - kind of variance: lies within - per_object_gap and per_object_gap+intra_gap_range e.g. [0.5 to 0.5 + 0.1]
        t_variance_tscbn = 0.02 # Variance of resulting TSCBN (after Parameter estimation)
        dbn_tolerance = 0.02 # tolerance
        train_test_split = 0.9  # percentage of training data
        id = "_"+ "_".join([str(v) for v in [object_nr, nodes_per_tv, states_per_tv, edges_per_tv, percentage_inter, per_object_gap, intra_gap_range, t_variance_tscbn, dbn_tolerance, state_change_prob, train_test_split]])
        append_csv = False
        id_time = datetime.datetime.now().strftime("%I_%M%p_%d_%B_%Y_%H_%M_%S")
        out_path = os.path.join(result_path, r"evaluation_%s.csv" % id_time)

        # Iteration options
        grid_sample_sequences_from_tscbn =  [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000] # unter 1000 macht hier gar keinen Sinn bei so vielen Daten e.g. hier 228 samples - bei 100 sequenzen sehe nichtmal bruchteil
        grid_em_sampling_frequency = [1000]
        grid_em_iterations = [5]

        # init
        sg = StructureGenerator(test_type = 1)
        sg.add_base_structure_models([TSCBNStructureModel, DBNStructureModel, CTBNStructureModel])  #  TNBNStructureModel, DBNStructureModel])
        sg.reference_model = TSCBNStructureModel # this model is used to generate sample data

        # Load sequences
        sequences_in = json.load(open('store/sequences%s.txt' % id))
        in_seq_in = json.load(open('store/in_seq%s.txt' % id))

        first = True

        if print_sequences:
            k = 0
            for sequence in sequences_in:
                k += 1; print(sequence)
                if k % 100 == 0:
                    r = input("To load more sequences type 'y' ")
                    if not r == "y": break

        # print True distribution of model
        if print_true_distribution:
            print("Actual distribution: ")
            with open('store/models%s.txt' % id, 'rb') as infile:
                real_models = dill.load(infile)
                infile.close()
            act_tscbn = real_models[sg.reference_model.__name__]

            for n in act_tscbn.Vdata:
                try:
                    if isinstance(act_tscbn.Vdata[n]["cprob"], dict):
                        for k in act_tscbn.Vdata[n]["cprob"]:
                            print("%s | %s = %s" % (n, k, str(act_tscbn.Vdata[n]["cprob"][k])))
                    else:
                        print("%s = %s" % (n, str(act_tscbn.Vdata[n]["cprob"])))
                except:
                    for k in act_tscbn.Vdata[n]["hybcprob"]:
                        print("%s | %s = mean: %s var: %s" % (n, k, str(act_tscbn.Vdata[n]["hybcprob"][k]["mean_base"]), str(act_tscbn.Vdata[n]["hybcprob"][k]["variance"])))
            print("\n\n")

        for estimator_id in estimators:
            for sample_sequences_from_tscbn in grid_sample_sequences_from_tscbn:
                for em_sampling_frequency in grid_em_sampling_frequency:
                    for em_iterations in grid_em_iterations:
                        print("\n-------------------------------\nDo: "+str(object_nr) +" "+ str(nodes_per_tv) +" "+ str(states_per_tv) +" "+ str(edges_per_tv) +" "+ str(percentage_inter) +" "+ str(per_object_gap) +" "+ str(t_variance_tscbn) +" "+ str(dbn_tolerance) +" "+ str(state_change_prob) +" "+ str(sample_sequences_from_tscbn) +" "+ str(em_sampling_frequency) +" "+ str(em_iterations))

                        # Load reference model
                        with open('store/models%s.txt' % id, 'rb') as infile:
                            real_models = dill.load(infile)
                            infile.close()
                        with open('store/specifications%s.txt' % id, 'rb') as infile:
                            specifications = dill.load(infile)
                            infile.close()
                        models = copy.deepcopy(real_models)
                        models["CTBNStructureModel"] = CTBNStructureModel()

                        # Parameter Estimation
                        pe = create_estimator(estimator_id)
                        ctbn_estimator = CTBNEstimator()
                        pe.original_tbn = copy.deepcopy(models[sg.reference_model.__name__])
                        original_tbn = copy.deepcopy(models[sg.reference_model.__name__])
                        if plot_model and first:
                            pe.original_tbn.draw("ext")
                            first = False

                        # Initialize Estimator and Evaluator
                        ev = ParameterEvaluator(append_csv);append_csv = True
                        ev.add_setting("estimator", str(estimator_id))
                        ev.add_setting("object_nr", object_nr)
                        ev.add_setting("nodes_per_tv", nodes_per_tv)
                        ev.add_setting("states_per_tv", states_per_tv)
                        ev.add_setting("edges_per_tv", edges_per_tv)
                        ev.add_setting("percentage_inter", percentage_inter)
                        ev.add_setting("per_tv_gap", per_object_gap)
                        ev.add_setting("tscbn_variance", t_variance_tscbn)
                        ev.add_setting("dbn_tolerance", dbn_tolerance)
                        ev.add_setting("sc_probability", state_change_prob)
                        ev.add_setting("sample_sequences_from_tscbn", sample_sequences_from_tscbn)
                        ev.add_setting("em_sampling_frequency", em_sampling_frequency)
                        ev.add_setting("em_iterations", em_iterations)
                        ev.set_output_path(out_path)
                        ev.rmse_tscb_variance = 0.1  # variance assumed per node - does not require parameter estimation
                        ev.rmse_mean_range = 0.2  # drift of mean will be within this range e.g. 0.1 means it will be drawn from correct +- drift*correct
                        ev.add_metric("runtime")
                        ev.add_metric("log-likelihood")
                        ev.add_metric("relative-entropy")
                        ev.add_metric("temp-log-likelihood")
                        pe.cpd_smoothing = cpd_smoothing
                        pe.sampling_frequency = em_sampling_frequency # sampling frequency for the MC MC Simulation
                        pe.iteration_frequency = em_iterations # EM Iterations
                        pe.set_parallel_processes(parallel_processes)
                        evidence = {}  # evidence when sampling
                        sg.set_model_visualization(plot = False, console_out = False)
                        Printos.print_settings(sg, pe, ev, 1, train_test_split, sample_sequences_from_tscbn, evidence, [])

                        # --------------------------------------------------------------------------------------------
                        #       Run tests
                        # --------------------------------------------------------------------------------------------
                        L().log.info("------------------ Running Test ------------------" )
                        if not ev._append_csv: eval_result = ev.write_header(True)
                        sequences = sequences_in[:sample_sequences_from_tscbn + 1]
                        in_seq = in_seq_in[:sample_sequences_from_tscbn + 1]

                        # choose random train and test data
                        from sklearn.model_selection import train_test_split
                        train_sequences, test_sequences, train_tscbn_sequences, test_tscbn_sequences = train_test_split(sequences, in_seq, test_size=0.1, random_state=0)

                        # ----------------------------------------------------------------------------------------
                        #       ESTIMATE PARAMETERS
                        # ----------------------------------------------------------------------------------------
                        for m in list(set(models)):
                            print("\nEstimating: %s ---" % str(m))
                            L().log.info("Parameter Estimation %s..." %(str(m)))

                            # TESTING
                            #print("_-------___TEST")
                            #if m != 'TSCBNStructureModel':continue


                            if m == 'CTBNStructureModel':
                                models[m] = ctbn_estimator.estimateStructureAndParameter(train_sequences,original_tbn)
                                continue

                            # Clear Models
                            pe.tbn = copy.deepcopy(models[m])
                            pe.original_tbn = copy.deepcopy(models[m])
                            if m == sg.reference_model.__name__:
                                pe.tbn.clear_parameters() # copy model structure only

                            # Estimate Parameters
                            ping = time.clock()
                            pe.estimateParameter(train_sequences, m, pe_debug_mode, ev, pe.original_tbn) # computes kl divergence per run
                            models[m] = pe.tbn
                            models[m].parameter_execution_time = time.clock() - ping # exeution time
                            print("Finished: %s ---" % str(m))

                        # ----------------------------------------------------------------------------------------
                        #       EVALUATION
                        # ----------------------------------------------------------------------------------------
                        try:
                            eval_result = ev.evaluate(models, reference = pe._reference, test_sequences = test_sequences, test_tscbn_sequences = test_tscbn_sequences)
                            ev.print_eval_results(eval_results = eval_result, specs = specifications, to_csv = True)
                        except:
                            print("bad ")
