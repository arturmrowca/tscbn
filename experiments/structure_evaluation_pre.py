import copy
from time import localtime, strftime, clock
import os
import numpy as np

from _include.discoverer.a_star_discoverer import AStarDiscoverer
from _include.discoverer.hill_climb_discoverer import HillClimbDiscoverer
from _include.discoverer.mmhc_discoverer import MMHCDiscoverer
from _include.discoverer.pc_discoverer import PCDiscoverer
from _include.discoverer.pc_tree_discoverer import PCTreeDiscoverer
from _include.discoverer.sb_tree_discoverer import SBTreeDiscoverer
from _include.evaluation.structure_evaluator import StructureEvaluator
from _include.m_libpgm.graphskeleton import GraphSkeleton
from _include.structure.structure_generator import TestStructureEnum, StructureGenerator
from _include.structure._tscbn_structure_model import TSCBNStructureModel
from _include.toolkit import sequences_to_intervals
from general.log import Log as L
from network.tscbn import TSCBN


def write_pgfplots_data(experiment_name, eval_results, relevant_metrics, discovery_algorithms,
                        variation_list, variation_name, target_path):
    for algorithm in discovery_algorithms:
        output_path = (os.path.join(target_path, r"plot_%s_%s_%s.dat" % (
            experiment_name, algorithm, strftime("%Y_%m_%d-%H_%M_%S", localtime()))))
        file = open(output_path, 'w')
        # header
        file.write(variation_name)
        for metric in relevant_metrics:
            file.write(' ' + metric)
        file.write('\n')
        for variation_value in variation_list:
            if algorithm not in eval_results[variation_value]:  # the algorithm was not performed for this value
                continue
            # results
            file.write(str(variation_value))
            for metric in relevant_metrics:
                results = eval_results[variation_value][algorithm][metric]
                average = sum(results) / len(results)
                file.write(' ' + str(average))
            file.write('\n')
        pass
    pass


def get_structure_discovery_algorithms():
    # create all the algorithms that should be compared
    discovery_algorithms = []
    sd = SBTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, approach='parent_graph', parallel=False, score='BIC',
                          max_time_difference=1.0)
    discovery_algorithms.append(('sbPTM_BIC_TH_1.0', sd))
    sd = SBTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, approach='parent_graph', parallel=False, score='AIC',
                          max_time_difference=1.0)
    discovery_algorithms.append(('sbPTM_AIC_TH_1.0', sd))
    sd = SBTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, approach='parent_graph', parallel=False, score='Bdeu',
                          max_time_difference=1.0)
    discovery_algorithms.append(('sbPTM_Bdeu_TH_1.0', sd))
    sd = SBTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, approach='parent_graph', parallel=False, score='K2',
                          max_time_difference=1.0)
    discovery_algorithms.append(('sbPTM_K2_TH_1.0', sd))
    sd = PCTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, parallel=False, alpha=0.01,
                          max_time_difference=1.0)
    discovery_algorithms.append(('cbPTM_0.01_TH_1.0', sd))
    sd = PCTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, parallel=False, alpha=0.05,
                          max_time_difference=1.0)
    discovery_algorithms.append(('cbPTM_0.05_TH_1.0', sd))
    sd = PCTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, parallel=False, alpha=0.1,
                          max_time_difference=1.0)
    discovery_algorithms.append(('cbPTM_0.1_TH_1.0', sd))
    sd = SBTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, approach='parent_graph', parallel=False, score='BIC',
                          max_time_difference=1.5)
    discovery_algorithms.append(('sbPTM_BIC_TH_1.5', sd))
    sd = SBTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, approach='parent_graph', parallel=False, score='AIC',
                          max_time_difference=1.5)
    discovery_algorithms.append(('sbPTM_AIC_TH_1.5', sd))
    sd = SBTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, approach='parent_graph', parallel=False, score='Bdeu',
                          max_time_difference=1.5)
    discovery_algorithms.append(('sbPTM_Bdeu_TH_1.5', sd))
    sd = SBTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, approach='parent_graph', parallel=False, score='K2',
                          max_time_difference=1.5)
    discovery_algorithms.append(('sbPTM_K2_TH_1.5', sd))
    sd = PCTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, parallel=False, alpha=0.01,
                          max_time_difference=1.5)
    discovery_algorithms.append(('cbPTM_0.01_TH_1.5', sd))
    sd = PCTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, parallel=False, alpha=0.05,
                          max_time_difference=1.5)
    discovery_algorithms.append(('cbPTM_0.05_TH_1.5', sd))
    sd = PCTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, parallel=False, alpha=0.1,
                          max_time_difference=1.5)
    discovery_algorithms.append(('cbPTM_0.1_TH_1.5', sd))
    sd = PCDiscoverer(alpha=0.01, max_reach=2)
    discovery_algorithms.append(('PC_0.01_mr_2', sd))
    sd = PCDiscoverer(alpha=0.05, max_reach=2)
    discovery_algorithms.append(('PC_0.05_mr_2', sd))
    sd = PCDiscoverer(alpha=0.1, max_reach=2)
    discovery_algorithms.append(('PC_0.1_mr_2', sd))
    sd = PCDiscoverer(alpha=0.01, max_reach=1)
    discovery_algorithms.append(('PC_0.01_mr_1', sd))
    sd = PCDiscoverer(alpha=0.05, max_reach=1)
    discovery_algorithms.append(('PC_0.05_mr_1', sd))
    sd = PCDiscoverer(alpha=0.1, max_reach=1)
    discovery_algorithms.append(('PC_0.1_mr_1', sd))
    sd = AStarDiscoverer(score='AIC')
    discovery_algorithms.append(('Astar_AIC', sd))
    sd = AStarDiscoverer(score='BIC')
    discovery_algorithms.append(('Astar_BIC', sd))
    sd = AStarDiscoverer(score='Bdeu')
    discovery_algorithms.append(('Astar_Bdeu', sd))
    sd = AStarDiscoverer(score='K2')
    discovery_algorithms.append(('Astar_K2', sd))
    sd = HillClimbDiscoverer(score='AIC', tabu_length=0)
    discovery_algorithms.append(('GHC_AIC', sd))
    sd = HillClimbDiscoverer(score='BIC', tabu_length=0)
    discovery_algorithms.append(('GHC_BIC', sd))
    sd = HillClimbDiscoverer(score='Bdeu', tabu_length=0)
    discovery_algorithms.append(('GHC_Bdeu', sd))
    sd = HillClimbDiscoverer(score='K2', tabu_length=0)
    discovery_algorithms.append(('GHC_K2', sd))
    sd = MMHCDiscoverer(score='AIC', alpha=0.01, max_reach=0)
    discovery_algorithms.append(('MMHC_AIC_alpha_0.01', sd))
    sd = MMHCDiscoverer(score='AIC', alpha=0.05, max_reach=0)
    discovery_algorithms.append(('MMHC_AIC_alpha_0.05', sd))
    sd = MMHCDiscoverer(score='BIC', alpha=0.01, max_reach=0)
    discovery_algorithms.append(('MMHC_BIC_alpha_0.01', sd))
    sd = MMHCDiscoverer(score='BIC', alpha=0.05, max_reach=0)
    discovery_algorithms.append(('MMHC_BIC_alpha_0.05', sd))
    sd = MMHCDiscoverer(score='Bdeu', alpha=0.01, max_reach=0)
    discovery_algorithms.append(('MMHC_Bdeu_alpha_0.01', sd))
    sd = MMHCDiscoverer(score='Bdeu', alpha=0.05, max_reach=0)
    discovery_algorithms.append(('MMHC_Bdeu_alpha_0.05', sd))
    sd = MMHCDiscoverer(score='K2', alpha=0.01, max_reach=0)
    discovery_algorithms.append(('MMHC_K2_alpha_0.01', sd))
    sd = MMHCDiscoverer(score='K2', alpha=0.05, max_reach=0)
    discovery_algorithms.append(('MMHC_K2_alpha_0.05', sd))
    return discovery_algorithms


def run_structure_experiment(target_path, parameter_temp_nodes_experiment=False, parameter_signals_experiment=False,
                          comparison_experiment_temp_nodes=False, comparison_experiment_signals=False,
                          comparison_experiment_scp=False):
    # number of iterations per experiment
    iterations = 25
    # number of sequences per experiment
    sample_size = 5000

    # ----------------------------------------------------------------------------------------
    #      Structure Generator Setup
    # ----------------------------------------------------------------------------------------
    sg = StructureGenerator(test_type=TestStructureEnum.SPECIFICATION)
    sg.add_base_structure_models([TSCBNStructureModel])
    sg.reference_model = TSCBNStructureModel

    # TIME SETTINGS (fixed for all experiments)
    sg.set_temporal_range(min_per_object_gap=0.5, max_per_object_gap=1.0)
    sg.set_temporal_variance(0.001)
    sg.set_dbn_tolerance(0.1)

    # PROBABILITY SETTINGS (fixed for all experiments)
    sg.set_state_change_probability(min_probability=0.95, max_probability=0.95)

    # ----------------------------------------------------------------------------------------
    #      Experiment with different parameters of the SBTreeDiscoverer
    # ----------------------------------------------------------------------------------------
    if parameter_temp_nodes_experiment or parameter_signals_experiment:
        sd = SBTreeDiscoverer(min_out_degree=0.1, k_infrequent=0.1, approach='parent_graph', parallel=False)
        # filtering parameters fixed at 0.1
        # parent graph approach means exact score optimization (but not exhaustive)
        # structure optimization not iteration in parallel

        for edges_per_object in [1, 3]:
            print('edges_per_object: ' + str(edges_per_object) + '...')
            L().log.info('edges_per_object: ' + str(edges_per_object) + '...')

            # EDGE SETTINGS
            sg.set_connection_ranges(min_edges_per_object=edges_per_object, max_edges_per_object=edges_per_object,
                                     min_percent_inter=1.0, max_percent_inter=1.0)

            if parameter_temp_nodes_experiment:
                # 1st experiment: Increase number of temporal variables per signal

                # EVALUATOR SETUP
                ev = StructureEvaluator(True)
                ev.set_output_path(os.path.join(target_path, r"structure_eval_%s.csv" % strftime("%Y_%m_%d-%H_%M_%S", localtime())))
                metrics = ["add-edges", "del-edges", "num-add-edges", "num-del-edges", "shd", "add-edges-skel",
                           "del-edges-skel", "num-add-edges-skel", "num-del-edges-skel", "shd-skel", "kld",
                           "execution-time", "psi-execution-time", "so-execution-time"]
                for metric in metrics:ev.add_metric(metric)
                eval_results = dict()
                discovery_algorithms = set()

                for number_of_signals in [2, 3, 4]:
                    print('number_of_signals: ' + str(number_of_signals) + '...')
                    L().log.info('number_of_signals: ' + str(number_of_signals) + '...')

                    if edges_per_object >= number_of_signals:
                        continue

                    numbers_of_temp_nodes = [1, 2, 3, 4, 5, 6, 7]
                    for number_of_temp_nodes in numbers_of_temp_nodes:
                        print('number_of_temp_nodes: ' + str(number_of_temp_nodes) + '...')
                        L().log.info('number_of_temp_nodes: ' + str(number_of_temp_nodes) + '...')

                        # NODE SETTINGS
                        sg.set_node_range(min_objects=number_of_signals, max_objects=number_of_signals,
                                          min_temp_nodes=number_of_temp_nodes, max_temp_nodes=number_of_temp_nodes,
                                          min_states=3, max_states=3)

                        eval_results.update({number_of_temp_nodes: dict()})

                        for iteration in range(0, iterations):
                            print('iteration: ' + str(iteration) + '...')
                            L().log.info('iteration: ' + str(iteration) + '...')

                            # SAMPLE DATA
                            models, specifications = sg.run_next_testcase()
                            in_seq = models[sg.reference_model.__name__].randomsample(sample_size, {})
                            sequences = \
                            sequences_to_intervals(in_seq, models[sg.reference_model.__name__].Vdata, False)[0]

                            # additional information for evaluation
                            additional_infos = dict()
                            additional_infos[sg.reference_model.__name__] = {'execution_time': 0.0, 'data': None}

                            for score in ['BIC', 'AIC', 'Bdeu', 'K2']:
                                print('score: ' + str(score) + '...')
                                L().log.info('score: ' + str(score) + '...')

                                for temporal_threshold in np.arange(0.0, 2.5, 0.5):
                                    print('temporal_threshold: ' + str(temporal_threshold) + '...')
                                    L().log.info('temporal_threshold: ' + str(temporal_threshold) + '...')

                                    # STRUCTURE DISCOVERER SETUP
                                    sd.score = score
                                    sd.max_time_difference = temporal_threshold

                                    sd_name = 'SBTreeDiscoverer_' + score + '_TH_' + str(temporal_threshold)
                                    if sd_name not in eval_results.get(number_of_temp_nodes):  # initialise metrics_dict
                                        metrics_dict = dict((metric, []) for metric in metrics)
                                        eval_results.get(number_of_temp_nodes).update({sd_name: metrics_dict})
                                        discovery_algorithms.add(sd_name)
                                    model_name = sd_name + ' (' + str(iteration) + ')'

                                    # RUN ALGORITHM
                                    L().log.info('----------------------------------------------------------')
                                    print('Run approach ' + model_name + '.')
                                    L().log.info('Run approach ' + model_name + '.')
                                    ping = clock()
                                    nodes, edges = sd.discover_structure(sequences)
                                    L().log.info('Nodes: ' + str(nodes))
                                    L().log.info('Edges: ' + str(edges))
                                    execution_time = clock() - ping
                                    additional_infos[model_name] = {'execution_time': execution_time, 'data': sd.data,
                                                                    'psi_execution_time': sd.parent_set_identification_time,
                                                                    'so_execution_time': sd.structure_optimization_time}
                                    L().log.info('Execution time: ' + str(execution_time))
                                    L().log.info('----------------------------------------------------------')

                                    # CREATE TSCBN
                                    skel = GraphSkeleton()
                                    skel.V = nodes
                                    skel.E = edges
                                    skel.toporder()
                                    model = TSCBN("", skel, models[sg.reference_model.__name__].Vdata, unempty=True,
                                                  forbid_never=True, discrete_only=True)

                                    # EVALUATION
                                    eval_result = ev.evaluate(model_dict={model_name: model},
                                                              reference=models[sg.reference_model.__name__],
                                                              additional_infos=additional_infos)
                                    ev.print_eval_results(eval_results=eval_result, specs=specifications, to_csv=True)
                                    for metric, value in eval_result[model_name].items():
                                        eval_results[number_of_temp_nodes][sd_name][metric].append(value)
                                    pass
                                pass
                            pass
                        pass
                    experiment_name = 'ParameterTmpNodesExperiment_EPO_' + str(edges_per_object) + '_Sig_' + \
                                      str(number_of_signals)
                    relevant_metrics = ["num-add-edges", "num-del-edges", "shd", "num-add-edges-skel",
                                        "num-del-edges-skel", "shd-skel", "kld", "execution-time", "psi-execution-time",
                                        "so-execution-time"]
                    write_pgfplots_data(experiment_name, eval_results, relevant_metrics, discovery_algorithms,
                                        numbers_of_temp_nodes, 'number_of_temp_nodes', target_path)
                pass
            pass

            if parameter_signals_experiment:
                # 2nd experiment: Increase number of signals

                if edges_per_object == 3:
                    continue  # TODO: remove this, when choosing a maximal number of signals larger than 5

                # EVALUATOR SETUP
                ev = StructureEvaluator(True)
                ev.set_output_path(os.path.join(target_path, r"structure_eval_%s.csv" % strftime("%Y_%m_%d-%H_%M_%S", localtime())))
                metrics = ["add-edges", "del-edges", "num-add-edges", "num-del-edges", "shd", "add-edges-skel",
                           "del-edges-skel", "num-add-edges-skel", "num-del-edges-skel", "shd-skel", "kld",
                           "execution-time", "psi-execution-time", "so-execution-time"]
                for metric in metrics:
                    ev.add_metric(metric)
                eval_results = dict()
                discovery_algorithms = set()

                for number_of_temp_nodes in [3, 5]:
                    print('number_of_temp_nodes: ' + str(number_of_temp_nodes) + '...')
                    L().log.info('number_of_temp_nodes: ' + str(number_of_temp_nodes) + '...')

                    numbers_of_signals = [2, 3, 4, 5]
                    evaluated_numbers_of_signals = copy.deepcopy(numbers_of_signals)
                    for number_of_signals in numbers_of_signals:
                        print('number_of_signals: ' + str(number_of_signals) + '...')
                        L().log.info('number_of_signals: ' + str(number_of_signals) + '...')

                        if edges_per_object >= number_of_signals:
                            evaluated_numbers_of_signals.remove(number_of_signals)
                            continue

                        # NODE SETTINGS
                        sg.set_node_range(min_objects=number_of_signals, max_objects=number_of_signals,
                                          min_temp_nodes=number_of_temp_nodes, max_temp_nodes=number_of_temp_nodes,
                                          min_states=3, max_states=3)

                        eval_results.update({number_of_signals: dict()})

                        for iteration in range(iterations):
                            print('iteration: ' + str(iteration) + '...')
                            L().log.info('iteration: ' + str(iteration) + '...')

                            # SAMPLE DATA
                            models, specifications = sg.run_next_testcase()
                            in_seq = models[sg.reference_model.__name__].randomsample(1000, {})
                            sequences = \
                            sequences_to_intervals(in_seq, models[sg.reference_model.__name__].Vdata, False)[0]

                            # additional information for evaluation
                            additional_infos = dict()
                            additional_infos[sg.reference_model.__name__] = {'execution_time': 0.0, 'data': None}

                            for score in ['BIC', 'AIC', 'Bdeu', 'K2']:
                                print('score: ' + str(score) + '...')
                                L().log.info('score: ' + str(score) + '...')

                                for temporal_threshold in np.arange(0.0, 2.5, 0.5):
                                    print('temporal_threshold: ' + str(temporal_threshold) + '...')
                                    L().log.info('temporal_threshold: ' + str(temporal_threshold) + '...')

                                    # STRUCTURE DISCOVERER SETUP
                                    sd.score = score
                                    sd.max_time_difference = temporal_threshold

                                    sd_name = 'SBTreeDiscoverer_' + score + '_TH_' + str(temporal_threshold)
                                    if sd_name not in eval_results.get(number_of_signals):  # initialise metrics_dict
                                        metrics_dict = dict((metric, []) for metric in metrics)
                                        eval_results.get(number_of_signals).update({sd_name: metrics_dict})
                                        discovery_algorithms.add(sd_name)
                                    model_name = sd_name + ' (' + str(iteration) + ')'

                                    # RUN ALGORITHM
                                    L().log.info('----------------------------------------------------------')
                                    print('Run approach ' + model_name + '.')
                                    L().log.info('Run approach ' + model_name + '.')
                                    ping = clock()
                                    nodes, edges = sd.discover_structure(sequences)
                                    L().log.info('Nodes: ' + str(nodes))
                                    L().log.info('Edges: ' + str(edges))
                                    execution_time = clock() - ping
                                    additional_infos[model_name] = {'execution_time': execution_time, 'data': sd.data,
                                                                    'psi_execution_time': sd.parent_set_identification_time,
                                                                    'so_execution_time': sd.structure_optimization_time}
                                    L().log.info('Execution time: ' + str(execution_time))
                                    L().log.info('----------------------------------------------------------')

                                    # CREATE TSCBN
                                    skel = GraphSkeleton()
                                    skel.V = nodes
                                    skel.E = edges
                                    skel.toporder()
                                    model = TSCBN("", skel, models[sg.reference_model.__name__].Vdata, unempty=True,
                                                  forbid_never=True, discrete_only=True)

                                    # EVALUATION
                                    eval_result = ev.evaluate(model_dict={model_name: model},
                                                              reference=models[sg.reference_model.__name__],
                                                              additional_infos=additional_infos)
                                    ev.print_eval_results(eval_results=eval_result, specs=specifications, to_csv=True)
                                    for metric, value in eval_result[model_name].items():
                                        eval_results[number_of_signals][sd_name][metric].append(value)
                                    pass
                                pass
                            pass
                        pass
                    experiment_name = 'ParameterSignalsExperiment_EPO_' + str(edges_per_object) + '_TmpNodes_' + \
                                      str(number_of_temp_nodes)
                    relevant_metrics = ["num-add-edges", "num-del-edges", "shd", "num-add-edges-skel",
                                        "num-del-edges-skel", "shd-skel", "kld", "execution-time", "psi-execution-time",
                                        "so-execution-time"]
                    write_pgfplots_data(experiment_name, eval_results, relevant_metrics, discovery_algorithms,
                                        evaluated_numbers_of_signals, 'num_signals', target_path)
                pass
            pass
        pass
    pass

    # ----------------------------------------------------------------------------------------
    #      Experiments with all algorithms
    # ----------------------------------------------------------------------------------------
    # 1st experiment: increase number of temporal nodes
    if comparison_experiment_temp_nodes:
        # EDGE SETTINGS
        sg.set_connection_ranges(min_edges_per_object=2, max_edges_per_object=2,
                                 min_percent_inter=1.0, max_percent_inter=1.0)

        # EVALUATOR SETUP
        ev = StructureEvaluator(True)
        ev.set_output_path(os.path.join(target_path, r"structure_eval_%s.csv" % strftime("%Y_%m_%d-%H_%M_%S", localtime())))
        metrics = ["add-edges", "del-edges", "num-add-edges", "num-del-edges", "shd", "add-edges-skel",
                   "del-edges-skel", "num-add-edges-skel", "num-del-edges-skel", "shd-skel", "kld", "execution-time"]
        for metric in metrics:
            ev.add_metric(metric)
        eval_results = dict()

        for number_of_signals in [3, 4]:
            print('number_of_signals: ' + str(number_of_signals) + '...')
            L().log.info('number_of_signals: ' + str(number_of_signals) + '...')

            discovery_algorithms = set()

            numbers_of_temp_nodes = [2, 3, 4, 5, 6, 7, 8]
            for number_of_temp_nodes in numbers_of_temp_nodes:
                print('number_of_temp_nodes: ' + str(number_of_temp_nodes) + '...')
                L().log.info('number_of_temp_nodes: ' + str(number_of_temp_nodes) + '...')

                # NODE SETTINGS
                sg.set_node_range(min_objects=number_of_signals, max_objects=number_of_signals,
                                  min_temp_nodes=number_of_temp_nodes, max_temp_nodes=number_of_temp_nodes,
                                  min_states=3, max_states=3)

                eval_results.update({number_of_temp_nodes: dict()})
                metrics_dict = dict((metric, []) for metric in metrics)

                # ---------------------------------------------------
                #   RUN Structure Discovery several times
                # ---------------------------------------------------
                for iteration in range(iterations):
                    print('iteration: ' + str(iteration) + '...')
                    L().log.info('iteration: ' + str(iteration) + '...')

                    # SAMPLE DATA
                    models, specifications = sg.run_next_testcase()
                    in_seq = models[sg.reference_model.__name__].randomsample(sample_size, {})
                    sequences = sequences_to_intervals(in_seq, models[sg.reference_model.__name__].Vdata, False)[0]

                    additional_infos = dict()
                    additional_infos[sg.reference_model.__name__] = {'execution_time': 0.0, 'data': None}

                    # ---------------------------------------------------
                    #   Discovery Algorithm
                    # ---------------------------------------------------
                    for sd_name, sd in get_structure_discovery_algorithms():

                        # LIMITATIONS DUE TO RUNTIME PROBLEMS
                        # TODO: run all algorithms for all networks on a better hardware
                        if str.startswith(sd_name, 'Astar') and number_of_signals * number_of_temp_nodes > 16:
                            print('Network to large for A* algorithm.')
                            continue
                        if str.startswith(sd_name, 'PC') and number_of_signals * number_of_temp_nodes > 24:
                            print('Network to large for PC algorithm.')
                            continue

                        discovery_algorithms.add(sd_name)
                        if sd_name not in eval_results.get(number_of_temp_nodes):
                            eval_results.get(number_of_temp_nodes).update({sd_name: copy.deepcopy(metrics_dict)})

                        model_name = sd_name + ' (' + str(iteration) + ')'
                        L().log.info('----------------------------------------------------------')
                        print('Run approach ' + model_name + '.')
                        L().log.info('Run approach ' + model_name + '.')

                        ping = clock()
                        nodes, edges = sd.discover_structure(sequences)
                        L().log.info('Nodes: ' + str(nodes))
                        L().log.info('Edges: ' + str(edges))
                        execution_time = clock() - ping
                        additional_infos[model_name] = {'execution_time': execution_time, 'data': sd.data}
                        L().log.info('Execution time: ' + str(execution_time))
                        L().log.info('----------------------------------------------------------')

                        # create TSCBN
                        skel = GraphSkeleton()
                        skel.V = nodes
                        skel.E = edges
                        skel.toporder()
                        model = TSCBN("", skel, models[sg.reference_model.__name__].Vdata, unempty=True,
                                      forbid_never=True, discrete_only=True)

                        # ----------------------------------------------------------------------------------------
                        #       EVALUATION
                        # ----------------------------------------------------------------------------------------
                        eval_result = ev.evaluate(model_dict={model_name: model},
                                                  reference=models[sg.reference_model.__name__],
                                                  additional_infos=additional_infos)
                        ev.print_eval_results(eval_results=eval_result, specs=specifications, to_csv=True)
                        for metric, value in eval_result[model_name].items():
                            eval_results[number_of_temp_nodes][sd_name][metric].append(value)
                        pass
                    pass
                pass
            experiment_name = 'TempNodesExperiment_Sig_' + str(number_of_signals)
            relevant_metrics = ["num-add-edges", "num-del-edges", "shd", "num-add-edges-skel", "num-del-edges-skel",
                                "shd-skel", "kld", "execution-time"]
            write_pgfplots_data(experiment_name, eval_results, relevant_metrics, discovery_algorithms,
                                numbers_of_temp_nodes, 'number_of_temp_nodes', target_path)

    # 2nd experiment: increase number of signals
    if comparison_experiment_signals:
        # EDGE SETTINGS
        sg.set_connection_ranges(min_edges_per_object=2, max_edges_per_object=2,
                                 min_percent_inter=1.0, max_percent_inter=1.0)

        # EVALUATOR SETUP
        ev = StructureEvaluator(True)
        ev.set_output_path(os.path.join(target_path, r"structure_eval_%s.csv" % strftime("%Y_%m_%d-%H_%M_%S", localtime())))
        metrics = ["add-edges", "del-edges", "num-add-edges", "num-del-edges", "shd", "add-edges-skel",
                   "del-edges-skel", "num-add-edges-skel", "num-del-edges-skel", "shd-skel", "kld", "execution-time",
                   "psi-execution-time", "so-execution-time"]
        for metric in metrics:
            ev.add_metric(metric)
        eval_results = dict()

        for number_of_temp_nodes in [3]:  # TODO: run with larger numbers on better hardware
            print('number_of_temp_nodes: ' + str(number_of_temp_nodes) + '...')
            L().log.info('number_of_temp_nodes: ' + str(number_of_temp_nodes) + '...')

            discovery_algorithms = set()

            numbers_of_signals = [3, 4, 5, 6, 7, 8]
            for number_of_signals in numbers_of_signals:
                print('number_of_signals: ' + str(number_of_signals) + '...')
                L().log.info('number_of_signals: ' + str(number_of_signals) + '...')

                # NODE SETTINGS
                sg.set_node_range(min_objects=number_of_signals, max_objects=number_of_signals,
                                  min_temp_nodes=number_of_temp_nodes, max_temp_nodes=number_of_temp_nodes,
                                  min_states=3, max_states=3)

                eval_results.update({number_of_signals: dict()})
                metrics_dict = dict((metric, []) for metric in metrics)

                # ---------------------------------------------------
                #   RUN Structure Discovery several times
                # ---------------------------------------------------
                for iteration in range(iterations):
                    print('iteration: ' + str(iteration) + '...')
                    L().log.info('iteration: ' + str(iteration) + '...')

                    # SAMPLE DATA
                    models, specifications = sg.run_next_testcase()
                    in_seq = models[sg.reference_model.__name__].randomsample(sample_size, {})
                    sequences = sequences_to_intervals(in_seq, models[sg.reference_model.__name__].Vdata, False)[0]

                    additional_infos = dict()
                    additional_infos[sg.reference_model.__name__] = {'execution_time': 0.0, 'data': None,
                                                                     'psi-execution-time': 0.0,
                                                                     'so-execution-time': 0.0}

                    # ---------------------------------------------------
                    #   Discovery Algorithm
                    # ---------------------------------------------------
                    for sd_name, sd in get_structure_discovery_algorithms():

                        # LIMITATIONS DUE TO RUNTIME PROBLEMS
                        # TODO: run all algorithms for all networks on a better hardware
                        if str.startswith(sd_name, 'Astar') and number_of_signals * number_of_temp_nodes > 16:
                            print('Network to large for A* algorithm.')
                            continue
                        if str.startswith(sd_name, 'PC') and number_of_signals * number_of_temp_nodes > 24:
                            print('Network to large for PC algorithm.')
                            continue
                        if str.startswith(sd_name, 'sbPTM') and number_of_signals * number_of_temp_nodes > 30:
                            print('Network to large for PTM algorithm.')
                            continue
                        if str.startswith(sd_name, 'cbPTM') and number_of_signals * number_of_temp_nodes > 30:
                            print('Network to large for PTM algorithm.')
                            continue

                        discovery_algorithms.add(sd_name)
                        if sd_name not in eval_results.get(number_of_signals):
                            eval_results.get(number_of_signals).update({sd_name: copy.deepcopy(metrics_dict)})

                        model_name = sd_name + ' (' + str(iteration) + ')'
                        L().log.info('----------------------------------------------------------')
                        print('Run approach ' + model_name + '.')
                        L().log.info('Run approach ' + model_name + '.')

                        ping = clock()
                        nodes, edges = sd.discover_structure(sequences)
                        L().log.info('Nodes: ' + str(nodes))
                        L().log.info('Edges: ' + str(edges))
                        execution_time = clock() - ping
                        additional_infos[model_name] = {'execution_time': execution_time, 'data': sd.data,
                                                        'psi_execution_time': 0.0, 'so_execution_time': 0.0}
                        if sd.parent_set_identification_time and sd.structure_optimization_time:
                            additional_infos[model_name].update(
                                {'psi_execution_time': sd.parent_set_identification_time,
                                 'so_execution_time': sd.structure_optimization_time})
                        L().log.info('Execution time: ' + str(execution_time))
                        L().log.info('----------------------------------------------------------')

                        # create TSCBN
                        skel = GraphSkeleton()
                        skel.V = nodes
                        skel.E = edges
                        skel.toporder()
                        model = TSCBN("", skel, models[sg.reference_model.__name__].Vdata, unempty=True,
                                      forbid_never=True, discrete_only=True)

                        # ----------------------------------------------------------------------------------------
                        #       EVALUATION
                        # ----------------------------------------------------------------------------------------
                        eval_result = ev.evaluate(model_dict={model_name: model},
                                                  reference=models[sg.reference_model.__name__],
                                                  additional_infos=additional_infos)
                        ev.print_eval_results(eval_results=eval_result, specs=specifications, to_csv=True)
                        for metric, value in eval_result[model_name].items():
                            eval_results[number_of_signals][sd_name][metric].append(value)
                        pass
                    pass
                pass
            experiment_name = 'SignalExperiment_TmpNodes_' + str(number_of_temp_nodes)
            relevant_metrics = ["num-add-edges", "num-del-edges", "shd", "num-add-edges-skel", "num-del-edges-skel",
                                "shd-skel", "kld", "execution-time", "psi-execution-time", "so-execution-time"]
            write_pgfplots_data(experiment_name, eval_results, relevant_metrics, discovery_algorithms,
                                numbers_of_signals, 'number_of_signals', target_path)

    # 3rd experiment: different values for the state change probability
    if comparison_experiment_scp:
        # EDGE SETTINGS
        sg.set_connection_ranges(min_edges_per_object=2, max_edges_per_object=2,
                                 min_percent_inter=1.0, max_percent_inter=1.0)

        # EVALUATOR SETUP
        ev = StructureEvaluator(True)
        ev.set_output_path(os.path.join(target_path, r"structure_eval_%s.csv" % strftime("%Y_%m_%d-%H_%M_%S", localtime())))
        metrics = ["add-edges", "del-edges", "num-add-edges", "num-del-edges", "shd", "add-edges-skel",
                   "del-edges-skel", "num-add-edges-skel", "num-del-edges-skel", "shd-skel", "kld",
                   "execution-time"]
        for metric in metrics:
            ev.add_metric(metric)
        eval_results = dict()

        for number_of_temp_nodes in [3, 4]:
            print('number_of_temp_nodes: ' + str(number_of_temp_nodes) + '...')
            L().log.info('number_of_temp_nodes: ' + str(number_of_temp_nodes) + '...')

            # NODE SETTINGS
            sg.set_node_range(min_objects=3, max_objects=3,
                              min_temp_nodes=number_of_temp_nodes, max_temp_nodes=number_of_temp_nodes,
                              min_states=2, max_states=4)
            sg.set_connection_ranges(min_edges_per_object=2, max_edges_per_object=3, min_percent_inter=0.5,
                                     max_percent_inter=1.0)

            discovery_algorithms = set()

            state_change_probabilities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for state_change_probability in state_change_probabilities:
                print('state_change_probability: ' + str(state_change_probability) + '...')
                L().log.info('state_change_probability: ' + str(state_change_probability) + '...')

                sg.set_state_change_probability(min_probability=state_change_probability,
                                                max_probability=state_change_probability)

                eval_results.update({state_change_probability: dict()})
                metrics_dict = dict((metric, []) for metric in metrics)

                # ---------------------------------------------------
                #   RUN Structure Discovery several times
                # ---------------------------------------------------
                for iteration in range(iterations):
                    print('iteration: ' + str(iteration) + '...')
                    L().log.info('iteration: ' + str(iteration) + '...')

                    # SAMPLE DATA
                    models, specifications = sg.run_next_testcase()
                    in_seq = models[sg.reference_model.__name__].randomsample(sample_size, {})
                    sequences = sequences_to_intervals(in_seq, models[sg.reference_model.__name__].Vdata, False)[0]

                    additional_infos = dict()
                    additional_infos[sg.reference_model.__name__] = {'execution_time': 0.0, 'data': None}

                    # ---------------------------------------------------
                    #   Discovery Algorithm
                    # ---------------------------------------------------
                    for sd_name, sd in get_structure_discovery_algorithms():

                        # LIMITATIONS DUE TO RUNTIME PROBLEMS
                        # TODO: run all algorithms for all networks on a better hardware
                        if str.startswith(sd_name, 'Astar') and 3 * number_of_temp_nodes > 16:
                            print('Network to large for A* algorithm.')
                            continue

                        discovery_algorithms.add(sd_name)
                        if sd_name not in eval_results.get(state_change_probability):
                            eval_results.get(state_change_probability).update({sd_name: copy.deepcopy(metrics_dict)})

                        model_name = sd_name + ' (' + str(iteration) + ')'
                        L().log.info('----------------------------------------------------------')
                        print('Run approach ' + model_name + '.')
                        L().log.info('Run approach ' + model_name + '.')

                        ping = clock()
                        nodes, edges = sd.discover_structure(sequences)
                        L().log.info('Nodes: ' + str(nodes))
                        L().log.info('Edges: ' + str(edges))
                        execution_time = clock() - ping
                        additional_infos[model_name] = {'execution_time': execution_time, 'data': sd.data}
                        L().log.info('Execution time: ' + str(execution_time))
                        L().log.info('----------------------------------------------------------')

                        # create TSCBN
                        skel = GraphSkeleton()
                        skel.V = nodes
                        skel.E = edges
                        skel.toporder()
                        model = TSCBN("", skel, models[sg.reference_model.__name__].Vdata, unempty=True,
                                      forbid_never=True, discrete_only=True)

                        # ----------------------------------------------------------------------------------------
                        #       EVALUATION
                        # ----------------------------------------------------------------------------------------
                        eval_result = ev.evaluate(model_dict={model_name: model},
                                                  reference=models[sg.reference_model.__name__],
                                                  additional_infos=additional_infos)
                        ev.print_eval_results(eval_results=eval_result, specs=specifications, to_csv=True)
                        for metric, value in eval_result[model_name].items():
                            eval_results[state_change_probability][sd_name][metric].append(value)
                        pass
                    pass
                pass
            experiment_name = 'SCP_Experiment_Sig_3_TmpNodes_' + str(number_of_temp_nodes)
            relevant_metrics = ["num-add-edges", "num-del-edges", "shd", "num-add-edges-skel", "num-del-edges-skel",
                                "shd-skel", "kld", "execution-time"]
            write_pgfplots_data(experiment_name, eval_results, relevant_metrics, discovery_algorithms,
                                state_change_probabilities, 'state_change_probability', target_path)
