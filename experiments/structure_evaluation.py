import copy
from time import localtime, strftime, clock
import os
import numpy as np
from sacred import Experiment

from _include.discoverer.a_star_discoverer import AStarDiscoverer
from _include.discoverer.hill_climb_discoverer import HillClimbDiscoverer
from _include.discoverer.novel_structure_discoverer import NovelStructureDiscoverer
from _include.discoverer.pc_discoverer import PCDiscoverer
from _include.discoverer.pc_tree_discoverer import PCTreeDiscoverer
from _include.discoverer.sb_tree_discoverer import SBTreeDiscoverer
from _include.evaluation.structure_evaluator import StructureEvaluator
from _include.m_libpgm.graphskeleton import GraphSkeleton
from _include.structure.structure_generator import TestStructureEnum, StructureGenerator
from _include.structure._tscbn_structure_model import TSCBNStructureModel
from _include.toolkit import sequences_to_intervals
from general.setup import Discoverers
from network.tscbn import TSCBN

ex = Experiment('structure_discovery')


def get_sd_approach(discoverer_id, sb_min_out_degree, sb_k_infrequent, sb_score, sb_max_time_difference, pc_min_out_degree,
                    pc_k_infrequent,pc_alpha,pc_max_time_difference, pcd_alpha, pcd_max_reach, astar_score, ghc_score, ghc_tabu_length, novel_filtering,novel_k_infrequent,novel_alpha,novel_draw_it,novel_min_out_degree,novel_max_reach, pc_chi):

    if discoverer_id == Discoverers.SB_TREE:
        return SBTreeDiscoverer(min_out_degree=sb_min_out_degree, k_infrequent=sb_k_infrequent, approach='parent_graph', parallel=False, score=sb_score,
                          max_time_difference=sb_max_time_difference) # parent graph approach means exact score optimization (but not exhaustive)

    if discoverer_id == Discoverers.PC_TREE:
        return PCTreeDiscoverer(min_out_degree=pc_min_out_degree, k_infrequent=pc_k_infrequent, parallel=False, alpha=pc_alpha,
                          max_time_difference=pc_max_time_difference, optimization_chi_square=False)

    if discoverer_id == Discoverers.PC_TREE_VAR:
        return PCTreeDiscoverer(min_out_degree=pc_min_out_degree, k_infrequent=pc_k_infrequent, parallel=False, alpha=pc_alpha,
                          max_time_difference=pc_max_time_difference, chi_square_thresh=pc_chi, optimization_chi_square=True)

    if discoverer_id == Discoverers.PC_DISCOVER:
        return PCDiscoverer(alpha=pcd_alpha, max_reach=pcd_max_reach)

    if discoverer_id == Discoverers.A_STAR:
        return AStarDiscoverer(score=astar_score)

    if discoverer_id == Discoverers.HILL_CLIMB:
        return HillClimbDiscoverer(score=ghc_score, tabu_length=ghc_tabu_length)

    if discoverer_id == Discoverers.PREFIX:
        return NovelStructureDiscoverer(filtering=novel_filtering, k_infrequent=novel_k_infrequent, alpha=novel_alpha, draw=novel_draw_it, max_reach=novel_max_reach, min_out_degree=novel_min_out_degree,  draw_only_result=novel_draw_it)

    return None

def initialize_generator(min_per_object_gap, max_per_object_gap, temporal_variance, dbn_tolerance, sc_probability, edges_per_object, inter_edge_percent,
                         number_of_signals, number_of_temp_nodes):
    sg = StructureGenerator(test_type=TestStructureEnum.SPECIFICATION)
    sg.add_base_structure_models([TSCBNStructureModel])
    sg.reference_model = TSCBNStructureModel

    # TIME SETTINGS (fixed for all experiments)
    sg.set_temporal_range(min_per_object_gap=min_per_object_gap, max_per_object_gap=max_per_object_gap)
    sg.set_temporal_variance(temporal_variance)
    sg.set_dbn_tolerance(dbn_tolerance)

    # PROBABILITY SETTINGS (fixed for all experiments)
    sg.set_state_change_probability(min_probability=sc_probability, max_probability=sc_probability)

    # EDGE SETTINGS
    sg.set_connection_ranges(min_edges_per_object=edges_per_object, max_edges_per_object=edges_per_object,
                             min_percent_inter=inter_edge_percent, max_percent_inter=inter_edge_percent)
    # NODE Settings
    sg.set_node_range(min_objects=number_of_signals, max_objects=number_of_signals,
                      min_temp_nodes=number_of_temp_nodes, max_temp_nodes=number_of_temp_nodes,
                      min_states=3, max_states=3)

    return sg

def initialize_evaluator():
    ev = StructureEvaluator(True)
    metrics = ["add-edges", "del-edges", "num-add-edges", "num-del-edges", "shd", "add-edges-skel",
               "del-edges-skel", "num-add-edges-skel", "num-del-edges-skel", "shd-skel", "kld",
               "execution-time", "psi-execution-time", "so-execution-time"]
    for metric in metrics:ev.add_metric(metric)
    return ev

def hw_limitation_reached(approach, number_of_signals, number_of_temp_nodes):
    if approach == Discoverers.A_STAR and number_of_signals * number_of_temp_nodes > 16:
        print('Network too large for A* algorithm.')
        return True
    if approach in [Discoverers.PC_DISCOVER, Discoverers.PC_TREE] and number_of_signals * number_of_temp_nodes > 24:
        print('Network too large for PC algorithm.')
        return True
    return False

@ex.config
def config():
    approach = Discoverers.SB_TREE
    sample_size = 5000 # number of sequences to learn from
    iterations = 25 # number of iterations

    # Structure Generator Settings
    # (i) Temporal
    min_per_object_gap = 0.5
    max_per_object_gap = 1.0
    temporal_variance = 0.001
    dbn_tolerance = 0.1

    # (ii) Structure
    sc_probability = 0.95
    edges_per_object = 3 # DOES prob. do nothing
    inter_edge_percent = 0.5 # NUMBER INTEREDGES = round(inter_edge_percent*number_of_temp_nodes)*number_of_signals  2 0.5 * 5 * 5 = 10   2 1.0 = 25   1 0.5 = 10  1 0.25 = 5   2 0.25 = 5
    number_of_signals = 4 # 3 0.5 8 5 = 16  # 3 0.5 8 10 = 40   # 3 0.75 8 10 = 64
    number_of_temp_nodes = 5

    # Score
    #temporal_thresholds = np.arange(0.0, 2.5, 0.5)

    # Approach Specific
    novel_filtering = True # Set to True if trie and DAWG should be filtered.
    novel_k_infrequent = 0.1 # An edge in the trie gets filtered when it appeared less than k_infrequent times the in frequencies of the node.
    novel_alpha = 0.1 #Significance level used in the G^2 test.
    novel_draw_it = False
    novel_max_reach = 2 # Maximal size of the condition set in the PC algorithm like procedure.
    novel_min_out_degree = 0.25 # A node in the trie gets filtered when the sum of outgoing frequencies is smaller than min_out_degree times the incoming frequencies. This is necessary to assure a good transformation to the dawg. Example: A --100--> B --100--> C --1--> D, D will be filtered

    sb_min_out_degree = 0.1
    sb_k_infrequent = 0.1
    sb_score = 'BIC' # 'BIC', 'AIC', 'Bdeu', 'K2'
    sb_max_time_difference = 1.0 # 1.0 1.5

    pc_min_out_degree=0.1
    pc_k_infrequent=0.1
    pc_alpha=0.01 # 0.01 0.05 0.1
    pc_max_time_difference=1.0 # 1.0 1.5
    pc_chi = 1.0 # chi square test

    pcd_alpha = 0.01 # 0.01 0.05 0.1
    pcd_max_reach = 2 #1 2

    astar_score = 'BIC' # 'BIC', 'AIC', 'Bdeu', 'K2'

    ghc_tabu_length = 0 # 0
    ghc_score = 'BIC' # 'BIC', 'AIC', 'Bdeu', 'K2'

    mmhc_score = 'AIC' # 'BIC', 'AIC', 'Bdeu', 'K2'
    mmhc_alpha = 0.01 # 0.01 0.05
    mmhc_max_reach = 0 # 0

@ex.automain
def experiment_discovery(_run, approach, sample_size, iterations, min_per_object_gap, max_per_object_gap, temporal_variance, dbn_tolerance,
                         sc_probability, edges_per_object, inter_edge_percent, number_of_signals,number_of_temp_nodes, sb_min_out_degree,
                         sb_k_infrequent, sb_score, sb_max_time_difference, pc_min_out_degree, pc_k_infrequent,pc_alpha,pc_max_time_difference, pcd_alpha,
                         pcd_max_reach, astar_score, ghc_score, ghc_tabu_length, novel_filtering,novel_k_infrequent,novel_alpha,novel_draw_it, novel_max_reach, novel_min_out_degree, pc_chi):


    # ----------------------------------------------------------------------------------------
    #      Setup
    # ----------------------------------------------------------------------------------------
    if edges_per_object >= number_of_signals:return
    # Generator Setup
    sg = initialize_generator(min_per_object_gap, max_per_object_gap, temporal_variance, dbn_tolerance, sc_probability, edges_per_object, inter_edge_percent,
                         number_of_signals, number_of_temp_nodes)
    # SD Approach
    sd = get_sd_approach(approach, sb_min_out_degree, sb_k_infrequent, sb_score, sb_max_time_difference, pc_min_out_degree,
                    pc_k_infrequent,pc_alpha,pc_max_time_difference, pcd_alpha, pcd_max_reach, astar_score, ghc_score, ghc_tabu_length, novel_filtering,novel_k_infrequent,novel_alpha,novel_draw_it,novel_min_out_degree, novel_max_reach, pc_chi)
    # Evaluation Metrics
    ev = initialize_evaluator()


    # ----------------------------------------------------------------------------------------
    #      Run Experiment
    # ----------------------------------------------------------------------------------------
    eval_results = dict()
    for iteration in range(iterations):
        print('iteration: ' + str(iteration+1) + '...')

        # SAMPLE DATA
        models, specifications = sg.run_next_testcase()
        print("NUMBER INTER EDGES: %s" % str(len([e for e in models["TSCBNStructureModel"].E  if e[0].split("_")[0] != e[1].split("_")[0] and not str.startswith(e[1], "dL_")])))
        in_seq = models[sg.reference_model.__name__].randomsample(sample_size, {})
        sequences = sequences_to_intervals(in_seq, models[sg.reference_model.__name__].Vdata, False)[0]

        additional_infos = dict()
        additional_infos[sg.reference_model.__name__] = {'execution_time': 0.0, 'data': None}

        # LIMITATIONS DUE TO RUNTIME PROBLEMS
        if hw_limitation_reached(approach, number_of_signals, number_of_temp_nodes):continue

        # RUN DISCOVERY
        ping = clock()
        nodes, edges = sd.discover_structure(sequences)
        execution_time = clock() - ping

        # CREATE GROUND TRUTH TSCBN
        skel = GraphSkeleton()
        skel.V = nodes
        skel.E = edges
        skel.toporder()
        model = TSCBN("", skel, models[sg.reference_model.__name__].Vdata, unempty=True, forbid_never=True, discrete_only=True)

        # ----------------------------------------------------------------------------------------
        #       Run Evaluation current Iteration
        # ----------------------------------------------------------------------------------------
        model_name = str(approach) + ' (' + str(iteration) + ')'
        additional_infos[model_name] = {'execution_time': execution_time, 'data': sd.data}
        eval_result = ev.evaluate(model_dict={model_name: model},
                                  reference=models[sg.reference_model.__name__],
                                  additional_infos=additional_infos)
        #ev.print_eval_results(eval_results=eval_result, specs=specifications, to_csv=True)
        for metric, value in eval_result[model_name].items():
            if not metric in eval_results:eval_results[metric] = []
            eval_results[metric].append(value)
            try:
                float(value)
                _run.log_scalar(metric, value)
            except:pass


    # ----------------------------------------------------------------------------------------
    #       Run Evaluation average over all Iteration
    # ----------------------------------------------------------------------------------------
    for metric in eval_results:
        try:
            float(eval_results[metric])
            _run.log_scalar("avg_%s" % metric, np.mean(eval_results[metric]))
        except:pass


