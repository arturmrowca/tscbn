import datetime, os
import numpy as np

from _include.evaluation.structure_evaluator import StructureEvaluator
from _include.structure.ctbn_structure_model import CTBNStructureModel
from _include.structure.dbn_structure_model import DBNStructureModel
from _include.structure.structure_generator import StructureGenerator, TestStructureEnum
from _include.structure._tscbn_structure_model import TSCBNStructureModel
from general.log import Log as L

def run_vary_structure(target_path):
    # ----------------------------------------------
    # GRID
    # ----------------------------------------------
    object_nr = [5, 10, 20, 30, 40]  # (from, to, steps)
    nodes_per_tv = [2, 50, 2]   # (from, to, steps)
    states_per_tv = [2, 6, 2]   # (from, to, steps)

    edges_per_tv = [3, 3, 2]
    percentage_inter = [0.8, 0.8, 0.2]
    per_object_gap = [0.5, 0.5, 0.01] # range is still within selected and selected + 0.5

    t_variance_tscbn = [0.1, 0.1, 0.02]
    dbn_tolerance = [0.02, 0.02, 0.02]

    state_change_prob = [1.0, 1.0, 0.01]
    append_csv = False
    eval_models = [CTBNStructureModel, DBNStructureModel, TSCBNStructureModel]

    id_time = datetime.datetime.now().strftime("%I_%M%p_%d_%B_%Y")
    out_path = os.path.join(target_path, r"model_evaluation_%s.csv" % id_time)
    print("store to %s" % out_path)

    run = 1
    expected_runs = 1
    expected_runs *= len(object_nr)
    expected_runs *= len(list(range(nodes_per_tv[0], nodes_per_tv[1] + 1, nodes_per_tv[2])))
    expected_runs *= len(list(range(states_per_tv[0], states_per_tv[1] + 1, states_per_tv[2])))
    expected_runs *= len(list(range(edges_per_tv[0], edges_per_tv[1] + 1, edges_per_tv[2])))
    expected_runs *= len(list(np.arange(percentage_inter[0], percentage_inter[1] + 0.000001, percentage_inter[2])))
    expected_runs *= len(list(np.arange(per_object_gap[0], per_object_gap[1] + 0.00000001, per_object_gap[2])))
    expected_runs *= len(list(np.arange(t_variance_tscbn[0], t_variance_tscbn[1] + 0.00000001, t_variance_tscbn[2])))
    expected_runs *= len(list(np.arange(dbn_tolerance[0], dbn_tolerance[1] + 0.00000001, dbn_tolerance[2])))
    expected_runs *= len(list(np.arange(state_change_prob[0], state_change_prob[1] + 0.00000001, state_change_prob[2])))


    for n_p_t in range(nodes_per_tv[0], nodes_per_tv[1] + 1, nodes_per_tv[2]):
        for s_p_t in range(states_per_tv[0], states_per_tv[1] + 1, states_per_tv[2]):
            for e_p_t in range(edges_per_tv[0], edges_per_tv[1] + 1, edges_per_tv[2]):
                if n_p_t<e_p_t:continue
                for per_iter in np.arange(percentage_inter[0], percentage_inter[1] + 0.000001, percentage_inter[2]):
                    for p_o_gap in np.arange(per_object_gap[0], per_object_gap[1] + 0.00000001, per_object_gap[2]):
                        for tscbn_var in np.arange(t_variance_tscbn[0], t_variance_tscbn[1] + 0.00000001, t_variance_tscbn[2]):
                            for dbn_tol in np.arange(dbn_tolerance[0], dbn_tolerance[1] + 0.00000001, dbn_tolerance[2]):
                                for sc_prob in np.arange(state_change_prob[0], state_change_prob[1] + 0.00000001, state_change_prob[2]):
                                    for o_nr in object_nr:
                                        print("\n----------------------------------\nobj_nr: %s\nnodes_p_t: %s\nstates_pt: %s\nedges_pt: %s\nper_iter: %s\np_o_gap: %s\ntscbn_var: %s\ndbn_tol: %s\nsc_prob: %s" % (o_nr, n_p_t, s_p_t, e_p_t, per_iter, p_o_gap, tscbn_var, dbn_tol, sc_prob))
                                        print("Remaining:  %s" % (str(expected_runs- run)))
                                        run += 1

                                        sg = StructureGenerator(test_type=TestStructureEnum.SPECIFICATION)
                                        ev = StructureEvaluator(append_csv);append_csv = True

                                        # Evaluation Parameters
                                        ev.add_setting("object_nr", o_nr)
                                        ev.add_setting("nodes_per_tv", n_p_t)
                                        ev.add_setting("states_per_tv", s_p_t)
                                        ev.add_setting("edges_per_tv", e_p_t)
                                        ev.add_setting("percentage_inter", per_iter)
                                        ev.add_setting("per_tv_gap", p_o_gap)
                                        ev.add_setting("tscbn_variance", tscbn_var)
                                        ev.add_setting("dbn_tolerance", dbn_tol)
                                        ev.add_setting("sc_probability", sc_prob)

                                        ev.set_output_path(out_path)
                                        ev.add_metric("num-edges")
                                        ev.add_metric("num-nodes")
                                        ev.add_metric("num-states")
                                        ev.add_metric("num-cpds")


                                        # ----------------------------------------------
                                        # Settings
                                        # ----------------------------------------------
                                        # Models
                                        sg.add_base_structure_models(eval_models)  # DBNStructureModel  TNBNStructureModel, TSCBNStructureModel
                                        if DBNStructureModel in sg.get_generator_models():[f for f in sg._generator_models if isinstance(f, DBNStructureModel)][0].EXPLICIT_DISABLING = True # set setting for DBN

                                        # Structure Generation Settings
                                        # NODE SETTINGS
                                        sg.set_node_range(min_objects=o_nr, max_objects=o_nr,  # number of temporal variables
                                                          min_temp_nodes=n_p_t, max_temp_nodes=n_p_t,  # number of nodes per temporal variable
                                                          min_states=s_p_t, max_states=s_p_t)  # number of states per node
                                        # EDGE SETTINGS
                                        sg.set_connection_ranges(min_edges_per_object=e_p_t, max_edges_per_object=e_p_t,
                                                                 # Anzahl der Temporal Variables die miteinander verbunden - haben jeweils x edges zwischen Objekten
                                                                 min_percent_inter=per_iter,
                                                                 max_percent_inter=per_iter)  # Range für Random - prozentualer Anteil an Querverbindungen pro TV im Bezug auf Knotenanzahl
                                        # TIME SETTINGS
                                        sg.set_temporal_range(min_per_object_gap=p_o_gap, max_per_object_gap = p_o_gap+0.5)
                                        sg.set_temporal_variance(tscbn_var)
                                        sg.set_dbn_tolerance(dbn_tol)

                                        # PROBABILITY SETTINGS
                                        sg.set_state_change_probability(min_probability=sc_prob,
                                                                        max_probability=sc_prob)  # probability of state change - at 1.0  parameter estimation should be exact

                                        # Generator Execution settings
                                        test_size = 1



                                        # Visualization parameters
                                        sg.set_model_visualization(plot=True, console_out=False)

                                        # ----------------------------------------------
                                        # Run tests
                                        # ----------------------------------------------
                                        for i in range(test_size):
                                            #print("\n\n------------------ Running Test %s ------------------" % (str(i + 1)))
                                            # Return test case
                                            try:
                                                models, specifications = sg.run_next_testcase()
                                            except:
                                                print("Invalid sample " + str(""))
                                                if not ev._append_csv: eval_result = ev.write_header(True)
                                                continue
                                            # evaluate result
                                            eval_result = ev.evaluate(models, specifications= specifications)

                                            # output
                                            ev.print_eval_results(eval_results=eval_result, specs=specifications, to_csv=True)
    L().log.info("-------------------- DONE -------------------------")
