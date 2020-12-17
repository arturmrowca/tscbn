import logging
import os, sys
from enum import Enum
from time import localtime, strftime

from _include.discoverer.prefix_structure_discoverer import PrefixStructureDiscoverer
from _include.estimator.em_algorithm_tscbn_estimator import EMAlgorithmParameterEstimator
from _include.estimator.mle_counting_local_tscbn_estimator import MLECountingLocalParameterEstimator
from _include.estimator.mle_local_tscbn_estimator import MLELocalParameterEstimator
from _include.estimator.trivial_counter_bn_estimator import TrivialCounterForBNEstimator
from _include.estimator.vi_opt_tscbn_estimator import VariationalInferenceOptimizedParameterEstimator
from _include.estimator.vi_parallel_tscbn_estimator import VariationalInferenceParallelParameterEstimator
from _include.estimator.vi_tscbn_estimator import VariationalInferenceParameterEstimator
from general.log import Log as L

def logging_setup(log_path, number_parallel):
    if number_parallel > 10: print("You chose to run more than 10 processes in parallel. Be aware that your machine requires according computational power for this. Else choose less parallel processes.\n")

    print("Starting Experiments...")
    #sys.stderr = open(os.devnull, 'w') # disable broken pipe error
    time_str = strftime("%Y_%m_%d-%H_%M_%S", localtime())
    log_path = os.path.join(log_path, "logging_bay_" + time_str + ".log")

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT,datefmt="%H:%M:%S  ")
    open(log_path, 'w').close()
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(FORMAT, "%H:%M:%S  "))

    L().log = logging.getLogger("tscbn_eval")
    L().log.addHandler(file_handler)
    L().log.setLevel(logging.INFO)
    L().log.parent.handlers = []
    L().log.info("Logger initialized...")

class Experiment(Enum):
    VARY_STRUCTURE = 0
    VARY_STATE_CHANGE_PROBABILITY = 1
    VARY_SAMPLE_NUMBER = 2
    CASE_HB = 3
    CASE_IND = 4
    CASE_BAYSPEC_HB = 5
    CASE_BAYSPEC_IND = 6

class ExperimentSD(Enum):
    VARY_SAMPLE_NUMBER = 0
    VARY_NODES_PER_TV = 1
    VARY_NUMBER_TVS = 2
    IND_VARY_T_THRESHOLD = 3
    VARY_SC_PROBABILITY = 4

    def to_enum(str_rep):
        if int(str_rep) == 0:
            return ExperimentSD.VARY_SAMPLE_NUMBER
        if int(str_rep) == 1:
            return ExperimentSD.VARY_NODES_PER_TV
        if int(str_rep) == 2:
            return ExperimentSD.VARY_NUMBER_TVS
        if int(str_rep) == 3:
            return ExperimentSD.IND_VARY_T_THRESHOLD
        if int(str_rep) == 4:
            return ExperimentSD.VARY_SC_PROBABILITY
class Estimators:
    COUNT = "COUNT"
    EM = "EM"
    MLE = "MLE"
    MLECOUNT = "MLECOUNT"
    VI_OPTIMIZED = "VI_OPTIMIZED"
    VI_PARALLEL = "VI_PARALLEL"
    VI = "VI"

class Discoverers:
    PREFIX = "PREFIX"
    SB_TREE = "SB_TREE" # Our approach
    PC_TREE = "PC_TREE" # Our approach
    PC_TREE_VAR = "PC_TREE_VAR" # Our approach chi square
    PC_DISCOVER = "PC_DISCOVER"
    A_STAR = "A_STAR"
    HILL_CLIMB = "HILL_CLIMB"


def create_estimator(estimator_id):
    '''
    Creates the parameter estimator depending on the selection
    '''
    if estimator_id == Estimators.MLE: return MLELocalParameterEstimator()
    if estimator_id == Estimators.MLECOUNT: return MLECountingLocalParameterEstimator()
    if estimator_id == Estimators.VI_PARALLEL: return  VariationalInferenceParallelParameterEstimator()
    if estimator_id == Estimators.VI_OPTIMIZED: return VariationalInferenceOptimizedParameterEstimator()
    if estimator_id == Estimators.EM: return  EMAlgorithmParameterEstimator()
    if estimator_id == Estimators.COUNT: return TrivialCounterForBNEstimator()
    if estimator_id == Estimators.VI: return VariationalInferenceParameterEstimator()

class Printos(object):
    def print_settings(sg, pe, ev, test_size, train_test_split, sample_sequences_from_tscbn, evidence, testmode_models):
        L().log.info("---------------------------------------------------------------------------------")
        L().log.info("                            SETTINGS                                             ")
        L().log.info("---------------------------------------------------------------------------------\n")
        ("\n\t\t\t\t\t\t ---> Execution Settings<---")
        L().log.info("Test size: \t\t\t\t\t\t%s" % str(test_size))
        #L().log.info("Traintest split percentage: \t%s per cent" % str(train_test_split * 100))
        L().log.info("Number of reference Samples: \t%s" % str(sample_sequences_from_tscbn))
        L().log.info("Evidence: \t\t\t\t\t\t%s" % str(evidence))
        L().log.info("Testmode Models : \t\t\t\t%s" % str(testmode_models))

        L().log.info("\n\t\t\t\t\t\t ---> Parameter Estimation <---")
        L().log.info("E-Step Sampling Frequency: \t\t%s" % str(pe.sampling_frequency))  # sampling frequency for the MC MC Simulation
        L().log.info("EM Iterations: \t\t\t\t\t%s" % str(pe.iteration_frequency))  # EM Iterations
        L().log.info("Parallel Processes: \t\t\t%s" % str(pe._parallel_processes))

        L().log.info("\n\t\t\t\t\t\t ---> TSCBN Infos <---")
        '''L().log.info("Object Range: \t\t\t\t\t%s" % str(sg._object_range))
        L().log.info("Models: \t\t\t\t\t\t%s" % str([m.__class__.__name__ for m in sg._generator_models]))
        L().log.info("Number TVs: \t\t\t\t\t%s" % str(sg._temp_node_range))
        L().log.info("Number States: \t\t\t\t\t%s" % str(sg._state_range))
        L().log.info("Number Inter-TV: \t\t\t\t%s" % str(sg._edges_inter_object_range))
        L().log.info("Percentage Inter-TV: \t\t\t%s" % str(sg._percentage_inter_edges))
        L().log.info("Intra Object Range: \t\t\t%s" % str(sg._intra_object_temp_range))
        L().log.info("TSCBN Temporal Variance: \t\t%s" % str(sg._temporal_variance))
        L().log.info("State Change Probability: \t\t%s" % str(sg._sc_probability))

        L().log.info("\n\t\t\t\t\t\t ---> Evaluation Settings <---")
        L().log.info("DBN Tolerance: \t\t\t\t\t%s" % str(sg._dbn_tolerance))'''
        L().log.info("RMSE TSCBN Variance: \t\t\t%s" % str(ev.rmse_tscb_variance))  # variance assumed per node - does not require parameter estimation
        L().log.info("RMSE TSCBN MEAN DRIFT: \t\t\t%s" % str(ev.rmse_mean_range))
        L().log.info("Evaluation Metrics")
        for m in ev._metrics:
            L().log.info("\t\t%s" % str(m))
        L().log.info("---------------------------------------------------------------------------------")
        L().log.info("                            END SETTINGS                                             ")
        L().log.info("---------------------------------------------------------------------------------\n\n\n")
        L().log.info("---------------------------------------------------------------------------------")
        L().log.info("                                RUN ")
        L().log.info("---------------------------------------------------------------------------------\n\n")
