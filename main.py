'''
Copyright (c) 2019
Artur Mrowca
Florian Gyrock
'''
from experiments.case_study import run_case_hb, run_case_ind, run_case_hb_bayspec, run_case_ind_bayspec
from experiments.model_evaluation import run_vary_structure
from experiments.parameter_estimation import run_vary_sc_probability, run_vary_sample_number
from general.setup import *
import sys

'''
Model + Paramter Estimation Experiments:    
'''
if __name__ == '__main__':
    ''' Experiments that are included in the Paper about Temporal State Change Bayesian Networks are 
        provided here. Please adjust the parameters in the settings section to run the experiments 
        The following Modes are provided:
            - VARY_STRUCTURE: compares the TSCBN to DBNs in terms of structure which is number of parameters, nodes, edges, etc.
            - VARY_SAMPLE_NUMBER: evaluation of the TSCBN when increasing the number of training samples
            - VARY_STATE_CHANGE_PROBABILITY: evaluation of the TSCBN when increasing the probability of a state change
            - CASE_HB: Case study of the HB Model presented in the paper
            - CASE_IND: Case study of the IND Model presented in the paper
        Further, this code requires Python Anaconda 3.6. 
        
        If you wish to run the evaluation of CTBNs please install 
            -   R and add RScript's Folder to your Environment Variables
            -   install CTBN-RLE from http://rlair.cs.ucr.edu/ctbnrle/
        
        Further, please note that the folders _include/m_libpgm AND _include/m_utils 
            are an extension of https://pythonhosted.org/libpgm/
            and thus, any notes or text given in files within this folder are not set by us!
            
        The experiments for Structure Discovery can be found in main_structure.py
    '''

    # This is required for parallel Execution of the parameter estimation
    from multiprocessing import freeze_support
    freeze_support()

    # --------------------------------------
    #   SETTINGS
    # --------------------------------------
    experiment_selection = Experiment.VARY_SAMPLE_NUMBER # Choose experiment VARY_STRUCTURE, VARY_SAMPLE_NUMBER, VARY_STATE_CHANGE_PROBABILITY, CASE_HB, CASE_IND, CASE_BAYSPEC_HB, CASE_BAYSPEC_IND
    print_sequences = False # If true will print all interval sequences
    plot_model = False # If true shows the network structure generated
    print_true_distribution = False # prints the true parameters of the ground truth
    cs_show_histograms = False # shows the histograms of estimated parameters in the case study
    number_parallel = 35 # Number of parallel processes used for processing
    target_path = r"D:\diss_eval_final\parameter_estimation" # Path to store results of PE to (logfile and csv)
    if experiment_selection == Experiment.VARY_STRUCTURE:target_path = r"D:\diss_eval_final\model" # Path to store results of model to (logfile and csv)
    number_TVs = 3 # Number of TVs here 3, 5 and 10 used in experiments
    estimators = [Estimators.EM] # EM  MLECOUNT VI_PARALLEL (=correct and fast version of VI)

    # Case Study BaySpec Parameters (CASE_BAYSPEC_HB and CASE_BAYSPEC_IND)
    # set one of the below to true at a time
    sd_hyperparameter_estimation_t_th = False # Step 1: Plots distribution of temporal gaps, used to define parameter t_th
    sd_hyperparameter_estimation_k = False # Step 2: Plots execution of structure discovery when varying k, used to find parameter k
    sd_hyperparameter_estimation_chi = False # Step 3: Plot varying chi
    learn_model = False # Step 4: based on the found k and t_th apply Structure Discovery followed by MLE Parameter Estimation - the result is stored
    bayspec = False # Step 5: Bayspec is used to extract specifications from the learned model
    map = False # Step 5: Use MAP sampling to find most probable system states

    # --------------------------------------
    #   EVALUATION
    # --------------------------------------
    # Initialize logger
    if not os.path.exists(target_path):
        print("Please change or create path to output folder %s" % str(target_path))
        sys.exit(0)
    logging_setup(target_path, number_parallel)

    # Run model evaluation grid
    if experiment_selection is Experiment.VARY_STRUCTURE:
        run_vary_structure(target_path)

    # Run parameter evaluation
    elif experiment_selection is Experiment.VARY_STATE_CHANGE_PROBABILITY:
        run_vary_sc_probability(number_TVs, number_parallel, target_path, print_sequences, plot_model, print_true_distribution, estimators)

    elif experiment_selection is Experiment.VARY_SAMPLE_NUMBER:
        run_vary_sample_number(number_TVs, number_parallel, target_path, print_sequences, plot_model, print_true_distribution, estimators)

    # Run case-study evaluation USE EM here!
    elif experiment_selection is Experiment.CASE_HB:
        run_case_hb(number_parallel, plot_model, estimators, cs_show_histograms) # results printed to console

    elif experiment_selection is Experiment.CASE_IND:
        run_case_ind(number_parallel, plot_model, estimators, cs_show_histograms) # results printed to console

    # Run case-study evaluation
    elif experiment_selection is Experiment.CASE_BAYSPEC_HB:
        run_case_hb_bayspec(sd_hyperparameter_estimation_t_th, sd_hyperparameter_estimation_k, sd_hyperparameter_estimation_chi, learn_model, bayspec, map) # results printed to console

    elif experiment_selection is Experiment.CASE_BAYSPEC_IND:
        run_case_ind_bayspec(sd_hyperparameter_estimation_t_th, sd_hyperparameter_estimation_k, sd_hyperparameter_estimation_chi, learn_model, bayspec, map) # results printed to console
