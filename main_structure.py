'''
Copyright (c) 2019 The Authors of the Paper "Temporal State Change Bayesian Networks: Modeling Multivariate State Sequences with Evolving Dependencies"
'''
import itertools
from pymongo import MongoClient
from sacred.observers import MongoObserver
import sys
from _include.toolkit import parallelize_stuff
from general.setup import *
from experiments.structure_evaluation import ex
import numpy as np

def run(config):
    print("Run: %s" % str(config))
    return ex.run(config_updates=config)

def config_by_experiment(config, experiment):

    if experiment == ExperimentSD.VARY_SAMPLE_NUMBER:
        config["sample_size"] = [50, 100, 200, 250, 350, 450] + list(range(500, 30001, 500))

    if experiment == ExperimentSD.VARY_NODES_PER_TV:
        config["number_of_temp_nodes"] = [3, 5, 10, 15, 20] # LENGTH OF EACH TV

    if experiment == ExperimentSD.VARY_NUMBER_TVS:
        config["number_of_signals"] = [3, 5, 10, 15] # Number of TVs

    if experiment == ExperimentSD.IND_VARY_T_THRESHOLD:
        config["sb_max_time_difference"] = list(np.arange(0.0, 2.51, 0.1))# true value is around 0.5

    if experiment == ExperimentSD.VARY_SC_PROBABILITY:
        config["sc_probability"] = list(np.arange(0.4, 1.01, 0.05)) # Number of TVs

    return config

'''
This was implemented using sacredboard. To run the experiments the following configurations need to be set.


1. Start Mongo db and sacredboard:
    START Mongo db server mongo.exe -> C:\Program Files\MongoDB\Server\4.0\bin
    START C:\Anaconda3\Scripts\sacredboard.exe -m structure_discovery_db
    
    
2. Run experiments using command line arguments. For the paper the following code was run 
    arg[0]: type of experiment i.e. vary sc probability, etc.
    arg[1]: name of approach to run
    arg[2]: 1: drops the results in Mongo DB for this experiment -- 0: appends results to experimental results

    python main_structure.py 0 PREFIX 1 
    python main_structure.py 0 SB_TREE 0
    python main_structure.py 0 HILL_CLIMB 0
    python main_structure.py 0 PC_TREE_VAR 0
    python main_structure.py 0 PC_DISCOVER 0
    python main_structure.py 0 PC_TREE 0
    python main_structure.py 1 PREFIX 1
    
    python main_structure.py 1 SB_TREE 0
    python main_structure.py 1 HILL_CLIMB 0
    python main_structure.py 1 PC_TREE_VAR 0
    python main_structure.py 1 PC_DISCOVER 0
    python main_structure.py 1 PC_TREE 0
    python main_structure.py 2 PREFIX 1 
    
    python main_structure.py 2 SB_TREE 0
    python main_structure.py 2 HILL_CLIMB 0
    python main_structure.py 2 PC_TREE_VAR 0
    python main_structure.py 2 PC_DISCOVER 0
    python main_structure.py 2 PC_TREE 0
    python main_structure.py 3 SB_TREE 1
    
    python main_structure.py 3 PC_TREE_VAR 0
    python main_structure.py 3 PC_TREE 0
    python main_structure.py 4 PREFIX 1
        
    python main_structure.py 4 SB_TREE 0
    python main_structure.py 4 HILL_CLIMB 0
    python main_structure.py 4 PC_TREE_VAR 0
    python main_structure.py 4 PC_DISCOVER 0
    python main_structure.py 4 PC_TREE 0

    
3. Write results to csv file
    IN MONGODB run:
    db.metrics.aggregate( [{$lookup:    {        from: 'runs',        localField: 'run_id',        foreignField: '_id',        as: 'output'    } }, {$out:"results"}] );
    
    IN C:\Program Files\MongoDB\Server\4.0\bin run :
    Export: mongoexport.exe --db structure_discovery_db --collection results --type=csv --out D:\diss_eval_final\structure_discovery\samples.csv --fields name,run_id,steps,timestamps,values,output

    
3. (optional) Query MongoDB for results with the following useful queries
    show dbs
    use structure_discovery_db
    
    db.dropDatabase()
    
    show collections;
    db.getCollectionNames();
    
    SELECT * FROM metrics == db.metrics.find( {} ) 
'''

if __name__ == '__main__':

    from multiprocessing import freeze_support
    freeze_support()

    # --------------------------------------
    #   SETTINGS
    # --------------------------------------
    log_path = r"C:\eval_results" # Path to store results to (logfile and csv)
    config, input_lst = dict(), []
    logging_setup(log_path, 0)

    # --------------------------------------
    #   Configure HERE
    # --------------------------------------
    experiment = ExperimentSD.to_enum(sys.argv[1]) #ExperimentSD.VARY_SAMPLE_NUMBER # VARY_SAMPLE_NUMBER, VARY_SC_PROBABILITY, VARY_NODES_PER_TV, VARY_NUMBER_TVS, IND_VARY_T_THRESHOLD
    number_parallel = 1 # Number of parallel processes used for processing
    config["approach"] = [sys.argv[2]]#[Discoverers.SB_TREE] #[Discoverers.A_STAR, Discoverers.PC_TREE_VAR, Discoverers.HILL_CLIMB, Discoverers.SB_TREE, Discoverers.PREFIX, Discoverers.PC_DISCOVER, Discoverers.PC_TREE,

    # --------------------------------------
    #   Run Experiments
    # --------------------------------------
    # DROP OLD DATABASE
    print("Run %s, %s" % (str(experiment), str(config["approach"])))
    db_name_in = 'structure_discovery_db_%s' % str(experiment).replace("ExperimentSD.","")
    if sys.argv[3] == 1:
        client = MongoClient()
        client.drop_database(db_name_in)
        client.close()
        print("Dropped Database")

    # RUN NEW EXPERIMENT
    ex.observers.append(MongoObserver.create(db_name=db_name_in))
    config = config_by_experiment(config, experiment)
    for x in itertools.product(*tuple([[(k,vi) for vi in v] for k, v in config.items()])):
        if number_parallel == 1:run(dict(x))
        else:input_lst += [[dict(x)]]
    if number_parallel > 1:parallelize_stuff(input_lst, run, simultaneous_processes=number_parallel)


