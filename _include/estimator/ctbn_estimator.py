from _include.structure.ctbn_structure_model import CTBNStructureModel
from general.base import Base
import os
import numpy as np
import pandas as pd

class CTBNEstimator(Base):
    '''
    This class learns the structure and the parameters of a CTBN
    given a sequence set - this is done by exporting this task
    to the R library
    The engine used is CTBN - RLE found at http://rlair.cs.ucr.edu/ctbnrle/v1.0.2/
    '''
    CTBN_ENGINE_IS_INSTALLED = False

    def __init__(self, TSCBN = None):
        '''
        Takes a TSC Bayesian Network as input and estimates its
        parameters        
        '''
        self.tbn = TSCBN
        self.original_tbn = None
        self._parallel_processes = 1

        self.R_installed = False
        self.CTBN_RLE_installed = False
        self.estimation_runnable = True
        self._check_requirements()


    def _check_requirements(self):
        ''' Assert RScript installed'''
        from subprocess import Popen, PIPE, STDOUT
        try:
            p = Popen(["RScript", '--version'], stdout=PIPE, stderr=STDOUT)
            out, err = p.communicate()
            self.R_installed = True
        except:
            pass

        # Check if CTBN present
        import os
        try:
            path = os.path.join(os.environ['R_HOME'], r"library/ctbn")
            if os.path.exists(path):
                self.CTBN_RLE_installed = True
        except:
            pass

        # Warn:
        msg = ""
        if (not self.R_installed) or (not self.CTBN_RLE_installed):
            self.estimation_runnable = False
            msg += "\n\n-------------------------------------\n"
            msg += "   WARNING\n"
            msg += "   CTBN Evaluation will be ommitted\n"
            msg += "-------------------------------------\n"

            if not self.R_installed:
                msg += " - No R installation was found. Please ensure that R_HOME is set as environment variable\n"
            if not self.CTBN_RLE_installed:
                msg += " - CTBN RLE is not installed in R please install it at CTBN-RLE from http://rlair.cs.ucr.edu/ctbnrle/ \n\n"
            print(msg)


    def sequence_to_dataframe(self, sequence, rename_dict):
        dfs = []
        for tv in sequence:
            dfs += [pd.DataFrame(sequence[tv], columns=[tv, "Time", "EndTime"])[["Time", tv]]]
        df = pd.concat(dfs)#, sort=False)
        df = df.sort_values("Time")
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')

        #  -> map each column to number!
        df = df.replace(rename_dict).round(3).drop_duplicates()
        return df

    def estimateStructureAndParameter(self, sequences, tscbn):
        '''
        This approach simply counts the number of occurrences 
        '''
        if not self.estimation_runnable:
            print("Skipping CTBNs - please check warning messages")
            return None

        # Create temporary folder for data
        path = r"store/run_%s" % str(np.random.rand()).replace(".","")
        if not os.path.exists(path):os.mkdir(path)
        print("Generating Samples and Structure...")

        # Extract information from Tscbn that we sampled from (e.g. Variables)
        node_names = list(set(["_".join(v.split("_")[:-1]) for v in tscbn.V if not str.startswith(v, "dL_")]))
        rename_dict = dict()
        vars = []
        for node_name in node_names:
            add_dict = dict()
            add_dict[node_name] = dict([(tscbn.Vdata[node_name + "_0"]["vals"][x], x) for x in range(len(tscbn.Vdata[node_name + "_0"]["vals"]))])
            rename_dict = {**rename_dict, **add_dict}
            vars += [(node_name, len(tscbn.Vdata[node_name + "_0"]["vals"]))]
        var_df = pd.DataFrame(vars, columns=["Name", "Value"])
        var_df.to_csv(os.path.join(path, "variables.var"), sep = ";", index=False)

        # assume given structure -> from ground truth to be fair
        intensity_params, transition_params, node_data = CTBNStructureModel().model_from_tscbn_ground_truth(tscbn, var_df, path, rename_dict)

        # Store sequences in this folder
        idx = -1
        for sequence in sequences:
            idx += 1
            df = self.sequence_to_dataframe(sequence, rename_dict)
            if len(df)> len(vars):
                df.to_csv(os.path.join(path, str(idx) + ".csv"), sep = ";", index=False)

        # Use R Script to learn parameters
        print("Running Parameter Estimation...")
        from subprocess import Popen, PIPE, STDOUT
        p = Popen(["RScript", 'script.r', path], stdout=PIPE, stderr=STDOUT)
        out, err = p.communicate()


        # read generated Structure and store it to transition and intensity matrices
        print("Reading Structure...")
        try:
            with open(os.path.join(path, "done.rctbn"), 'r') as file:
                data = file.read()
        except:
            print("RScript crashed with message \n%s" % str(out))
        elements = data.split("</") # tscbn

        # set intensity_params, transition_params node_data["transition"] node_data["intensity"]
        idx = 0
        for element in elements:
            idx += 1
            #print("Processingg %s" % str(idx))
            if "<BNCPTS>" in element:
                # read intensities
                shorter = element[element.index("<BNCPTS")+9:]
                l = shorter.split("\n")
                n = 3
                parts = [l[i:i + n] for i in range(0, len(l), n)]

                for part in parts:
                    if not part[0]: continue
                    [node, condition] = part[0].split("$")
                    if "=" in condition:
                        # then conditioned
                        node_data[node]["transition"][str(condition.split(","))] = np.array([float(a) for a in part[2].split(",")[1:]])
                    else:
                        node_data[node]["transition"] = np.array([float(a) for a in part[2].split(",")[1:]])

            if "<DYNCIMS>" in element:
                # read transition matrices
                # read intensities
                shorter = element[element.index("<DYNCIMS") + 10:]
                l = shorter.split("\n")
                part = []
                for line in l:

                    # process parts
                    if "$" in line and part:
                        if not part[0]: continue
                        [node, condition] = part[0].split("$")
                        if "=" in condition:
                            # then conditioned
                            node_data[node]["intensity"][str(condition.split(","))] = np.array([[float(k) for k in l.split(",")[1:]] for l in part[2:]])
                        else:
                            node_data[node]["intensity"] = np.array([[float(k) for k in l.split(",")[1:]] for l in part[2:]])
                        part = []

                    part += [line]

        node_data["rename_dict"] = rename_dict

        # remove temporary files
        import shutil
        shutil.rmtree(path)

        return node_data #

    def set_parallel_processes(self, number):
        self._parallel_processes = number

