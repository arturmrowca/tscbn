from general.base import Base
from general.log import Log as L
import csv

class BaseEvaluator(Base):
    '''
    Based on given results this class is supposed to return an evaluation
    '''

    def __init__(self, append_csv=True):
        self._metrics = [] # Metrics to use for evaluation
        self._output_path = None
        self._append_csv = append_csv

        self._settings_variables = {} # settings at this evaluation

    def evaluate(self, result, reference = None):
        self.not_implemented("evaluate")

    def add_setting(self, setting_name, setting_value):
        self._settings_variables[setting_name] = setting_value

    def add_metric(self, metric):
        '''
        Adds a metric that is used for evaluation
        '''
        self._metrics.append(metric)

    def write_header(self, to_csv = False):

        # 1. CSV Writer
        if to_csv and not self._append_csv:

            mode = 'w'
            if self._append_csv: mode = 'a'

            with open(self._output_path, mode) as csvfile:
                fieldnames = ["model_name"] + list(self._settings_variables.keys()) + list(self._metrics)
                writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
                if mode=="w": writer.writeheader()
                result = self._settings_variables

    def print_eval_results(self, eval_results=None, specs={}, to_csv=False):
        '''
        Given the result of the evaluate method this method prints the result
        '''
        if eval_results is None: eval_results = self._last_eval_results
        L().log.info("\n-----------------------------------\n  Results of evaluation \n-----------------------------------")

        # 1. CSV Writer
        if to_csv:

            mode = 'w'
            if self._append_csv: mode = 'a'

            with open(self._output_path, mode) as csvfile:
                fieldnames = ["model_name"] + list(self._settings_variables.keys()) + list(self._metrics)
                writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
                if mode=="w": writer.writeheader()
                result = self._settings_variables

                for model_name in eval_results:
                    result["model_name"] = model_name
                    for metric in eval_results[model_name]:
                        result[metric] = eval_results[model_name][metric]
                    writer.writerow(result)


        for model_name in eval_results:
            L().log.info("\n-----> Model name: \t\t%s" % str(model_name))
            for metric in eval_results[model_name]:
                metric_result = eval_results[model_name][metric]
                L().log.info("%s: \t\t%s" % (self._readable_metric(metric), str(metric_result)))

    def set_output_path(self, path):
        '''
        Sets the output csv File to write to
        '''
        self._output_path = path

    def _readable_metric(self, metric):
        '''
        Translates a metric key to a readable version
        '''
        if metric == "relative-entropy": return "Relative Entropy"
        return metric

    def _compute_metric(self, model, reference, metric):
        '''
        Given the resulting model this method computes the metrics specified
        '''
        if metric == "relative-entropy":
            return "Nothing Implemented"
