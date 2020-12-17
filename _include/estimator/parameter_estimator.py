from general.base import Base

class ParameterEstimator(Base):
    '''
    Interface defining parameterEstimators
    '''

    def __init__(self, TSCBN = None):
        '''
        Takes a TSC Bayesian Network as input and estimates its
        parameters        
        '''
        super(ParameterEstimator, self).__init__(visual = True)
        self.tbn = TSCBN
        self.original_tbn = None
        self._parallel_processes = 1
        self._seen_tvs = dict()

    def estimateParameter(self, seqs):
        '''
        This approach simply counts the number of occurrences 
        '''
        raise NotImplementedError("estimateParameter() not implemented in class %s" % str(__class__))

    def set_parallel_processes(self, number):
        self._parallel_processes = number

    def _tv(self, node_name):
        '''
        Extracts the TV name from the node name
        :param node_name:
        :return:
        '''
        if node_name in self._seen_tvs:
            return self._seen_tvs[node_name]
        else:
            self._seen_tvs[node_name] = "_".join(node_name.split("_")[:-1])
            return  self._seen_tvs[node_name]

