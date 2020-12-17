from _include.m_libpgm.discretebayesiannetwork import DiscreteBayesianNetwork


class DBNDiscrete(DiscreteBayesianNetwork):
    '''
    This class represents a Bayesian network with discrete CPD tables. It contains the attributes *V*, *E*, and *Vdata*, as well as the method *randomsample*.

    '''


    def __init__(self, orderedskeleton=None, nodedata=None, path=None):
        super(DBNDiscrete, self).__init__(orderedskeleton, nodedata, path)
        self.start_time = 0.0
        self.end_time = 0.0
        self.resolution = 0.0
        self.eval_cpd_entries = -1

        self.skeleton = orderedskeleton
