from _include.visual.visualizer import Visualizer
class Base(object):
    '''
    object that is similar to object, with additional functionality that is required by multiple
    subclasses
    '''
    def __init__(self, visual=False):
        if visual:  self._visual = Visualizer()

    def not_implemented(self, method_name):
        raise NotImplementedError("Method %s was not implemented in class %s" % (str(method_name), str(self.__class__.__name__)))
