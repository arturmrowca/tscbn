#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import random
import numpy
import copy
from enum import Enum

from _include.structure.base_structure_model import BaseStructureModel


class TestStructureEnum(Enum):
    SPECIFICATION = 1
    SIMPLE = 2

class StructureGenerator(object):
    '''
    Generates various structures as specified
    
    The Goal is to model objects that are in a state for a certain amount of time and
    that change their state and depend on the state of other objects
    With this generator a test-case for such a scenario is created
    '''

    def __init__(self, test_type = TestStructureEnum.SPECIFICATION):
        self._generator_models = [] # Models to generate from specification e.g. TNBN, DBN, TSCBN
        self._structure_type = test_type # Type of testcase to create

        self._sc_probability = False # disable state change probability if not in use
        self._temporal_variance = 0.01
        self._dbn_tolerance = 0.01

        self.show_plot_generated = False # show generated models
        self.show_text_generated = False


    def run_next_testcase(self):
        '''
        Returns the next test case
        '''
        # Draw structure
        if self._structure_type == TestStructureEnum.SPECIFICATION:
            return self._run_spec_structure()

        if self._structure_type == TestStructureEnum.SIMPLE:
            return self._run_simple_structure()

    def _run_spec_structure(self):
        '''
        Generate testcase as specified by a specification that is set in the
        generator
        :return:
        '''
        specification = self._draw_spec_structure()

        # 2. generate model for this setting
        models = {}
        for m in self._generator_models:
            models[m.model_key()] = m.generate_model(specification)
            models[m.model_key()].show_plot_generated = self.show_plot_generated
            models[m.model_key()].show_text_generated = self.show_text_generated

        # 3. return all generated models
        return models, specification

    def _run_simple_structure(self):
        '''
        Generate a simple structure by not passing any specifications but just
        drawing from the given models
        :return:
        '''

        # 2. generate model for this setting
        models = {}
        for m in self._generator_models:
            models[m.model_key()] = m.generate_model({})
            models[m.model_key()].show_plot_generated = self.show_plot_generated
            models[m.model_key()].show_text_generated = self.show_text_generated

        # 3. return all generated models
        return models, {}

    def _draw_uniform(self, in_range):
        if in_range[0] == in_range[1]: return in_range[0]
        return in_range[0] + random.random() * (in_range[-1] - in_range[0])

    def _draw_spec_structure(self):
        result = {}

        # Set temporal variance per node - ONLY FOR TSCBN!
        result["temporal_variance"] = self._temporal_variance

        # Set tolerance percentage in DBN - a slice is not allowed to be further away than this
        result["dbn_tolerance"] = self._dbn_tolerance

        # Draw number of objects to create
        object_number = round(self._draw_uniform(self._object_range))
        result["object_number"] = object_number

        # Draw number of nodes per object
        per_object_chain_number = [round(self._draw_uniform(self._temp_node_range))  for _ in range(object_number)]
        result["per_object_chain_number"] = per_object_chain_number

        # Draw number of states per node
        states_per_object = [round(self._draw_uniform_float(self._state_range)) for _ in range(object_number)]
        result["states_per_object"] = states_per_object

        # Draw probability of a state change per node
        result["state_change"] = [[self._draw_uniform_float(self._sc_probability)
                                   for _ in range(per_object_chain_number[i])] for i in range(object_number)]

        # Draw temporal gap between intra-nodes
        temp_gap_between_objects = [[self._draw_uniform_float(self._intra_object_temp_range)
                                     for _ in range(per_object_chain_number[i]-1)] for i in range(object_number)]
        result["temp_gap_between_objects"] = temp_gap_between_objects
        
        # Set object and state names
        result["object_names"], result["object_states"] = self._set_object_properties(object_number, states_per_object)
        
        # Draw number of objects that connect to this object
        result["inter_edges_to_this_object"] = self._draw_objects_to_connect(object_number, result["object_names"])

        # Draw number of nodes per object
        result["nodes_per_object"] = [round(kk * self._draw_uniform_float(self._percentage_inter_edges)) for kk in per_object_chain_number]

        return result

    def set_model_visualization(self, plot, console_out):
        self.show_plot_generated = plot  # show generated models
        self.show_text_generated = console_out # show command line output for models

    def _set_object_properties(self, object_number, states_per_object):
        object_names, object_states = ["O%s" % str(i) for i in range(object_number)], {}
        for i in range(len(object_names)):
            object_states[object_names[i]] = ["o%s_%s" % (str(i), str(j)) for j in range(states_per_object[i])]
        return object_names, object_states

    def _draw_objects_to_connect(self, object_number, object_names):
        inter_edges_to_this_object_pre = [self._draw_uniform(self._edges_inter_object_range) for _ in
                                          range(object_number)]
        if numpy.any(numpy.array(inter_edges_to_this_object_pre) >= object_number): raise AssertionError(
            "Number of connecting edges needs to be smaller then object number")

        # per edge draw from this range and remove an edge in this range
        t = -1
        inter_edges_to_this_object = []
        for obj_edge_num in inter_edges_to_this_object_pre:
            t += 1
            p_list = copy.deepcopy(object_names)
            p_list.remove(object_names[t])  # edge to myself is meaningless for inter edges - remove it
            object_edges = self._draw_uniform_samples_from_list(p_list, int(obj_edge_num))
            inter_edges_to_this_object.append(object_edges)

        return inter_edges_to_this_object


    def set_temporal_variance(self, variance):
        '''
        Sets the variance of the time around the mean between each intravariable distance
        :param variance:
        :return: -
        '''
        self._temporal_variance = variance

    def set_dbn_tolerance(self, dbn_tolerance):
        '''
        Set tolerance percentage in DBN - a slice is not allowed to be further
        away than this
        '''
        self._dbn_tolerance = dbn_tolerance

    def _draw_uniform_samples_from_list(self, lst, sample_nr):
        ''' Draw sample_nr random samples from a list'''
        res = []
        for _ in range(sample_nr):
            idx = round(self._draw_uniform([0, len(lst)-1]))-1
            if idx >= len(lst):
                idx = len(lst)-1
            res.append(lst[idx])
            
            lst.remove(lst[idx])
        return res

    def _draw_uniform_float(self, min_max):
        min_val, max_val = min_max
        return min_val + (max_val - min_val)* random.random()

    def set_node_range(self, min_objects, max_objects, min_temp_nodes, max_temp_nodes, min_states, max_states):
        '''
        This method sets the parameters for the node creation. Each object has several states that change over time.
        Per test case different numbers of objects and states within specified ranges are created. The temporal 
        length of the chain to be created is defined by min_temp_nodes and max_temp_nodes, which is the range 
        within which each object has nodes. E.g. object 1 could have 3 nodes 
        '''
        self._object_range = [min_objects, max_objects]
        self._temp_node_range = [min_temp_nodes, max_temp_nodes]
        self._state_range = [min_states, max_states]
        
    def set_state_change_probability(self, min_probability, max_probability):
        '''
        Defines probability within which the probability that a state change happens lies
        If 1.0 state change will always occur at 0.0 no state will change
        '''    
        self._sc_probability = [min_probability, max_probability]

    def set_temporal_range(self, min_per_object_gap, max_per_object_gap):
        '''
        Defines the temporal range for an object - with which one value occurs 
        after the other - also specify the average distance within one object occurs after 
        another
        '''
        self._intra_object_temp_range = [min_per_object_gap, max_per_object_gap]
        # range zwischen Objekten ist durch restliche Parameter definierts
        
    def set_connection_ranges(self, min_edges_per_object, max_edges_per_object,
                              min_percent_inter=0.0, max_percent_inter=1.0):
        '''
        Defines the number of edges between 
        similarity_variation: Defines the deviation per inter node connection that is possible
                              0 means no variation. I.e. the number of specified edges is always the same
                              e.g. 1 means - one edge could be missing/added between nodes
        edges_per_node        Defines the number of inter node/object connections that are possible
        '''
        self._edges_inter_object_range = [min_edges_per_object, max_edges_per_object]
        self._percentage_inter_edges = [min_percent_inter, max_percent_inter]
        
    def add_base_structure_model(self, structure_model):
        ''' Adds a model that can be created '''
        assert (issubclass(structure_model, BaseStructureModel)), ("Object of class BaseStructureModel required - %s given" % (str(structure_model.__class__.__name__)))
        self._generator_models += [structure_model()]
        
    def add_base_structure_models(self, structure_models):
        for m in structure_models:
            self.add_base_structure_model(m)

    def get_generator_models(self):
        return  self._generator_models
        
