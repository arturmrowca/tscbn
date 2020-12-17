#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead  
from collections import OrderedDict
import PyQt5
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from snakes.nets import *
from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.plotting import figure, output_file, show
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, StaticLayoutProvider, Oval
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import *
import numpy
import matplotlib as mpl
from bokeh.models.sources import ColumnDataSource
import math
from bokeh.core.property_mixins import LineProps
from bokeh.models.annotations import LabelSet, Label
from bokeh.models.arrow_heads import TeeHead
from bokeh.models.tools import WheelZoomTool, PanTool
mpl.use('WXAgg')
mpl.interactive(False)
import pylab as pl
from pylab import get_current_fig_manager as gcfm
import numpy as np
import random
import wx


if os.name == "nt":  # if windows
    pyqt_plugins = os.path.join(os.path.dirname(PyQt5.__file__),
                                "..", "..", "..", "Library", "plugins")
    QApplication.addLibraryPath(pyqt_plugins)


class Visualizer(object):
    '''
    used to visualize data
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
        
    def draw_all_pattern_graphs(self, pattern_df, target_col, if_not_empty = False, support_thrshld = None):
        '''
        Cannot draw pattern of length 0 or 1!
        '''
        for i in range(len(pattern_df)):
            print("\n\n--- Sequence " + str(i))
            cur_dic = eval(pattern_df.iloc[i][target_col])
            for j in cur_dic.keys():
                if if_not_empty and not cur_dic[j]: #skip empty pattern
                    continue 
                
                self.draw_pattern_graph(cur_dic[j], support_thrshld)
        
    def show_graph_with_labels(self, adjacency_matrix, mylabels, title =None):
        plt.close()
        fig = plt.figure(0)
        rows, cols = np.where(adjacency_matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.DiGraph()
        gr.add_edges_from(edges, weight= 0.5)
        pos=nx.spring_layout(gr)
        
        # remove unneeded labels
        rmv = []
        for k in mylabels.keys():
            if k not in pos:
                rmv.append(k)                
        for k in rmv:
            mylabels.pop(k, None)
       
        
        nx.draw(gr,pos, node_size=500, labels=mylabels, with_labels=True)
        if title != None:            
            fig.canvas.set_window_title(title)
        
        plt.show()

        ''' 
        edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
    
        # For later use: colors
        #red_edges = [('C','D'),('D','A')]
        #edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
        pos=nx.spring_layout(G) # circular_layout    random_layout       shell_layout    spring_layout    spectral_layout
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        nx.draw(G,pos, node_color = values, node_size=1500)
        nx.draw_networkx_labels(G, pos, node_labels, font_size=10)
        plt.show()
        '''
        
    def draw_pattern_graph(self, pattern_list, support_thrshld = None):
        
        for lst in pattern_list: # Todo: each in one subplot
            try:
                G = nx.DiGraph()
                one_pattern = ["\n".join(el) for el in lst[0]] # list of strings = nodelabels
                support = float("{0:.2f}".format(lst[1]))
                
                if support_thrshld != None:
                    if support < support_thrshld:
                        continue
                
                if len(one_pattern) < 2:
                    print("Pattern " + str(one_pattern) + " too short! Skipped it!")
                    continue 
                
                # set edges
                edges = []
                node_labels ={}
                val_map = {}
                prev = one_pattern[0]
                w_elements = list(np.linspace(0,1, len(one_pattern)))
                i=0
                first = True
                # same_items
                same = []
                for el in one_pattern: 
                    if str(el) in same:
                        el = "1_"+el
                    same.append(el)
                    
                    if first: 
                        first = False
                    else: 
                        # Edges
                        edges.append((prev, el))
                    prev = el
                    
                    # values
                    val_map[el] = w_elements[i] # Vorteil identische Elemente bekommen hier identische Zuordnung!
                    node_labels[el] = "\n".join([i.split("=")[1] for i in el.split("\n")])
                    if len(node_labels[el])>13:
                        n = 13
                        cur = node_labels[el]
                        node_labels[el] = "-\n".join([cur[i:i+n] for i in range(0, len(cur), n)])
                        
                        #node_labels[el] = node_labels[el][:13] + "-\n" + node_labels[el][13:]
                        
                    i += 1
            except:
                print("Problematicz")
                continue
            print("EDGES: "+str(one_pattern))
            
            # Set stuff
            G.add_edges_from(edges, weight=support)
            values = [val_map.get(node, 0.45) for node in G.nodes()]     
            edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
        
            # For later use: colors
            #red_edges = [('C','D'),('D','A')]
            #edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
            pos=nx.spring_layout(G) # circular_layout    random_layout       shell_layout    spring_layout    spectral_layout
            nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
            nx.draw(G, pos, node_color = values, node_size=1500)
            nx.draw_networkx_labels(G, pos, node_labels, font_size=10)
            plt.show()
            
    def draw_network_graph(self):        
        G = nx.DiGraph()
        
        
        G.add_edges_from([('A', 'B'),('C','D'),('G','D')], weight=1)
        G.add_edges_from([('D','A'),('D','E'),('B','D'),('D','E')], weight=2)
        G.add_edges_from([('B','C'),('E','F')], weight=3)
        G.add_edges_from([('C','F')], weight=4)
        
        
        val_map = {'A': 1.0,'D': 0.5714285714285714, 'H': 0.0}
        
        values = [val_map.get(node, 0.45) for node in G.nodes()]
        edge_labels=dict([((u,v,),d['weight'])
                         for u,v,d in G.edges(data=True)])
        red_edges = [('C','D'),('D','A')]
        edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
        
        pos=nx.shell_layout(G) # circular_layout    random_layout       shell_layout    spring_layout    spectral_layout
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        nx.draw(G,pos, node_color = values, node_size=1500,edge_color=edge_colors)
        
        
        labels ={}
        labels['A'] = "A"
        nx.draw_networkx_labels(G, pos, labels, font_size=16)
        
        plt.show()
        
    def draw_bn(self, bn):
        self.draw_network_graph_given(bn.E,bn.V)
        
    def draw_network_graph_given2(self, edges, nodes):     
        '''
        Input: edges [[s,e],[s,e],...]
        
        '''   
        G = nx.DiGraph()
                
        if not nodes:
            nodes = []
            for e in edges:
                if e[0] not in nodes:
                    nodes.append(e[0])
                if e[1] not in nodes:
                    nodes.append(e[1])
        
        
        G.add_edges_from(edges, weight="")
        #G.add_edges_from([('D','A'),('D','E'),('B','D'),('D','E')], weight=2)
        #G.add_edges_from([('B','C'),('E','F')], weight=3)
        #G.add_edges_from([('C','F')], weight=4)
        
        edge_labels=dict([((u,v,),d['weight'])
                         for u,v,d in G.edges(data=True)])
        
        edge_colors = len(G.edges())*['black']
        
        pos=nx.shell_layout(G) # circular_layout    random_layout       shell_layout    spring_layout    spectral_layout
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        nx.draw(G,pos, node_size=3000,edge_color=edge_colors)
                
        labels ={}
        for n in nodes:
            labels[n] = n
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
           
    def draw_network_graph_given(self, edges, nodes, probsa, vdata, eOnly):     
        # probs = dict key: nodename, value: dict(probname:probval)
        
        probs = dict()    
        if not eOnly:
            for nam in nodes:
                if not nam in probsa: cond = "[]"
                else:  cond = probsa[nam]            
                probs[nam] = dict()
                try:
                    cur_dist = vdata[nam]["cprob"][cond]
                    #print("No cprob given.")
                except:
                    pass
                
                for i in range(len(vdata[nam]["vals"])):
                    try:
                        probs[nam][vdata[nam]["vals"][i]] = cur_dist[i]
                    except:
                        pass
        
        G = nx.DiGraph()                
        if not nodes:
            nodes = []
            for e in edges:
                if e[0] not in nodes:
                    nodes.append(e[0])
                if e[1] not in nodes:
                    nodes.append(e[1])
        if eOnly:
            edges_new = []
            for e in edges:
                if e[0] in nodes and e[1] in nodes:
                    edges_new.append(e)
            edges = edges_new
            
                
        G.add_edges_from(edges, weight="")        
        #edge_labels=dict([((u,v,),d['weight'])
        #                 for u,v,d in G.edges(data=True)])        
        #edge_colors = len(G.edges())*['black']        
        #pos=nx.shell_layout(G) # circular_layout    random_layout       shell_layout    spring_layout    spectral_layout
        #nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        #nx.draw(G,pos, node_size=3000,edge_color=edge_colors)                
        #labels ={}
        #for n in nodes:
        #    labels[n] = n
        #nx.draw_networkx_labels(G, pos, labels, font_size=10)       
       
        # PLOT
        plot = Plot(plot_width=1300, plot_height=800, x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
        plot.title.text = "Temporal Dependency Bayesian Network"
        hover = HoverTool(tooltips=[("desc", "@desc"),])  
        plot.add_tools(hover, TapTool(), BoxSelectTool())
        plot.add_tools(WheelZoomTool())
        plot.add_tools(PanTool())
        
        graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0,0))
        
        desc_probs = []
        for n in nodes:
            try:
                nodeCur = probs[n]
                pr = ""
                for k in nodeCur.keys():
                    pr += "\nP("+str(k) +")="+str(nodeCur[k]) +""
                desc_probs +=  [pr]
            except:
                desc_probs +=  [""]
        node_source = ColumnDataSource(data=dict(index=nodes, desc=desc_probs))
        graph_renderer.node_renderer.data_source.data = node_source.data 
        

        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
        graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="firebrick", line_alpha=0.8, line_width=1.1)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)
        plot.renderers.append(graph_renderer)
        graph_renderer.selection_policy = NodesAndLinkedEdges()
        #graph_renderer.inspection_policy = EdgesAndLinkedNodes()
        
        # PLOT ARROWS FOR EDGES
        pos_dict = graph_renderer.layout_provider.graph_layout
        done = []
        for e in edges:            
            plot.add_layout(Arrow(end=VeeHead(line_width=2, size = 4),x_start=pos_dict[e[0]][0], y_start=pos_dict[e[0]][1], x_end=pos_dict[e[1]][0], y_end=pos_dict[e[1]][1]))
            
            font_size = "8pt"
            if not str(e[0]) in done:
                done += [str(e[0])]
                plot.add_layout(Label(x=pos_dict[e[0]][0], y=pos_dict[e[0]][1]+0.01, text=str(e[0]),text_font_size=font_size, render_mode='css', background_fill_alpha=1.0))
            if not str(e[1]) in done:
                done += [str(e[1])]
                plot.add_layout(Label(x=pos_dict[e[1]][0], y=pos_dict[e[1]][1]+0.01, text=str(e[1]), text_font_size=font_size, render_mode='css', background_fill_alpha=1.0))

        graph_renderer.edge_renderer.glyph.line_join='round' 
        output_file("interactive_graphs.html")
        show(plot)
        return
                
