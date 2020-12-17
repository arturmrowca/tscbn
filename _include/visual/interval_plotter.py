import os
import PyQt5
from PyQt5.QtWidgets import QApplication
from numpy import arange
import numpy
if os.name == "nt":  # if windows
    pyqt_plugins = os.path.join(os.path.dirname(PyQt5.__file__),
                                "..", "..", "..", "Library", "plugins")
    QApplication.addLibraryPath(pyqt_plugins)

import matplotlib.pyplot as plt
import re

class IntervalPlotter(object):
    '''
    classdocs
    '''

    def __init__(self, x_label = None, y_label = None, x_ticks = [], y_ticks = []):
        '''
        Constructor
        '''
        self._x_label = self._set_value(x_label, None, "x")
        self._y_label = self._set_value(y_label, None, "y")
        
        self._x_ticks = x_ticks
        self._y_ticks = y_ticks
        
        self._y_tick_nr_dict = dict()
        self._cnt = 0

    def plot_sequences(self, sequences, hide_no_sc =  False):
        iq = 0
        for sequence in sequences:
            iq += 1
            x_start = []
            x_stop = []
            y = []
            label = []
            for otto in sequence:
                lst_seq = sequence[otto]
                for l in lst_seq:
                    if hide_no_sc and len(lst_seq)==1:
                        try:
                            label.remove(l[0])
                        except:
                            pass
                        continue
                    x_start += [l[1]]
                    x_stop += [l[2]]
                    y += [otto]
                    label += [l[0]]

            self.plot_timeline(x_start, x_stop, y, label, y_shift=0.08, title="Sample %s" % str(iq))
            plt.show()
            plt.clf()

    def plot_timeline(self, x_start = [1, 2], x_stop = [2, 4], y = ["tick", "tack"], label = ["test", "test"], color = "k", y_shift = 0.02, title = ""):

        # Map y to ticks
        self._cnt = 0
        for val in set(y):
            if val not in self._y_tick_nr_dict:
                if not self._y_ticks: self._y_ticks = [val]
                else: self._y_ticks.append(val)
                self._y_tick_nr_dict[val] = self._cnt
            self._cnt += 1

        self._update_ticks()

        self._timelines(self._map(y, self._y_tick_nr_dict), x_start, x_stop, label, color, y_shift)
        plt.title(title)
        plt.show()        
    
    def plot_sequence(self, seq, title=""):  
        if len(seq)<1:return   
        x_start, x_stop, y, label = [], [], [], []
       
        all_my_vals = dict()       
        all_time_high= 0 
        for s in seq:
            try:
                all_my_vals[s[1]] += [[s[0], s[2]]] # [start, val]
            except:
                all_my_vals[s[1]] = [[s[0], s[2]]]
                
            if s[0]> all_time_high: all_time_high = s[0]
            
        all_time_high += 5
        for k in all_my_vals:
            fr = all_my_vals[k][0]
            
            if len(all_my_vals[k])==1:
                x_start += [fr[0]]
                x_stop += [all_time_high]
                y += [k]
                label += [fr[1]]
            else:
                for el in all_my_vals[k][1:]:
                    to = el
                    x_start += [fr[0]]
                    x_stop += [to[0]]
                    y += [k]
                    label += [fr[1]]
                    fr = to
                    
                # last element
                x_start += [fr[0]]
                x_stop += [all_time_high]
                y += [k]
                label += [fr[1]]
                
        '''for s in seq[1:]:
            to = s
            x_start += [fr[0]]
            x_stop += [to[0]]
            y += [fr[1]]
            label += [fr[2]]
            fr = to
        '''
        self.plot_timeline(x_start, x_stop, y, label, y_shift = 0.08, title = title)
    
    def safeplot_random_example_tbn(self, random_samples, vertex_data):
                
        for s in random_samples:
            
            x_start, x_stop, y, label, map = [], [], [], [], dict()
            
            for el in s:
                name = el.replace("tp_", "").replace("dL_", "")      
                if not name in map: map[name] = [name, 2, 3]
                    
                # i.e. it never occurred         
                try: 
                    if str.startswith(el, "tp_"): map[name][1] = eval("[" + s[el] + "]") 
                except: map[name] = ["", [0, 0], [0, 0]] 
                    
                # i.e. it has no end 
                try:                   
                    if str.startswith(el, "dL_"): map[name][2] = eval("[" + s[el] + "]")
                except:
                    map[name] = [map[name][0], map[name][1], [0, 0]] # i.e. it is an event
                
            # get abs. parent_times for interval start
            abs_time = {}
            k_to_proc = list(map.keys())
            i = -1
            while True:
                if len(k_to_proc)== 0: break
                i+=1
                if i >= len(k_to_proc): i = 0
                k = k_to_proc[i]
                # parent time is max of parents time  + my tprev
                try:
                    abs_time[k] = numpy.mean(map[k][1]) + max([0] + [abs_time[p[3:]] for p in vertex_data["tp_" + k]["parents"] if str.startswith(p, "tp_")])
                    k_to_proc.remove(k)
                except:
                    pass #print("Stuck") # not all parents there yet
                
                
            # plot this sample - abs_time[k] is absolute time of tprev (mean)
            for k in map:
                label += [s[k]]
                if s[k]=="Never":
                    x_start += [0] 
                    x_stop += [0]
                else:                
                    x_start += [abs_time[k]] # Start defined by tprev
                    x_stop += [abs_time[k] + numpy.mean([map[k][2]])] # plus mean of delta L
                y += [k]
                        
            self.plot_timeline(x_start, x_stop, y, label, y_shift = 0.08)

    def plot_random_example_tbn(self, random_samples, vertex_data, sep=False):
        iq = 0
        for s in random_samples:
            iq+=1
            x_start, x_stop, y, label, map = [], [], [], [], dict()

            # get abs. parent_times for interval start
            abs_time = {}
            k_to_proc = list(s.keys())
            i = -1
            
            while True:
                if len(k_to_proc)== 0: break
                i+=1
                if i >= len(k_to_proc): i = 0
                k = k_to_proc[i]
                if str.startswith(k, "dL_"): 
                    k_to_proc.remove(k)
                    continue                
                
                # parent time is max of parents time  + my tprev
                try:
                    # look for value for abs_time of all parents
                    rel_time_to_me = s["dL_"+k]
                    
                    if not vertex_data[k]["parents"]: pars = []
                    else: pars = vertex_data[k]["parents"]
                    
                    found_time = [0]
                    for p in pars:
                        found_time.append(abs_time[p])
                    
                    abs_time_of_my_earliest_parent = max(found_time)#[0]+[abs_time[p[3:]] for p in pars])
                    
                    #max([0] + [abs_time[p[3:]] for p in vertex_data["dL_" + k]["parents"] if str.startswith(p, "dL_")])
                    
              
                    abs_time[k] =  rel_time_to_me + abs_time_of_my_earliest_parent
                    k_to_proc.remove(k)
                except:
                    pass #print("Stuck") # not all parents there yet
                
                
            # plot this sample - abs_time[k] is absolute time of tprev (mean)
            overall_high= max(list(abs_time.values())) + 0.5*max(list(abs_time.values())) # buffer indicating end
            for el in s:
                if str.startswith(el, "dL_"): continue

                if not sep:
                    span = re.search("\d", el)
                    name = el[:span.start()]
                    number = int(el[span.start():])
                else:
                    name = el.split(sep)[0]
                    number = int(el.split(sep)[1])


                
                x_s = abs_time[el]
                try:
                    if sep:
                        x_e = abs_time[name +sep+ str(number + 1)]  #
                    else:
                       x_e = abs_time[name+ str(number+1)] #
                except:
                    x_e= overall_high # no next element
                
                yy = name
                lab = s[el]
                
                x_start.append(x_s)
                x_stop.append(x_e)
                y.append(yy)
                label.append(lab)
                
                
                
            self.plot_timeline(x_start, x_stop, y, label, y_shift = 0.08, title="Sample %s"%str(iq))
            plt.show()
            
            
    def pplot_random_example_tbn(self, random_samples, vertex_data):
                
        for s in random_samples:
            
            x_start, x_stop, y, label = [],[],[],[]
            relative_times = {}
            
            for el in s:
                if str.startswith(el, "dL_"): continue
                
                span = re.search("\d", el)
                name =  el[:span.start()]
                number = int(el[span.start():])
                
                # e.g. bin x0 h
                
                
                if number==0:
                    x_s = 0
                    x_e = s["dL_"+el]
                else:
                    x_s = s["dL_"+el] # dL definiert als so lange hat es gedauert bis ich aufgetreten bin
                    try:
                        x_e = s["dL_"+name+ str(number+1)] 
                    except:
                        x_e = x_s
                yy = name
                lab = s[el]
                
                x_start.append(x_s)
                x_stop.append(x_e)
                y.append(yy)
                label.append(lab)
                
                
                
            self.plot_timeline(x_start, x_stop, y, label, y_shift = 0.08)
        return
                
                
                
        
    def _map(self, vec, c_dict):
        out = []
        for v in vec:
            out.append(c_dict[v])
        return out
        
    def _timelines(self, y, xstart, xstop, text, color='b', y_shift = 0.06):
        """Plot timelines at y from xstart to xstop with given color."""   
        plt.hlines(y, xstart, xstop, color, lw=4)
        plt.vlines(xstart, [yi+0.1 for yi in y], [yi-0.1 for yi in y], color, lw=2)
        plt.vlines(xstop, [yi+0.1 for yi in y], [yi-0.1 for yi in y], color, lw=2)
        yShifted = []; v = y_shift
        for yi in y:
            v = v* -1             
            yShifted += [yi+v]
        for idx in range(len(text)):
            txt = plt.text(xstart[idx] + (xstop[idx]- xstart[idx])/2.0, yShifted[idx], text[idx], ha="center", clip_on=True)

    def set_ticks(self, x_ticks = [], y_ticks = []):
        if x_ticks: self._x_ticks = x_ticks
        if y_ticks: self._y_ticks = y_ticks
        self._update_ticks()

    def _update_ticks(self):
        char_nr = 23
        for r in range(len(self._y_ticks)):
            if len(self._y_ticks[r])>10: self._y_ticks[r] = "\n".join([self._y_ticks[r][i: i + char_nr] for i in range(0, len(self._y_ticks[r]), char_nr)])
        
        if self._x_ticks: plt.xticks(arange(len(self._x_ticks)), self._x_ticks )
        if self._y_ticks: plt.yticks(arange(len(self._y_ticks)), self._y_ticks )

    def _set_value(self, value, check, alternative):
        if value==check:
            return alternative
        else: 
            return value
