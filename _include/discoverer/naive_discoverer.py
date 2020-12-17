from _include.discoverer.prefix_structure_discoverer import PrefixStructureDiscoverer


class NaiveDiscoverer(PrefixStructureDiscoverer):

    def __init__(self, **kwargs):
        super(NaiveDiscoverer, self).__init__(**kwargs)

    def discover_structure(self, sequences):
        """
        This approach discovers the structure of a TSCBN.
        :param sequences: input sequences
        """
        _, _, event_sequences = self.create_ordered_event_sequences(sequences)
        event_sequences, nodes = self.number_event_sequences(event_sequences)
        self.data = self.get_datastructure(event_sequences)
        edges = self.discover_structure_from_statistics(self.data, nodes)
        return nodes, edges

    def number_event_sequences(self, event_sequences):
        """
        Takes the event sequences and adds occurrence numbers to the events. A naive approach is used (i-th event of
        signal Vj is numbered as Vj_i.
        :param event_sequences: ordered event sequences
        :return: numbered event sequences
        """
        nodes = set()
        numbered_event_sequences = []
        for event_sequence in event_sequences:
            counter_dict = {}
            numbered_event_sequence = []
            for event in event_sequence:
                signal = event[0]
                if signal not in counter_dict:
                    complete_name = signal + '_' + str(0)
                    nodes.add(complete_name)
                    numbered_event_sequence.append((complete_name, event[1]))
                    counter_dict.update({signal: 0})
                else:
                    complete_name = signal + '_' + str(counter_dict.get(signal) + 1)
                    nodes.add(complete_name)
                    numbered_event_sequence.append((complete_name, event[1]))
                    counter_dict.update({signal: counter_dict.get(signal) + 1})
                pass
            numbered_event_sequences.append(numbered_event_sequence)
        return numbered_event_sequences, list(nodes)

    def discover_structure_from_statistics(self, data, nodes):
        raise NotImplementedError("discover_structure_from_statistics() not implemented in class %s" % str(__class__))
