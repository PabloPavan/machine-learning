from gain import *
import numpy as np
from utils import *

class decision_tree():

    l_data = None
    l_att = []
    inference_att = None
    result_label = None
    parent = None
    l_children = []
    gain = None
    level = None
    name_att = None

    def __init__(self, l_data=None, l_att=[], inference_att=None, result_label=None, parent=None, gain=None,name_att=None, old_level=0):

        self.l_data = l_data
        self.l_att = l_att
        self.inference_att = inference_att
        self.result_label = result_label
        self.parent = parent
        self.gain = gain
        self.level = old_level+1
        self.name_att = name_att

        if len(self.l_att) < 1:
            if len(self.l_data.shape) == 2:
                l_data_temp = Counter(self.l_data.values[:, (self.l_data.shape[1] - 1)])
                self.result_label = l_data_temp.most_common(1)[0][0]
            else:
                self.result_label = self.l_data.values[-1]

            return

        if len(self.l_att) > 1:
            if len(self.l_data.shape) == 2:
                dict_y = self.l_data.values[:, (self.l_data.shape[1] - 1)]
            else:
                dict_y = self.l_data.values[-1]

            if len(np.unique(dict_y)) == 1:
                self.result_label = np.unique(dict_y)

                return
            else:
                index, gain = most_gain(self.l_data)
                if self.level == 1:
                    self.gain = gain
                name_att_inf = self.l_data.keys()[index]
                self.name_att = name_att_inf
                l_att = del_that_works(self.l_att, name_att_inf)
                labels = self.l_data.values[:, index]
                dict_labels = np.unique(labels)

                for children_labels in dict_labels:

                    dfa = self.l_data.set_index(name_att_inf, drop=True, inplace=False)

                    l_data_children = dfa.loc[children_labels]

                    if len(l_data_children.keys()) < 1:

                        if len(self.l_data.shape) == 2:
                            l_data_temp = Counter(self.l_data_children.values[:, (self.l_data_children.shape[1] - 1)])
                            self.result_label = l_data_temp.most_common(1)[0][0]
                        else:
                            self.result_label = self.l_data.values[-1]
                        return

                    else:
                        self.l_children.append(decision_tree(l_data=l_data_children, l_att=l_att, gain=gain,
                                                 inference_att=children_labels, parent=self, old_level=self.level, name_att=name_att_inf))


    def __str__(self, level=0):
        ret = "" * int(self.level) + repr(self)
        ret += "" * int(self.level) + repr(self.l_children) + "\n"
        return ret

    def __repr__(self):
        if self.level is 1:
            if self.result_label is None:
                return "ROOT [{0}] -> level: {1} gain: {2} \n\n".format(self.name_att, self.level, self.gain)
            else:
                return "\t" * int(self.level) + "ROOT[{0},[{1}]] -> results: {2}  level: {3}  gain: {4}\n\n".format(self.inference_att, self.name_att, self.result_label, self.level, self.gain)

        else:
            if self.result_label is None:
                return "\t" * int(self.level) + "CHILD [{0},[{1}]] -> level: {2}  gain: {3}\n\n".format(self.inference_att, self.name_att, self.level, self.gain)
            else:
                return "\t" * int(self.level) + "CHILD [{0},[{1}]] -> results: {2}  level: {3}  gain: {4}\n\n".format(self.inference_att, self.name_att, self.result_label, self.level, self.gain)

    def evaluate(self, instance, list_att):

        level = 1
        level_range = 1
        max_range = len(self.l_children)
        for j in range(0, max_range):
            if level_range < self.l_children[j].level:
                level_range = self.l_children[j].level
        #print(level_range)

        if self.result_label is not None:
            return self.result_label
        else:
            name_att  = self.name_att
            loc = list_att.index(name_att)
            name_att_inf = instance[loc]

            for w in range(0, level_range):

                for  child in self.l_children:
                    if child.inference_att == name_att_inf:
                        if child.result_label is not None:
                            return child.result_label
                        else:
                                loc = list_att.index(child.name_att)
                                name_att_inf = instance[loc]

