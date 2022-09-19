import pandas as pd
import math
class ManualDown():
    def __init__(self, dataset_name, label_info, Xt):
        self.dataset_name = dataset_name
        self.target_data = Xt
        self.label_info = label_info
    def _get_prob(self, size):
        a0 = -1.597994
        a1 = 0.003414518
        temp = size.transform(lambda  x: math.exp(a0+a1*x))
        prob = temp/(1+temp)
        return prob


    def run(self):
        size_metric = self.label_info[self.dataset_name]['loc_label']
        tX_by_sizemetric = self.target_data[size_metric]

        len_data = len(tX_by_sizemetric)
        defect_thresold = int(len_data/2)
        tX_by_sizemetric=tX_by_sizemetric.sort_values(ascending=False)
        index = tX_by_sizemetric.index
        prob = self._get_prob(tX_by_sizemetric)
        True_list = [True]*defect_thresold
        False_list = [False]*(len_data-defect_thresold)
        totla_list = True_list + False_list
        pred = pd.Series(totla_list,index=index)
        prob = prob.sort_index()
        pred = pred.sort_index()

        return pred, prob
