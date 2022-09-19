import numpy as np
from sklearn.linear_model import LogisticRegression


class Camargo():
    def __init__(self):
        self.clssfier = LogisticRegression(class_weight='balanced')


    def transform(self, source,target):
        col_name = source.columns
        for col in col_name:
            s_median = np.log(source[col].median()+1)
            t_median = np.log(target[col].median()+1)
            s_trans_val = np.log(1+source[col])+(s_median - t_median)
            t_trans_val = np.log(1+target[col])
            source[col] = s_trans_val
            target[col] = t_trans_val
        return source,target

    def run(self, sX, sY, tX, tY):
        self.clssfier.fit(sX, sY)
        pred = self.clssfier.predict(tX)
        prob = self.clssfier.predict_proba(tX)

        return pred, prob
