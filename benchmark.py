from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from baselines import CDE_SMOTE as CDE
from baselines import Manunal as MD
from baselines import Camargo as CM

from sklearn.linear_model import LogisticRegression

from baselines import FeSCH as FS
from baselines import GIS as GI
import pandas as pd
import warnings
import copy
import utilgroup2 as ut
import time

def get_agg_project(dataset):
    df = pd.DataFrame()
    for S_project_name, data in dataset.items():
        data=ut.datasetSame(data)
        df = df.append(data, ignore_index=True)
    return df

if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    warnings.filterwarnings("ignore")
    THRESOLD=0.5
    ITER = 30
    PRO_LABELNM = { 'defect_label' : 'defects', 'loc_label' : 'loc'}
    JIRA_LABELNM = { 'defect_label' : 'RealBugCount', 'loc_label' : 'CountLineCode'}
    AEEEM_LABELNM = { 'defect_label' : 'class', 'loc_label' : 'ck_oo_numberOfLinesOfCode'}

    n_promise_setList, promise_setList, o_promise_setList, jira_setList, AEEEM, Relink, Auodi = ut.load_dataset()
    dataset = {'promise': promise_setList, 'jira':jira_setList, 'aeeem': AEEEM}
    dataset_label = {'promise': PRO_LABELNM, 'jira':JIRA_LABELNM, 'aeeem': AEEEM_LABELNM}
    proindex=0
    BASELINE = 'Camargo'
    outcomeList = pd.DataFrame()
    for name, data in dataset.items():
        Tdataset = data

        NUM_PROJECT = len(Tdataset)
        # defect_label=PRO_LABELNM['defect_label']
        # loc_label=PRO_LABELNM['loc_label']

        Ori_dataset = copy.deepcopy(Tdataset)
        tindex = 0
        for T_project_name, data in Tdataset.items():
            print('Target project is :', T_project_name)
            Tdataset = copy.deepcopy(Ori_dataset)
            Ori_T_data = copy.deepcopy(data)

            T_data = ut.datasetSame(data)

            del(Tdataset[T_project_name])

            S_project = get_agg_project(Tdataset)
            S_data = ut.datasetSame(S_project)
            for iter in range(ITER):
                print("================"+ str(iter) + "th Trails ====================")
                if BASELINE == 'CDE' :
                    sX, tX, sY, tY = ut.preprocessing(S_data, T_data, "None")
                    clssifier1 = DecisionTreeClassifier()
                    clssifier2 = RandomForestClassifier()
                    clssifier3 = GaussianNB()
                    votingC = VotingClassifier(estimators=[('c1',clssifier1),
                                                           ('c2',clssifier2),
                                                           ('c3',clssifier3)],voting='soft')
                    cde =CDE.CDE_SMOTE(votingC)
                    pred,prob = cde.run(sX, sY, tX, tY)

                    auc, best_auc_thr, best_Gmean_thr, best_f1_thr = ut.get_auc(tY, prob[:,1], True)
                    PD, PF, Gmean, Fmeasure, Balance, MCC,FIR, COST,_ = ut.get_FIRLIR(tY, pred)

                elif BASELINE == 'FesCH' :
                    sX, tX, sY, tY = ut.preprocessing(S_data, T_data, "None")
                    clssifier1 = LogisticRegression(class_weight='balanced')
                    FSCH =FS.FeSCH()
                    sX, sY, tX, tY = FSCH.run(sX, sY, tX, tY)
                    clssifier1.fit(sX, sY)
                    pred=clssifier1.predict(tX)
                    prob=clssifier1.predict_proba(tX)
                    auc, best_auc_thr, best_Gmean_thr, best_f1_thr = ut.get_auc(tY, prob[:, 1], True)
                    PD, PF, Gmean, Fmeasure, Balance, MCC,FIR, COST,_ = ut.get_FIRLIR(tY, pred)

                elif BASELINE == 'GIS':
                    sX, tX, sY, tY = ut.preprocessing(S_data, T_data, "None")
                    clssifier1 = GaussianNB()
                    GIS = GI.GIS(clssifier1,'None',3,3)
                    pred,prob = GIS.run(sX, sY, tX, tY)
                    print(prob.shape, tY.shape)
                    auc, best_auc_thr, best_Gmean_thr, best_f1_thr = ut.get_auc(tY, prob, True)
                    PD, PF, Gmean, Fmeasure, Balance, MCC,FIR, COST,_ = ut.get_FIRLIR(tY, pred)

                elif BASELINE == 'ManualDown':
                    sX, tX, sY, tY = ut.preprocessing(S_data, T_data, "None")
                    clssifier1 = MD.ManualDown(name,dataset_label,tX)
                    pred,prob = clssifier1.run()
                    print(prob.shape, tY.shape)
                    auc, best_auc_thr, best_Gmean_thr, best_f1_thr = ut.get_auc(tY, prob, True)
                    PD, PF, Gmean, Fmeasure, Balance, MCC,FIR, COST,_ = ut.get_FIRLIR(tY, pred)

                elif BASELINE == 'Camargo':
                    CM_I = CM.Camargo()
                    agg_source_X = pd.DataFrame()
                    agg_source_Y = pd.Series()
                    for name, S_project in Tdataset.items():
                        S_data = ut.datasetSame(S_project)
                        sX, tX, sY, tY = ut.preprocessing(S_data, T_data, "None")
                        sX, tX= CM_I.transform(sX,tX)
                        agg_source_X = pd.concat([agg_source_X,sX],ignore_index=True)
                        agg_source_Y = pd.concat([agg_source_Y,sY],ignore_index=True)
                    pred, prob = CM_I.run(agg_source_X,agg_source_Y,tX,tY)
                    print(prob.shape, tY.shape)
                    auc, best_auc_thr, best_Gmean_thr, best_f1_thr = ut.get_auc(tY, prob[:, 1], True)
                    PD, PF, Gmean, Fmeasure, Balance, MCC, FIR, COST, _ = ut.get_FIRLIR(tY, pred)
                temp_result = pd.Series([T_project_name, auc, PD, PF, Gmean, Fmeasure, Balance, MCC,FIR, COST])
                outcomeList = outcomeList.append(temp_result, ignore_index=True)
    outcomeList.columns = ['Target', 'AUC', 'PD', 'PF', 'Gmean', 'Fmeasure', 'Balance', 'MCC','FIR','cost']
    tm = (time.gmtime(time.time())).tm_sec
    fileName = "./outcome/" + BASELINE + str(tm) + "_.csv"
    print(fileName)
    outcomeList.to_csv(fileName, index=False)

