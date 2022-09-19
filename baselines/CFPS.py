import pandas as pd
import copy
import warnings
import utilgroup2 as ut
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict


def getSimisore(dataset, target):
    source_mata_list = {}
    for S_project_name, data in dataset.items():
        data = ut.datasetSame(data)
        sX, tX, sY, tY = ut.preprocessing(data, target, "aa")
        avg = sX.apply(lambda x : np.average(x), axis =0)
        sd = sX.apply(lambda x : np.std(x), axis =0)
        meta = np.append(avg, sd)
        source_mata_list[S_project_name]=meta
    target_meta = np.append(tX.apply(lambda x : np.average(x), axis =0),
                            tX.apply(lambda x : np.std(x), axis =0))
    sim_score_list = {}
    for S_project_name, meta in source_mata_list.items():
        dist = np.linalg.norm(meta-target_meta)
        sim_score_list[S_project_name]=1/(1+dist)

    return sim_score_list

def getAppsore(dataset):
    appScore = pd.DataFrame(np.zeros((len(dataset.keys()), len(dataset.keys()))),
                            index=dataset.keys(), columns=dataset.keys())
    Ori_dataset = copy.deepcopy(dataset)
    for T_project_name, data in dataset.items():
        Tdataset = copy.deepcopy(Ori_dataset)
        T_data = ut.datasetSame(data)
        del (Tdataset[T_project_name])
        for S_project_name, data in Tdataset.items():
            S_data = ut.datasetSame(data)
            sX, tX, sY, tY = ut.preprocessing(S_data, T_data, "no")
            auc_list = []
            for iter in range(30):
                clssifier = RandomForestClassifier(class_weight='balanced')
                clssifier.fit(sX, sY)
                prob = clssifier.predict_proba(tX)
                # d = clssifier.decision_function(tX)
                # prob = softmax(np.c_[-d, d])
                pred = clssifier.predict(tX)

                auc, best_auc_thr, best_Gmean_thr, best_f1_thr = ut.get_auc(tY, prob[:, 1], True)
                PD, PF, Gmean, Fmeasure, Balance, MCC, FIR, COST, _ = ut.get_FIRLIR(tY, pred)
                auc_list.append(auc)
            appScore.loc[T_project_name, S_project_name] = np.array(auc).mean()
    return appScore

def getRecSore(appScore, simScore):
    project_namelist = list(simScore.keys())
    recSore = defaultdict(float)
    for project_name1 in project_namelist:
        for project_name2 in project_namelist:
            if project_name1 != project_name2 :
                recSore[project_name1] += simScore[project_name1]*appScore.loc[project_name1][project_name2]
    return recSore



if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    warnings.filterwarnings("ignore")
    THRESOLD=0.5

    PRO_LABELNM = { 'defect_label' : 'defects', 'loc_label' : 'loc'}
    JIRA_LABELNM = { 'defect_label' : 'RealBugCount', 'loc_label' : 'CountLineCode'}
    promise_setList, jira_setList, AEEEM, Relink, Auodi = ut.load_dataset()
    dataset = [promise_setList, jira_setList, AEEEM, Relink, Auodi]
    proindex=0
    outcomeList = pd.DataFrame()
    for data in dataset:
        Tdataset = data

        NUM_PROJECT = len(Tdataset)
        defect_label=PRO_LABELNM['defect_label']
        loc_label=PRO_LABELNM['loc_label']

        Ori_dataset = copy.deepcopy(Tdataset)
        tindex = 0
        for T_project_name, data in Ori_dataset.items():
            print('Target project is :', T_project_name)
            Tdataset = copy.deepcopy(Ori_dataset)
            Ori_T_data = copy.deepcopy(data)
            T_data = ut.datasetSame(data)
            del(Tdataset[T_project_name])
            temp_Tdataset = copy.deepcopy(Tdataset)
            simiscore = getSimisore(temp_Tdataset,T_data)
            print("simiscore_Done")
            appscore = getAppsore(temp_Tdataset)
            recscore = dict(getRecSore(appscore, simiscore))
            recscore = sorted(recscore.items(), key=lambda item : item[1])
            print(recscore)
            sel_source_name = recscore.pop()[0]
            print(sel_source_name)
            source_data = Tdataset[sel_source_name]
            S_data = ut.datasetSame(source_data)
            sX, tX, sY, tY = ut.preprocessing(S_data, T_data, "no")
            for iter in range(30):
                clssifier = RandomForestClassifier(class_weight='balanced')
                clssifier.fit(sX, sY)
                prob = clssifier.predict_proba(tX)
                pred = clssifier.predict(tX)
                auc, best_auc_thr, best_Gmean_thr, best_f1_thr = ut.get_auc(tY, prob[:, 1], True)
                PD, PF, Gmean, Fmeasure, Balance, MCC, FIR, COST, _ = ut.get_FIRLIR(tY, pred)
                temp_result = pd.Series([T_project_name, auc, PD, PF, Gmean, Fmeasure, Balance, MCC, FIR, COST])
                outcomeList=outcomeList.append(temp_result,ignore_index=True)
                print(outcomeList)
    outcomeList.columns = ['Target', 'AUC', 'PD', 'PF', 'Gmean', 'Fmeasure', 'Balance', 'MCC', 'FIR', 'cost']
    print(outcomeList)
    fileName = "./outcome/_" + str(proindex) + "CFPS" +"_.csv"
    outcomeList.to_csv(fileName, index=False)
    proindex += 1
