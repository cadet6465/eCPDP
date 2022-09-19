import pandas as pd
import copy
import warnings
import utilgroup as ut
import eCPDPClassfiler as cl
import numpy as np
import time
import winsound
import random
from tqdm import tqdm


def pick_col(data,pick_rate,pick_list):

    chose_col = False
    reshaped_data = {}

    for project_name, project_data in data.items():
        project_data = ut.datasetSame(project_data)
        if chose_col == False:
            len_col = project_data.shape[1]
            is_not_include = True
            while is_not_include :
                chosen_col = random.sample(range(len_col-1), pick_rate)
                chosen_col.sort()
                if chosen_col not in pick_list:
                    print("\nChosen_col",chosen_col)
                    pick_list.append(chosen_col[:])
                    print("\nPick_list", pick_list)
                    is_not_include = False
            chosen_col.append(len_col - 1)
            chose_col = True
        temp = project_data.iloc[:,chosen_col]
        reshaped_data[project_name] =temp
    return reshaped_data

if __name__ == '__main__':
    start = time.time()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    warnings.filterwarnings("ignore")
    THRESOLD=0.50

    # [Znorm,box-cox]
    NORM = 'box-cox'
    IS_SVD = True
    GRID = True
    promise_setList, jira_setList, AEEEM = ut.load_dataset()
    dataset = [promise_setList]
    CHOOSE_PER = [2,4,6,8,10,12,14,16,18]


    for cho_per in CHOOSE_PER :
        outcomeList = pd.DataFrame()
        for data in dataset:
            Tdataset = data
            NUM_PROJECT = len(Tdataset)
            tindex = 0
            pick_list = []
            for i in tqdm(range(30)):
                Ori_dataset = copy.deepcopy(Tdataset)
                Ori_dataset = pick_col(Ori_dataset,cho_per,pick_list)
                for T_project_name, data in Ori_dataset.items():
                    Tdataset1 = copy.deepcopy(Ori_dataset)
                    T_data = data
                    del(Tdataset1[T_project_name])
                    outcome_prob_T = np.zeros((T_data.shape[0], NUM_PROJECT-1))
                    outcome_prob_F = np.zeros((T_data.shape[0], NUM_PROJECT-1))
                    outcome_pred = np.zeros((T_data.shape[0], NUM_PROJECT- 1))

                    sindex = 0
                    s_pro_name = []
                    thr_list = []
                    for S_project_name, data in Tdataset1.items():
                        s_pro_name.append(S_project_name)
                        # S_data = ut.datasetSame(data)
                        S_data = data
                        sX, tX, sY, tY = ut.preprocessing(S_data,T_data,NORM)
                        classfier = cl.SVD2Classifier(grid =GRID)

                        if IS_SVD == False:
                            classfier.fit(sX, sY)
                            prob, pred = classfier.predict(tX)
                        else :
                            Aligned_sX, Aligned_tX = classfier.align_data(sX, tX)
                            FSed_sX, FSed_tX, result = classfier.feature_select(Aligned_sX, sY, Aligned_tX)
                            classfier.fit(FSed_sX, sY)
                            prob, pred = classfier.predict(FSed_tX)
                        outcome_prob_T[:,sindex] = prob[:,1]
                        outcome_prob_F[:,sindex] = prob[:,0]
                        outcome_pred[:,sindex]=pred
                        sindex = sindex+1
                    T_prob = pd.DataFrame(outcome_prob_T,columns=s_pro_name)
                    T_prob['Tmean2'] = T_prob.apply(np.average, axis=1)
                    final_prob = T_prob.Tmean2
                    result = pd.concat([final_prob, T_data.iloc[:, -1]], axis=1)
                    result.columns = ['probT', 'Defect']
                    auc, best_auc_thr,best_Gmean_thr,best_f1_thr =ut.get_auc(result.Defect, result.probT, True)

                    pred = result.probT.apply(lambda x: True if x > THRESOLD else False)
                    PD, PF, Gmean, Fmeasure, Balance, MCC, FIR, COST, costeff = ut.get_FIRLIR(tY, pred)

                    temp_result = pd.Series([T_project_name, auc, PD, PF, Gmean, Fmeasure, Balance, MCC, FIR, COST,costeff])
                    outcomeList=outcomeList.append(temp_result,ignore_index=True)
                    # print(outcomeList)


        outcomeList.columns = ['Target', 'AUC', 'PD', 'PF', 'Gmean', 'Fmeasure', 'Balance', 'MCC','FIR','cost','costeff']
        avg_data = ['avg', outcomeList['AUC'].mean(),outcomeList['PD'].mean(),outcomeList['PF'].mean(),outcomeList['Gmean'].mean(),
                    outcomeList['Fmeasure'].mean(),outcomeList['Balance'].mean(),outcomeList['MCC'].mean(),outcomeList['FIR'].mean(),outcomeList['cost'].mean(),
                    outcomeList['costeff'].mean()]
        outcomeList.loc[len(outcomeList)]=avg_data

        e_tm = time.gmtime(time.time())
        tm = (time.gmtime(time.time())).tm_sec

        fileName = "./outcome/_first" + "_NORM = " + NORM + "__Is_SVD = " + str(IS_SVD) +  "__GRID = " + str(GRID) + "__" "__C_RATE = "+ str(cho_per) + "_" + str(tm) + "_.csv"

        outcomeList.to_csv(fileName, index=False)

        print(fileName)
        print("time :", time.time() - start)
    winsound.Beep(frequency=440, duration=1000)


