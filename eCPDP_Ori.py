import pandas as pd
import copy
import warnings
import utilgroup as ut
import eCPDPClassfiler as cl
import numpy as np
import time
import collections
import winsound
from tqdm import tqdm

if __name__ == '__main__':
    start = time.time()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    warnings.filterwarnings("ignore")
    THRESOLD=0.50

    # [Znorm,box-cox]
    NORM = 'Znorm'
    IS_SVD = True
    GRID = False
    promise_setList, jira_setList, AEEEM = ut.load_dataset()
    dataset = [promise_setList, jira_setList, AEEEM]

    outcomeList = pd.DataFrame()
    for data in dataset:
        Tdataset = data

        NUM_PROJECT = len(Tdataset)
        Ori_dataset = copy.deepcopy(Tdataset)
        tindex = 0
        for i in tqdm(range(30)):
            for T_project_name, data in Ori_dataset.items():
                # print('Target project is :', T_project_name)
                Tdataset = copy.deepcopy(Ori_dataset)
                T_data = ut.datasetSame(data)
                counts = collections.Counter(list(T_data.iloc[:,-1]))
                del(Tdataset[T_project_name])

                outcome_prob_T = np.zeros((T_data.shape[0], NUM_PROJECT-1))
                outcome_prob_F = np.zeros((T_data.shape[0], NUM_PROJECT-1))
                outcome_pred = np.zeros((T_data.shape[0], NUM_PROJECT - 1))
                dist = np.zeros((NUM_PROJECT - 1))

                sindex = 0
                s_pro_name = []
                thr_list = []
                for S_project_name, data in Tdataset.items():
                    s_pro_name.append(S_project_name)
                    S_data = ut.datasetSame(data)
                    # Applied at once for convenience, calculation is independent of each target model data.
                    sX, tX, sY, tY = ut.preprocessing(S_data,T_data,NORM)
                    classfier = cl.SVD2Classifier(grid =GRID)

                    if IS_SVD == False:
                        classfier.fit(sX, sY)
                        prob, pred = classfier.predict(tX)

                    else :
                        # Applied at once for convenience, calculation is independent of each target model data.
                        Aligned_sX, Aligned_tX = classfier.align_data(sX, tX)
                        FSed_sX, FSed_tX, result = classfier.feature_select(Aligned_sX, sY, Aligned_tX)
                        classfier.fit(FSed_sX, sY)
                        prob, pred = classfier.predict(FSed_tX)

                    outcome_prob_T[:, sindex] = prob[:,1]
                    outcome_prob_F[:, sindex] = prob[:,0]
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

    outcomeList.columns = ['Target', 'AUC', 'PD', 'PF', 'Gmean', 'Fmeasure', 'Balance', 'MCC','FIR','cost','costeff']
    avg_data = ['avg', outcomeList['AUC'].mean(),outcomeList['PD'].mean(),outcomeList['PF'].mean(),outcomeList['Gmean'].mean(),
        outcomeList['Fmeasure'].mean(),outcomeList['Balance'].mean(),outcomeList['MCC'].mean(),outcomeList['FIR'].mean(),outcomeList['cost'].mean(),
                outcomeList['costeff'].mean()]
    outcomeList.loc[len(outcomeList)]=avg_data
    e_tm = time.gmtime(time.time())

    tm = (time.gmtime(time.time())).tm_sec

    fileName = "./outcome/_first123" + "_NORM = " + NORM + "__Is_SVD = " + str(IS_SVD) +  "__GRID = " + str(GRID) + "__" +  str(tm) + "_.csv"

    outcomeList.to_csv(fileName, index=False)

    print(fileName)
    print("time :", time.time() - start)
    winsound.Beep(frequency=440, duration=1000)





