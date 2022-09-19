import pandas as pd
import copy
import warnings
import utilgroup as ut
import eCPDPClassfiler as cl



def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # pd.options.display.float_format = '{:.2f}'.format

    PRO_LABELNM = { 'defect_label' : 'defects', 'loc_label' : 'loc'}
    JIRA_LABELNM = { 'defect_label' : 'RealBugCount', 'loc_label' : 'CountLineCode'}

    NORM = 'Znorm'
    PROMISE_DATASET, JIRA_DATASET, AEEEM = ut.load_dataset()

    Tdataset = JIRA_DATASET

    outcomeList = pd.DataFrame()
    classfier = cl.SVD2Classifier(grid=False)
    for T_project_name, data in Tdataset.items():
        temp_Tdataset = copy.deepcopy(Tdataset)
        T_data = ut.datasetSame(data)
        del(temp_Tdataset[T_project_name])
        for S_project_name, data in temp_Tdataset.items():
            S_data = ut.datasetSame(data)
            Ori_S_data = copy.deepcopy(S_data)
            sX, tX, sY, tY = ut.preprocessing(S_data, T_data, 'No')
            before_dist = mmd_linear(sX,tX)
            sX1, tX1, sY, tY = ut.preprocessing(S_data, T_data, NORM)
            prepro_dist = mmd_linear(sX1,tX1)
            Aligned_sX, Aligned_tX = classfier.align_data(sX1, tX1)
            align_dist = mmd_linear(Aligned_sX, Aligned_tX)
            temp_result = pd.Series([T_project_name, S_project_name,before_dist,prepro_dist,align_dist])
            outcomeList=outcomeList.append(temp_result,ignore_index=True)
    print(outcomeList)
