import pandas as pd
import numpy as np
import scipy.io.arff as arff
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import re
import os

def fnameList(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            fnameList(file_path, list_name)
        else:
            list_name.append(file_path)

def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


def GetData(filename, showType=False):
    # if 'JURECZKO' in filename:
    with open(filename, 'r') as f:
        data = f.readlines()
    x = []
    y = []
    empty = []
    # get the types of metrics from first line
    type = data[0].strip().split(';')
    type.pop()
    type.pop(0)
    # get the detail data of metrics
    for line in data[1:]:
        tmp = []
        odom = line.strip().split(';')
        # delete the project information
        for i in range(3):
            odom.pop(0)
        for i in range(len(odom)):
            if is_number(odom[i]):
                tmp.append(float(odom[i]))
            else:
                if i not in empty:
                    empty.append(i)
                tmp.append(0)
        if tmp.pop() > 0:
            y.append(1)
        else:
            y.append(-1)
        x.append(tmp)
    x = np.delete(np.asarray(x), empty, axis=1)
    empty = sorted(empty)
    for i in range(len(empty)):
        type.pop(empty[len(empty) - i - 1])
    if showType:
        return x, np.asarray(y), type
    else:
        return x, np.asarray(y)


def load_dataset():

    ant1_7 = pd.read_csv('./dataset/jeriko/csv_result-ant-17.csv')
    poi2_0 = pd.read_csv('./dataset/jeriko/csv_result-poi-20.csv')
    camel1_4 = pd.read_csv('./dataset/jeriko/csv_result-camel-14.csv')
    ivy2_0= pd.read_csv('./dataset/jeriko/csv_result-ivy-20.csv')
    jedit4_0 = pd.read_csv('./dataset/jeriko/csv_result-jedit-40.csv')
    log1_0= pd.read_csv('./dataset/jeriko/csv_result-log4j-10.csv')
    xal2_4=pd.read_csv('./dataset/jeriko/csv_result-xalan-24.csv')
    vel1_6=pd.read_csv('./dataset/jeriko/csv_result-velocity-16.csv')
    tom6_0=pd.read_csv('./dataset/jeriko/csv_result-tomcat.csv')
    xer1_3=pd.read_csv('./dataset/jeriko/csv_result-xerces-13.csv')
    luc2_4 = pd.read_csv('./dataset/jeriko/lucene.csv')
    syn1_2 = pd.read_csv('./dataset/jeriko/synapse.csv')

    jira_act = load_arff('./dataset/jira/activemq-5.0.0.arff')
    jira_der = load_arff('./dataset/jira/derby-10.5.1.1.arff')
    jira_gro = load_arff('./dataset/jira/groovy-1_6_BETA_1.arff')
    jira_hba = load_arff('./dataset/jira/hbase-0.94.0.arff')
    jira_hiv = load_arff('./dataset/jira/hive-0.9.0.arff')
    jira_wic = load_arff('./dataset/jira/wicket-1.3.0-beta2.arff')
    jira_jru = load_arff('./dataset/jira/jruby-1.1.arff')

    AEEEM_EQ = pd.read_csv('./dataset/aeeem/csv_result-EQ.csv')
    AEEEM_JDT = pd.read_csv('./dataset/aeeem/csv_result-JDT.csv')
    AEEEM_LC = pd.read_csv('./dataset/aeeem/csv_result-LC.csv')
    AEEEM_ML = pd.read_csv('./dataset/aeeem/csv_result-ML.csv')
    AEEEM_PDE = pd.read_csv('./dataset/aeeem/csv_result-PDE.csv')


    promise_setList = {'ant1_7': ant1_7, 'poi2_0':poi2_0, 'camel1_4':camel1_4, 'ivy2_0':ivy2_0,
                       'jedit4_0':jedit4_0, 'log1_0':log1_0, 'xal2_4':xal2_4, 'vel1_6':vel1_6,
                       'tom6_0':tom6_0, 'xer1_3':xer1_3, 'luc2_4':luc2_4, 'syn1_2':syn1_2}

    jira_setList = {'jira_act':jira_act, 'jira_der':jira_der, 'jira_gro':jira_gro, 'jira_hba':jira_hba,
                    'jira_hiv':jira_hiv, 'jira_wic':jira_wic, 'jira_jru':jira_jru}

    AEEEM = {'AEEEM_EQ':AEEEM_EQ, 'AEEEM_JDT':AEEEM_JDT, 'AEEEM_LC':AEEEM_LC, 'AEEEM_ML':AEEEM_ML,
                    'AEEEM_PDE':AEEEM_PDE}

    return promise_setList,  jira_setList, AEEEM
def preprocessing(s_data,t_data,NORM):

    sY = s_data.iloc[:, -1]
    tY = t_data.iloc[:, -1]

    sX = s_data.iloc[:, 0:-1]
    tX = t_data.iloc[:, 0:-1]

    if NORM == 'Znorm':
        scaler = StandardScaler()
        scaler.fit(sX)
        sX = scaler.transform(sX)
        tX = scaler.transform(tX)

    elif NORM == 'MinMax':
        scaler = MinMaxScaler()
        scaler.fit(sX)
        sX = scaler.transform(sX)
        tX = scaler.transform(tX)

    elif NORM == 'Power':
        transformer = make_column_transformer(
            (PowerTransformer(), sX.columns), remainder='passthrough')
        transformer.fit(sX)
        sX = transformer.transform(sX)
        tX = transformer.transform(tX)

    elif NORM == 'Robust':
        scaler = RobustScaler()
        scaler.fit(sX)
        sX = scaler.transform(sX)
        tX = scaler.transform(tX)

    elif NORM == 'box-cox':
        transformer = make_column_transformer(
            (PowerTransformer(method='box-cox', standardize=False), sX.columns), remainder='passthrough')
        # sX[sX<0]=0
        # tX[tX<0]=0

        sX = sX + 0.00001
        sX.iloc[0,:] = sX.iloc[0,:] + 0.00001
        tX = tX + 0.00001
        transformer.fit(sX)

        sX = transformer.transform(sX)
        tX = transformer.transform(tX)

        scaler = StandardScaler()
        scaler.fit(sX)
        sX = scaler.transform(sX)
        tX = scaler.transform(tX)
    else:
        a=True

    return sX, tX, sY, tY


def toNominal(data):
    data.loc[data['bug'] >= 1, 'bug'] = 1
    data.loc[data['bug'] <= 0, 'bug'] = 0
    d = {1: True, 0: False}
    data['bug'] = data['bug'].map(d)
    return data


def datasetSame(dataset):
    dfcolname=dataset.columns
    if 'id' in dfcolname :
        dataset = dataset.drop(["id"], axis=1)
    dataset_colname = np.append(dataset.columns.values[:-1], ['Defect'])
    dataset.columns = dataset_colname
    dataset = classSame(dataset)
    return dataset


def classSame(dataset):
    if isinstance(dataset['Defect'][0], str):
        dataset['Defect'] = np.where(dataset['Defect'] == 'clean', False, True)
    elif isinstance(dataset['Defect'][0], np.integer) or isinstance(dataset['Defect'][0], np.float64):
        dataset['Defect'] = np.where(dataset['Defect'] == 0, False, True)
    else:
        dataset = dataset
    return dataset


def load_arff(root):
    data = arff.loadarff(root)
    df = pd.DataFrame(data[0])
    return(df)


def get_auc(cl, prob, pos_label):
    fpr, tpr, thresholds = metrics.roc_curve(cl, prob, pos_label=pos_label)
    gmean = np.sqrt(tpr * (1 - fpr))
    auc = metrics.auc(fpr, tpr)
    J = tpr - fpr
    ix = np.argmax(J)
    best_auc_thr = thresholds[ix]

    index = np.argmax(gmean)
    best_Gmean_thr = thresholds[index]

    precision, recall, thresholds = metrics.precision_recall_curve(cl, prob)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(np.nan_to_num(fscore))
    best_f1_thr = thresholds[ix]

    return auc, best_auc_thr,best_Gmean_thr,best_f1_thr

def predtype(real, prob):
    if(real==True and prob == True):
        return "TP"
    elif(real==False and prob == False):
        return "TN"
    elif(real==True and prob == False):
        return "FN"
    else:
        return "FP"

def get_FIRLIR(defect_label,prob):

    tn, fp, fn, tp = metrics.confusion_matrix(defect_label, prob).ravel()
    # print(metrics.confusion_matrix(defect_label, prob))
    # print(tp, fn,fp,tn )
    PD = tp / (tp + fn) #recall
    PF = fp / (fp + tn) #pf
    SP = tn / (fp + tn)
    PRE = tp / (tp + fp)
    CI_u = 1
    CFN_u = 3
    CI_I = 2
    CFN_I = 6
    FI = (tp + fp) / (tn + fp + fn + tp)
    FIR = (PD-FI)/PD
    Cost = (CI_u*(tp+fp)+CFN_u*fn)/(CI_I*(tn + fp + fn + tp))
    Gmean = (PD*SP)**0.5
    Fmeasure = 2*PRE*PD/(PRE+PD)
    Balance = 1-(((0-PF)**2+(1-PD)**2)/2)**0.5
    MCC = metrics.matthews_corrcoef(defect_label, prob)
    Costeff = fn/(fn+tn)

    return PD,PF,Gmean,Fmeasure,Balance,MCC,FIR,Cost,Costeff
