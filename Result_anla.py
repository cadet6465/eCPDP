import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from numpy import std, mean, sqrt

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    result = (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)
    size = print_cohen(abs(result))
    return result, size

def print_star(pval):
    if pval <0.001:
        return "***"
    elif pval <0.01:
        return "**"
    elif pval <0.05:
        return "*"
    else:
        return "n.s."

def print_star2(pval):
    if pval <0.05:
        return "win"
    else:
        return "tie"

def print_cohen(d):
    if d <0.2:
        return "neg"
    elif d >= 0.2 and d <=0.5:
        return "small"
    elif d > 0.5 and d <=0.8:
        return "medium"
    else:
        return "large"



if __name__ == '__main__':


    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    # # RQ1,RQ2
    # warnings.filterwarnings("ignore")
    # GIS = pd.read_csv('./outcome/RQ1,2/GIS13_.csv')
    # FesCH = pd.read_csv('./outcome/RQ1,2/FesCH_.csv')
    # CFPS = pd.read_csv('./outcome/RQ1,2/CFPS_.csv')
    # CDE = pd.read_csv('./outcome/RQ1,2/CDE25_.csv')
    # TPTL = pd.read_csv('./outcome/RQ1,2/TPTL_22.csv')
    # ManualD = pd.read_csv('./outcome/RQ1,2/ManualDown20_.csv')
    # LSKDSA = pd.read_csv('./outcome/RQ1,2/LSKDSA13_.csv')
    # Camargo = pd.read_csv('./outcome/RQ1,2/Camargo22_.csv')
    # journal = pd.read_csv('./outcome/RQ1,2/_first_NORM = box-cox__Is_SVD = True__GRID = True__37_.csv')
    #
    # project_list = sorted(list(set(journal.Target)))
    # if 'avg' in project_list :
    #     project_list.remove('avg')
    # print(project_list)
    # proindex = [13,12,14,15,16,0,2,3,4,17,18,19,20,21,22,23,5,10,1,11,8,7,6,9]
    # sort_list = list(range(len(proindex)))
    # for idx, val in enumerate(proindex):
    #     sort_list[val] = project_list[idx]
    # project_list = sort_list
    # print(project_list)
    # metric_list = ['cost']
    # # metric_list = ['AUC']
    # test_list = ['win','tie','lose','neg','small','medium','large']
    #
    # source_List = {'GIS': GIS,'FesCH': FesCH,'CFPS': CFPS,'CDE': CDE,'ManualD': ManualD,'LSKDSA': LSKDSA,'Camargo': Camargo,
    #                'TPTL':TPTL}
    #
    # # result_list = {'none': none, 'qrs' : qrs, 'qrs_box' : qrs_box, 'journal' : journal}
    # avg_list = ['project','FesCH','GIS','CDE','TPTL','CFPS','ManualD','LSKDSA','Camargo','journal']
    #
    # result_avg = pd.DataFrame(columns=avg_list)
    # result_avg['project']=project_list
    #
    # result_var = pd.DataFrame(columns=avg_list)
    # result_var['project'] = project_list
    #
    #
    # target = journal
    #
    #
    # for metric in metric_list:
    #     print("       =================== Metrics is {} ===================".format(metric))
    #     result_test = pd.DataFrame(columns=avg_list)
    #     result_test['project'] = test_list
    #     result_test = result_test.fillna(0)
    #     for Sname, source in source_List.items():
    #         source_by_metric = source[["Target",metric]]
    #         target_by_metric = target[["Target",metric]]
    #         for project_name in project_list:
    #             source_by_metric_project = source_by_metric.query("Target  == @project_name")
    #             target_by_metric_project = target_by_metric.query("Target  == @project_name")
    #
    #             source_mean = mean(source_by_metric_project[metric])
    #             target_mean = mean(target_by_metric_project[metric])
    #             source_var = std(source_by_metric_project[metric])
    #             target_var = std(target_by_metric_project[metric])
    #             index = result_avg.index[(result_avg['project'] == project_name)]
    #
    #
    #             result_avg.loc[index, Sname] = round(source_mean,3)
    #             result_avg.loc[index, 'journal'] = round(target_mean,3)
    #             result_var.loc[index, Sname] = round(source_var,3)
    #             result_var.loc[index, 'journal'] = round(target_var,3)
    #
    #             # cliffs_result = cliffs_delta.cliffs_delta(target_by_metric_project[metric],source_by_metric_project[metric])
    #             c_result, c_size = cohen_d(target_by_metric_project[metric], source_by_metric_project[metric])
    #
    #             if metric in ['cost', 'PF']:
    #                 # result = ranksums(target_by_metric_project[metric], source_by_metric_project[metric], alternative='less')
    #                 result= wilcoxon(target_by_metric_project[metric], source_by_metric_project[metric], zero_method='wilcox', alternative='less')
    #                 if c_result > 0:
    #                     index = result_test.index[(result_test['project'] == 'neg')]
    #                     result_test.loc[index, Sname] += 1
    #                 else :
    #                     index = result_test.index[(result_test['project'] == c_size)]
    #                     result_test.loc[index, Sname] += 1
    #             else :
    #                 # result = ranksums(target_by_metric_project[metric],source_by_metric_project[metric], alternative='greater')
    #                 result= wilcoxon(target_by_metric_project[metric], source_by_metric_project[metric], zero_method='wilcox', alternative='greater')
    #
    #                 if c_result < 0:
    #                     index = result_test.index[(result_test['project'] == 'neg')]
    #                     result_test.loc[index, Sname] += 1
    #                 else :
    #                     index = result_test.index[(result_test['project'] == c_size)]
    #                     result_test.loc[index, Sname] += 1
    #
    #             if result.pvalue < 0.05:
    #                 index = result_test.index[(result_test['project'] == 'win')]
    #                 result_test.loc[index, Sname] += 1
    #             else:
    #                 result= wilcoxon(target_by_metric_project[metric], source_by_metric_project[metric], zero_method='wilcox')
    #                 if result.pvalue < 0.05 :
    #                     index = result_test.index[(result_test['project'] == 'lose')]
    #                     result_test.loc[index, Sname] += 1
    #                 else:
    #                     index = result_test.index[(result_test['project'] == 'tie')]
    #                     result_test.loc[index, Sname] += 1
    #
    #
    #
    #     avg_filepath = "./outcome/RQ1,2/temp/metic_{}__avg.csv".format(metric)
    #     var_filepath = "./outcome/RQ1,2/temp/metic_{}__var.csv".format(metric)
    #     test_filepath = "./outcome/RQ1,2/temp/metic_{}__test.csv".format(metric)
    #
    #     result_avg.to_csv(avg_filepath, index=False)
    #     result_var.to_csv(var_filepath, index=False)
    #     result_test.to_csv(test_filepath, index=False)







    # RQ3
    # warnings.filterwarnings("ignore")
    # NB = pd.read_csv('./outcome/RQ3/_NB_30.csv')
    # NN = pd.read_csv('./outcome/RQ3/_NN_30_.csv')
    # RF = pd.read_csv('./outcome/RQ3/_RF_30.csv')
    # SVC = pd.read_csv('./outcome/RQ3/_SVC_30.csv')
    # JN = pd.read_csv('./outcome/RQ3/_first_NORM = box-cox__Is_SVD = True__GRID = True__37_.csv')
    # # metric_list = ['AUC', 'Gmean', 'Fmeasure', 'Balance', 'MCC','FIR','cost']
    # metric_list = ['cost']
    # source_List = {'NB': NB, 'NN':NN, 'RF' : RF, 'SVC' : SVC, 'LR':JN}
    # # source_List = {'NB': NB, 'NN':NN, 'RF' : RF}
    #
    # # result_list = {'none': none, 'qrs' : qrs, 'qrs_box' : qrs_box, 'journal' : journal}
    # avg_list = ['NB','NN','RF','SVC','LR']
    #
    # for metric in metric_list:
    #     print('processing {}'.format(metric))
    #     dataset = pd.DataFrame(columns=avg_list)
    #     for Sname, source in source_List.items():
    #         value=source[[metric]]
    #         if metric == 'cost':
    #             dataset[Sname]=-value
    #         else:
    #             dataset[Sname] = value
    #     result_bayesian = autorank(dataset, alpha=0.05, order='ascending',verbose=False,effect_size='cohen_d')
    #     print(result_bayesian)
    #     plot_stats(result_bayesian)
    #     plt.title(metric)
    #     plt.show()



    #Discussion-1
    # warnings.filterwarnings("ignore")
    # none = pd.read_csv('./outcome/_first_NORM = None__Is_SVD = False__GRID = False__12_.csv')
    # znorm = pd.read_csv('./outcome/_first_NORM = Znorm__Is_SVD = False__GRID = False__11_.csv')
    # qrs = pd.read_csv('./outcome/_first123_NORM = Znorm__Is_SVD = True__GRID = False__49_.csv')
    # qrs_box = pd.read_csv('./outcome/_first_NORM = box-cox__Is_SVD = True__GRID = False__16_.csv')
    # journal = pd.read_csv('./outcome/_first_NORM = box-cox__Is_SVD = True__GRID = True__37_.csv')
    #
    # project_list = list(set(none.Target))
    # project_list.remove('avg')
    # metric_list = ['AUC', 'PD', 'PF', 'Gmean', 'Fmeasure', 'Balance', 'MCC','FIR','cost']
    # # metric_list = ['AUC']
    # test_list = ['win','tie','lose','neg','small','medium','large']
    #
    # source_List = {'none': none, 'znorm':znorm, 'qrs' : qrs, 'qrs_box' : qrs_box}
    #
    # # result_list = {'none': none, 'qrs' : qrs, 'qrs_box' : qrs_box, 'journal' : journal}
    # avg_list = ['project','none','znorm', 'qrs','qrs_box','journal']
    #
    # result_avg = pd.DataFrame(columns=avg_list)
    # result_avg['project']=project_list
    #
    # result_var = pd.DataFrame(columns=avg_list)
    # result_var['project'] = project_list
    #
    #
    # target = journal
    #
    #
    # for metric in metric_list:
    #     print("       =================== Metrics is {} ===================".format(metric))
    #     result_test = pd.DataFrame(columns=avg_list)
    #     result_test['project'] = test_list
    #     result_test = result_test.fillna(0)
    #     for Sname, source in source_List.items():
    #         # print("   =================== Compare with {} ===================".format(Sname))
    #         source_by_metric = source[["Target",metric]]
    #         target_by_metric = target[["Target",metric]]
    #         for project_name in project_list:
    #             source_by_metric_project = source_by_metric.query("Target  == @project_name")
    #             target_by_metric_project = target_by_metric.query("Target  == @project_name")
    #
    #             source_mean = mean(source_by_metric_project[metric])
    #             target_mean = mean(target_by_metric_project[metric])
    #             source_var = std(source_by_metric_project[metric])
    #             target_var = std(target_by_metric_project[metric])
    #             index = result_avg.index[(result_avg['project'] == project_name)]
    #
    #
    #             result_avg.loc[index, Sname] = round(source_mean,3)
    #             result_avg.loc[index, 'journal'] = round(target_mean,3)
    #             result_var.loc[index, Sname] = round(source_var,3)
    #             result_var.loc[index, 'journal'] = round(target_var,3)
    #
    #             cliffs_result = cliffs_delta.cliffs_delta(target_by_metric_project[metric],source_by_metric_project[metric])
    #             c_result, c_size = cohen_d(target_by_metric_project[metric], source_by_metric_project[metric])
    #
    #             if metric in ['cost', 'PF']:
    #                 # result = ranksums(target_by_metric_project[metric], source_by_metric_project[metric], alternative='less')
    #                 result= wilcoxon(target_by_metric_project[metric], source_by_metric_project[metric], zero_method='wilcox', alternative='less')
    #                 if c_result > 0:
    #                     index = result_test.index[(result_test['project'] == 'neg')]
    #                     result_test.loc[index, Sname] += 1
    #                 else :
    #                     index = result_test.index[(result_test['project'] == c_size)]
    #                     result_test.loc[index, Sname] += 1
    #             else :
    #                 # result = ranksums(target_by_metric_project[metric],source_by_metric_project[metric], alternative='greater')
    #                 result= wilcoxon(target_by_metric_project[metric], source_by_metric_project[metric], zero_method='wilcox', alternative='greater')
    #
    #                 if c_result < 0:
    #                     index = result_test.index[(result_test['project'] == 'neg')]
    #                     result_test.loc[index, Sname] += 1
    #                 else :
    #                     index = result_test.index[(result_test['project'] == c_size)]
    #                     result_test.loc[index, Sname] += 1
    #
    #             if result.pvalue < 0.05:
    #                 index = result_test.index[(result_test['project'] == 'win')]
    #                 result_test.loc[index, Sname] += 1
    #             else:
    #                 result= wilcoxon(target_by_metric_project[metric], source_by_metric_project[metric], zero_method='wilcox')
    #                 if result.pvalue < 0.05 :
    #                     index = result_test.index[(result_test['project'] == 'lose')]
    #                     result_test.loc[index, Sname] += 1
    #                 else:
    #                     index = result_test.index[(result_test['project'] == 'tie')]
    #                     result_test.loc[index, Sname] += 1
    #
    #
    #
    #     avg_filepath = "./outcome/Dis/metic_{}__avg.csv".format(metric)
    #     var_filepath = "./outcome/Dis/metic_{}__var.csv".format(metric)
    #     test_filepath = "./outcome/Dis/metic_{}__test.csv".format(metric)
    #
    #     result_avg.to_csv(avg_filepath, index=False)
    #     result_var.to_csv(var_filepath, index=False)
    #     result_test.to_csv(test_filepath, index=False)



    #             # print("Compare with " + Sname + ", typr of metric " + metric + ", target_project is " + project_name)
    #             # print("journal : {}({}) / {} : {}({})".format(target_mean,target_var,project_name,source_mean,source_var))
    #             # print("p_val : " + str(result.pvalue) + "(" + print_star(result.pvalue) + ")")
    #             # print(cliffs_result)
    #             # result_string = "({}, {})".format(c_result, c_size)
    #             # print(result_string)
    #             #
    #             # print("         ======================================")


    # Discussion-2
    # warnings.filterwarnings("ignore")
    rate_10 = pd.read_csv('./outcome/Dis/2/_first_NORM = box-cox__Is_SVD = True__GRID = True____C_RATE = 2_0_.csv')
    rate_20 = pd.read_csv('./outcome/Dis/2/_first_NORM = box-cox__Is_SVD = True__GRID = True____C_RATE = 4_22_.csv')
    rate_30 = pd.read_csv('./outcome/Dis/2/_first_NORM = box-cox__Is_SVD = True__GRID = True____C_RATE = 6_5_.csv')
    rate_40 = pd.read_csv('./outcome/Dis/2/_first_NORM = box-cox__Is_SVD = True__GRID = True____C_RATE = 8_59_.csv')
    rate_50 = pd.read_csv('./outcome/Dis/2/_first_NORM = box-cox__Is_SVD = True__GRID = True____C_RATE = 10_13_.csv')
    rate_60 = pd.read_csv('./outcome/Dis/2/_first_NORM = box-cox__Is_SVD = True__GRID = True____C_RATE = 12_39_.csv')
    rate_70 = pd.read_csv('./outcome/Dis/2/_first_NORM = box-cox__Is_SVD = True__GRID = True____C_RATE = 14_23_.csv')
    rate_80 = pd.read_csv('./outcome/Dis/2/_first_NORM = box-cox__Is_SVD = True__GRID = True____C_RATE = 16_20_.csv')
    rate_90 = pd.read_csv('./outcome/Dis/2/_first_NORM = box-cox__Is_SVD = True__GRID = True____C_RATE = 18_35_.csv')
    rate_100 =pd.read_csv('./outcome/RQ1,2/_first_NORM = box-cox__Is_SVD = True__GRID = True__37_.csv')

    promise_List = ['ant1_7', 'poi2_0', 'camel1_4', 'ivy2_0', 'jedit4_0', 'log1_0', 'xal2_4', 'vel1_6','tom6_0', 'xer1_3', 'luc2_4', 'syn1_2']
    data_list = {'rate_10':rate_10,'rate_20':rate_20,'rate_30':rate_30,'rate_40':rate_40,
                 'rate_50':rate_50,'rate_60':rate_60,'rate_70':rate_70,'rate_80':rate_80,
                 'rate_90':rate_90,'rate_100':rate_100}
    metric_list = ['AUC', 'Gmean', 'Fmeasure', 'Balance', 'MCC']

    # promise - 0:360
    # jira - 361:570
    # aeeem - 351:719

    collist= list(data_list.keys())
    df = pd.DataFrame(columns=collist)
    for metric in metric_list:
        for col_name , data in data_list.items() :
            data_promise = data.iloc[0:360]
            data_promise_by_metric = data_promise[metric]
            df[col_name] = data_promise_by_metric
        print("===================",metric,"===================")
        result_bayesian = autorank(df, alpha=0.05, order='ascending',verbose=False,effect_size='cliff_delta')
        print(result_bayesian)
        plot_stats(result_bayesian)
        plt.show()

