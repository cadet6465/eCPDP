import numpy as np
import warnings
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import softmax


from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


class SVD2Classifier(object):

    def __init__(self, grid=False):

        # Set atttributes

        # KNN
        # if grid :
        #     grid = {
        #     'n_neighbors' : list(range(1,20)),
        #     'weights': ["uniform", "distance"],
        #     'algorithm': ['ball_tree', 'kd_tree', 'brute']
        #     }
        #     self.ori_clssifier = KNeighborsClassifier()
        #     self.clf = GridSearchCV(self.ori_clssifier, param_grid=grid,scoring='f1')
        # else :
        #     self.clf = KNeighborsClassifier()

        # RF
        # if grid :
        #     grid = {
        #         'n_estimators': [10,30,60,100],
        #         'criterion' : ['gini', 'entropy'],
        #         'min_samples_split': [2 ,4, 6, 8, 16, 20]
        #     }
        #     self.ori_clssifier = RandomForestClassifier(class_weight='balanced')
        #     self.clf = GridSearchCV(self.ori_clssifier, param_grid=grid,scoring='f1')
        # else :
        #     self.clf = RandomForestClassifier(class_weight='balanced')

        # NB
        # if grid :
        #     grid = {
        #         'var_smoothing' : np.logspace(0,-9, num=50)
        #     }
        #     self.ori_clssifier = GaussianNB()
        #     self.clf = GridSearchCV(self.ori_clssifier, param_grid=grid,scoring='f1')
        # else :
        #     self.clf = GaussianNB()

        # SCV
        # if grid:
        #     if grid:
        #         grid = {
        #             'C': [0.001, 0.01, 0.1, 1, 10, 100],
        #             'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        #         }
        #     self.ori_clssifier = SVC(class_weight='balanced', probability=True)
        #     self.clf = GridSearchCV(self.ori_clssifier, param_grid=grid, scoring='f1', n_jobs=4)
        # else:
        #     self.clf = SVC(class_weight='balanced')

        # LR
        if grid :
            grid = {
                'C' : [100, 10, 1.0, 0.1, 0.01],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'penalty' : ['none', 'l1', 'l2', 'elasticnet']
            }
            clssifier = LogisticRegression(class_weight='balanced',n_jobs=4)
            self.clf = GridSearchCV(clssifier, param_grid=grid,scoring='balanced_accuracy',n_jobs=4)
        else :
            self.clf = LogisticRegression(class_weight='balanced',n_jobs=4)

        self.is_trained = False


    def align_data(self,X, Z):

        def get_svd(dataset):
            t_Dataset = dataset.T
            svdresult = np.linalg.svd(t_Dataset, full_matrices=False)
            return (svdresult)

        SVDX = get_svd(X)
        U = SVDX[0]
        D = SVDX[1]
        V_T = SVDX[2]
        Z_T = Z.T
        CX = np.dot(np.diag(D), V_T).T


        # Applied at once for convenience, calculation is independent of each target model data.
        CZ = np.dot(U.T, Z_T).T

        # Align only a target module data
        # t_CZ = np.dot(U.T, Z_T[:,0]).T
        # print('t_CZ is ', CZ[0,:])
        # Check first module data, after calculation at once
        # print('CZ is ', CZ[0,:])
        # Two resurlts are same


        return CX, CZ


    def feature_select(self,CX, Y, CZ):

        self.clf.fit(CX, Y)
        imps = permutation_importance(self.clf, CX, Y, scoring='balanced_accuracy',n_repeats=10, n_jobs=4)
        importances = imps.importances_mean
        self.clf.coef_ = importances
        sel_model = SelectFromModel(self.clf, prefit=True)

        trunCX = sel_model.transform(CX)
        trunCZ = sel_model.transform(CZ)
        result=True

        return trunCX, trunCZ, result



    def fit(self, trunCX, Y):
        self.clf.fit(trunCX, Y)
        self.is_trained = True

    def predict(self, trunCZ):
        probs = self.clf.predict_proba(trunCZ)
        preds = self.clf.predict(trunCZ)
        # d = self.clf.decision_function(trunCZ)
        # probs = softmax(np.c_[-d, d])
        return probs, preds

