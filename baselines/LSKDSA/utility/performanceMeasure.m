function measure = performanceMeasure(test_label, predict_label)

% True Positive: the number of defective modules predicted as defective
% False Negative: the number of defective modules predicted as non-defective
% False Positive: the number of non-defective modules predictied as defective
% Ture Nagative: the number of non-defective modules predictied as non-defective

% confusion matrix
%  
%                            defective             non-defective (true class)
%  ------------------------------------------------------
%   predicted   defective        TP                  FP (Type I error)
%    class      ---------------------------------------------
%               non-defective    FN (Type II error)  TN
%  -------------------------------------------------------
% 
% Pd = recall = TP / (TP+FN)
% 
% Pf = FP / (FP+TN)
% 
% preceison = TP / (TP+FP)
% 
% f-measure = recall*precision / (recall+precison)
% 

[X_coordi,Y_coordi,T_thre,AUC] = perfcurve(test_label',predict_label',1); % AUC_score

predict_label(predict_label>0.5) = 1;
predict_label(predict_label<=0.5) = 0;

test_total = length(test_label);
posNum = sum(test_label == 1); % defective
negNum = test_total - posNum;

pos_index = test_label == 1;
pos = test_label(pos_index);
pos_predicted = predict_label(pos_index);
FN = sum(pos ~= pos_predicted); % Type II error

neg_index = test_label == 0; % defective free
neg = test_label(neg_index);
neg_predicted = predict_label(neg_index);
FP = sum(neg ~= neg_predicted);  % Type I error

TP = posNum-FN;
TN = negNum-FP;

% if TP+FN ~= 0
%     pd_recall = TP/(TP+FN); 
% else
%     pd_recall = 0;
% end
% 
% if FP+TN ~= 0
%     pf = FP/(FP+TN); % negNum
% else
%     pf = 0;
% end

% if TP+FP ~= 0
%     precision = TP/(TP+FP); 
% else
%     precision = 0;
% end

% if (pd_recall+precision) ~= 0
%     f_measure = 2 * pd_recall * precision / (pd_recall+precision);
% else 
%     f_measure = 0;
% end

% if pd_recall+1-pf ~= 0
%     g_measure = 2*pd_recall*(1-pf)/(pd_recall+1-pf);
% else
%     g_measure = 0;
% end
% 
% temp = sqrt((TP+FN)*(TP+FP)*(FN+TN)*(FP+TN));
% if temp ~= 0
%     MCC = (TP*TN-FP*FN)/temp;
% else 
%     MCC = 0;
% end

%% f1-score
PRE = TP / (TP + FP);
PD    = TP / (TP + FN);
PF = FP / (FP + TN);
SP = TN / (FP + TN);

f1 = 2*PRE*PD / (PRE+PD);
balance = 1-(((0-PF)^2+(1-PD)^2)/2)^0.5;
Gmean = (PD*SP)^0.5;
MCC = ((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^0.5);
FI = (TP + FP) / (TN + FP + FN + TP);
FIR = (PD-FI)/PD;
CI  = 1 ;
CFN = 3;
Cost = (CI*(TP+FP)+CFN*FN)/(CI*(TN + FP + FN + TP));

f1(isnan(f1))=0;

% display
measure = [AUC,PD,PF,Gmean,f1,balance,MCC,FIR,Cost]; 
