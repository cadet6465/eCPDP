function [AUC,PD,PF,Gmean,f1,balance,MCC,FIR,Cost] = WekaError(obs,pre,dis,line)
    %% confusing matrix
    TP = sum(obs==1  & pre==1);
    FP = sum(obs==1  & pre==-1);
    FN = sum(obs==-1 & pre==1);
    TN = sum(obs==-1  & pre==-1);


    %% f1-score
    PRE = TP / (TP + FP);
    PD    = TP / (TP + FN);
    PF = FP / (FP + TN);
    SP = TN / (FP + TN);

    f1 = 2*PRE*PD / (PRE+PD);
    balance = 1-(((0-PF)^2+(1-PD)^2)/2)^0.5;
    Gmean = (PD*SP)^0.5;
    MCC = ((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^0.5);
    FI = (TP + FP) / (TN + FP + FN + TP)
    FIR = (PD-FI)/PD
    CI  = 1 
    CFN = 3
    Cost = (CI*(TP+FP)+CFN*FN)/(CI*(TN + FP + FN + TP))

    f1(isnan(f1))=0;
    
    
    %% pofb20
    obs(obs==-1) = 0;
    list = sortrows([dis,line,obs],-1);
    clist = cumsum(list(:,2))./sum(list(:,2));
    pofb20 = sum(list(1:find(clist>=0.2,1),3));
    pofb20 = pofb20/sum(obs);

    %% auc
    [X,Y,T,AUC] = perfcurve(obs,dis,1);

end