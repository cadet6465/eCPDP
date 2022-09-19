addpath('../../Liblinear','../../Weka','../../TCA+')

%% load 42 defect projects
load_promise;

%% LT_All
results = [];

for i=1:42
    fprintf('%i 42\n',i);
    src  = CrossProjectsComb{i,1};  
    tar  = CrossProjectsComb{i,2};  
    line = CrossProjectsComb{i,3};  
    obs  = tar(:,end);              
    
    [src,tar] = standard(src,tar);
    r = WekaClassify(src,tar,'J48');
    pre = r.pre;
    dis = r.dis;

    [f1,pofb20] = WekaError(obs,pre,dis,line);
    results(i,:) = [f1,pofb20];
end
