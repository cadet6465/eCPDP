addpath('../../Liblinear','../../Weka','../../TCA+')

%% load 42 defect projects
load_promise;

%% TCA+_Rnd
index        = cell2mat(CrossProjects(:,1:2));
results_mean = [];
results_std  = [];
results      = [];

for i=1:42  
    for j=1:10  
        fprintf('%i %i \n',i,j);
        
        data  = CrossProjects(index(:,1)==i,:);
        d     = data(randi(size(data,1)),:);
        src   = d{3};         
        tar   = d{4};          
        line  = d{5};         
        obs   = tar(:,end);   

        [src,tar] = tca_plus(src,tar);
        [pre,dis] = liblinear(src,tar);
        [f1,pofb20] = WekaError(obs,pre,dis,line);
        results(j,:) = [f1,pofb20]; 
    end
    results_mean(i,:) = mean(results);
    results_std(i,:) = std(results);
end