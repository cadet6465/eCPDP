clear all
close all
clc

addpath('.\utility\')
addpath('.\liblinear\')

load_promise
pro_name = Projects(:,1); % project name
methodName = 'LSKDSA'; 

savePath = '.\output\';
if exist(savePath,'dir') == 0
    mkdir(savePath);
end

ratio = 0.9; % select 90% of source modules as training set
rep = 30; % repeat 30 times
expRESULT = cell(size(pro_name,1)*rep,2);
num=size(pro_name,1);
measure = [];
ori_Projects2 = Projects;

for loop = 1:rep
    ori_Projects = ori_Projects2;
    ra = randperm(size(ori_Projects,1));
%     aeeem[1,5,4,3,2]
%     jeriko[1,6,2,3,4,5,9,8,7,10,11,12]
%     jira[1,2,3,4,5,7,6]
    ori_Projects = ori_Projects(ra,:);

    for i = 1:size(ori_Projects)
        Projects = ori_Projects;
        tatget_name =Projects{i,1}; 
        
        target = Projects{i,2}; 
        [Xt,Yt] = normalizeTarget(target); % normalize target data
        Projects(i,:) = [];
        agg_Xs = [];
        agg_Ys = [];
        disp(size(Projects))
        for j = 1:size(Projects)
            source_name = Projects{j,1};
            source = Projects{j,2};
            [Xs,Ys] = normalizeTarget(source); % normalize target data
            agg_Xs = cat(1,agg_Xs,Xs);
            agg_Ys = cat(1,agg_Ys,Ys);
        end
        [agg_Xs, agg_Ys] = normalizeSource(agg_Xs,agg_Ys,ratio);
        agg_Xs = agg_Xs';
        agg_Ys = agg_Ys';
        Xt = Xt';
        Yt = Yt';
        y = LSKDSA(agg_Xs,agg_Ys,Xt,Yt);
        mea = performanceMeasure(Yt, y);
%         measure = [measure; mea];
        expRESULT{i+(loop-1)*num,1}=tatget_name
        expRESULT{i+(loop-1)*num,2}=mea
    end
end


save([savePath,methodName,'_EXPRESULT_aeeem.mat'], 'expRESULT')
disp('running programe done !')

