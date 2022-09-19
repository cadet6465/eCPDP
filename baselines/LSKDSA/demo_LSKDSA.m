clear all
close all
clc

addpath('.\liblinear\');
addpath('.\utility\')

% Load data 
load('.\data\DATA.mat')
pro_name = data(:,2); % project name

methodName = 'LSKDSA'; 
savePath = '.\output\';
if exist(savePath,'dir') == 0
    mkdir(savePath);
end

% setting paramters
ratio = 0.9; % select 90% of source modules as training set
rep = 1; % repeat 30 times
time = zeros(size(data,1),rep); % running time
expRESULT = cell(size(data,1),1);

for i = 1:size(data,1)  % dataset
    tar_name = pro_name{i,1};
    tar_name
    target = data{i,1};
    [Xt,Yt] = normalizeTarget(target'); % normalize target data

    if i<=5 % NASA
        Xs = data(1:5,:);
    elseif i>5 && i<=10 % AEEEM
        Xs = data(6:10,:);
    elseif i>10 % ReLink
        Xs = data(11:13,:);
    end
    
    t = zeros(1,rep); % time
    MEASURE = cell(rep,1);
    for loop = 1:rep
        measure = [];
        tStart = tic; % timer start 
        
        for j = 1:size(Xs,1)
            src_name = Xs{j,2};
            src_name
            if strcmp(tar_name,src_name) == 0
                source = Xs{j,1};
                IDX = Xs{j,3};
                
                % normalize source data
                [Xss,Ys] = normalizeSource(source,IDX(loop,:),ratio);
                
                y = LSKDSA(Xss,Ys,Xt',Yt');
                mea = performanceMeasure(Yt, y');
                mea
                measure = [measure; mea];
            end
        end
        % timer end
        tElapsed = toc(tStart);
        t(1,loop)  = tElapsed;
        MEASURE{loop,1} = measure;
    end
    time(i,:) = t;
    expRESULT{i,1} = MEASURE;
end

save([savePath,methodName,'_EXPRESULT3.mat'], 'expRESULT','time')
disp('running programe done !')

