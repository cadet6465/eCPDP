function [X,Y] = normalizeTarget(data)

X = data(:,1:end-1); % data set
Y = data(:,end); % label 
Y(Y>1) = 1;
Y(Y<1) = 0;

% normaliztion 
X = zscore(X,0,1); 


