function [kX] = KNN(X,Y,k)  
%   Input:   
%   X is the train data set
%   Y the test data with P*N  
%   k is the K neighbor parameter  
%   kX the selected neighbor data from X

N = size(Y,2);
index = [];
for i = 1:N
    D = pdist2(X', Y(:,i)');
    [~, neighbors] = sort(sqrt(D'));

    % The neighbors are the index of top k nearest points.  
    neighbors = neighbors(1:k);  
    index = [index,neighbors];
end
index = unique(index);
kX = X(:,index);

