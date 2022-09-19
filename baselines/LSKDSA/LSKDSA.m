function [y, predict_label] = LSKDSA(Xs,Ys,Xt,Yt)
% LSKDSA: Landmark Selection based Kernelized Discriminant Subspace Alignment
% input:
% Xs: the source data
% Ys: the corresponding label set of Xs
% Xt: the target data
% output
% y; the prediction result of Xt

%% 1. Landmark selection
% the Nearest Neighbor Landmark Selection  
k = 1;
knnXs= KNN(Xs,Xt,k); 
knnXt= KNN(Xt,Xs,k); 
X = [knnXs,knnXt];

%% 2. Projection on landmarks
% setting kerenl parameers
options.KernelType = 'Gaussian';
dist = pdist(X');
t1 = mean(dist); 
options.t = t1;

% constructing kernel
Ks = constructKernel(X',Xs',options);
Kt = constructKernel(X',Xt',options);

% normalization 
Ks = Ks./repmat(sum(Ks,1),size(Ks,1),1);
Kt = Kt./repmat(sum(Kt,1),size(Kt,1),1);
Ks = zscore(Ks,0,2);
Kt = zscore(Kt,0,2);



%% 3. Aligning source and target subspaces
% PCA
[Xss,Xseigvalue] = PCA(Ks);
[Xtt,Xteigvalue] = PCA(Kt);

% preserve the principal components that account for at least for 95% of 
% the data variance based on the cumulative contribution rate of eigenvalue
ds = find(cumsum(Xseigvalue)/sum(Xseigvalue)>0.95);
dt = find(cumsum(Xteigvalue)/sum(Xteigvalue)>0.95);



% get projection directions
Ws = Xss(:,1:ds(5));
Wt = Xtt(:,1:dt(5));
% Ws = Xss(:,1:20);
% Wt = Xtt(:,1:20);


% discriminant constraint term
L = LDAReg(Ks,Ys,2);
% get M
lambda = 0.001; % default setting
M = (Ws'*Ws+lambda*Ws'*L*Ws)\(Ws'*Wt); 

% get projected source and target data
Ps = Ks'*(Ws*M);
Pt = Kt'*Wt;

%% LR classifier
str = '-s 0 -c 1 -B -1 -q';
model = train(Ys', sparse(Ps),str); % num * fec
[predict_label, acc, prob_estimates] = predict(Yt', sparse(Pt), model, '-b 1');
predict_label = predict_label';
y = prob_estimates(:,1)'; % prediction score


