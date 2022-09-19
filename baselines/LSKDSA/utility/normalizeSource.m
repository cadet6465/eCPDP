function [X,Y] = normalizeSource(X,Y,ratio)

posind = find(Y == 1); % defective
negind = find(Y == 0); % non-defectuve
temp1 = X(posind,:);
temp2 = X(negind,:);
X = cat(1,temp1,temp2);

% split training index
trIdxPos = 1:floor(ratio*length(posind));
trIdxNeg = length(posind)+1:length(posind)+floor(ratio*length(negind));

% trIdxPos = randperm(length(posind),floor(ratio*length(posind)));
% trIdxNeg = randperm(length(negind),floor(ratio*length(posind)));
% trIdxNeg = (length(posind)+1) + trIdxNeg;
% trIdxNeg

trIdx = [trIdxPos,trIdxNeg];

X = X(trIdx,:);
Y = Y(trIdx,:);




