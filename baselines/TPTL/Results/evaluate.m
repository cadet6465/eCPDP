%% load and combine results of 7 models
load('results_tca_rnd.mat');
load('results_tca_all.mat');
load('results_tds.mat');
load('results_lt.mat');
load('results_dycom.mat');
load('results_tptl.mat');
 
f1  = [tca_rnd_mean(:,1),tca_all(:,1),tds(:,1),lt_c45(:,1),dycom_mean(:,1),tptl(:,1)];
p20 = [tca_rnd_mean(:,2),tca_all(:,2),tds(:,2),lt_c45(:,2),dycom_mean(:,2),tptl(:,2)];
std = [tca_rnd_std,dycom_std];

mean(f1)    
mean(p20)   
mean(std)  

%% Statistics of F1-score
f1 = roundn(f1,-3);
for i=1:size(f1,2)-1
    t = f1(:,end);                              % The proposed model
    s = f1(:,i);                                % A baseline model
    stat_f1(i,1) = (mean(t)-mean(s))/mean(s);   % Improved%
    stat_f1(i,2) = sum(t<s);                    % W
    stat_f1(i,3) = sum(t==s);                   % T
    stat_f1(i,4) = sum(t>s);                    % L
    stat_f1(i,5) = ranksum(t,s);                % Wilcoxon signed-rank test
    stat_f1(i,6) = CliffDelta(t,s);             % Cliff's delta
end
stat_f1(:,5)  = bonf_holm(stat_f1(:,5));        % Bonferroni correction

%% Statistics of PofB20
p20 = roundn(p20,-3);
for i=1:size(f1,2)-1
    t = p20(:,end);                             % The proposed model
    s = p20(:,i);                               % A baseline model
    stat_p20(i,1) = (mean(t)-mean(s))/mean(s);  % Improved%
    stat_p20(i,2) = sum(t<s);                   % W
    stat_p20(i,3) = sum(t==s);                  % T
    stat_p20(i,4) = sum(t>s);                   % L
    stat_p20(i,5) = ranksum(t,s);               % Wilcoxon signed-rank test
    stat_p20(i,6) = CliffDelta(t,s);            % Cliff's delta
end
stat_p20(:,5) = bonf_holm(stat_p20(:,5));       % Bonferroni correction