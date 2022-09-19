jeriko = importdata("output\LSKDSA_EXPRESULT_jeriko.mat");
aeeem = importdata("output\LSKDSA_EXPRESULT_aeeem.mat") ;
jira = importdata("output\LSKDSA_EXPRESULT_jira.mat");

aggresult = [];
objname = [];
for i = 1:size(jeriko,1)
    temp = jeriko{i,2};
    aggresult = cat(1,aggresult,temp);
end

