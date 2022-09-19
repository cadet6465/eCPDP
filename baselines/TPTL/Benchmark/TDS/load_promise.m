addpath('../../Liblinear','../../Weka')

%% load 42 Promise projects
addr = '../../Promise/';
files = dir(addr);
Projects = cell(length(files)-2,1);
for i=3:length(files)
    name = [addr,files(i).name];
    Projects{i-2,1} = files(i).name;
    Projects{i-2,2} = WekaArff2Data(name);
end

%% all combinations of cross-project predictions
load('project_id.mat'); 
k = 1;
CrossProjects = cell(42,1);
for i=1:length(Projects)
    CrossProjects{i,1}.test.data = Projects{i,2};
    CrossProjects{i,1}.test.line = Projects{i,2}(:,11);
    t = Projects(project_id(:,1)~=project_id(i,1),1:2);
    for j=1:size(t,1);
        CrossProjects{i,1}.train{j,1}.data = t{j,2};
        CrossProjects{i,1}.train{j,1}.line = t{j,2}(:,11);
    end
end


    

