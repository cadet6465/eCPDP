addpath('../../Weka','../../TCA+');

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
CrossProjects = cell(42,3);
for i=1:length(Projects)
    target_id = project_id(i,1);
    target_project = Projects{i,2};
    source_id = find(project_id(:,1)~=target_id(1));
    source_projects = Projects(project_id(:,1)~=target_id(1),:);
    CrossProjects{i,1} = source_projects(:,2);    
    CrossProjects{i,2} = target_project;          
    CrossProjects{i,3} = target_project(:,11);  
end


    

