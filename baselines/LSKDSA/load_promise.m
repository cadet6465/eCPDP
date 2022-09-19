%% load 42 Promise projects
addr = './dataset/aeeem/';
files = dir(addr);
Projects = cell(length(files)-2,1);
for i=3:length(files)
    name = [addr,files(i).name];
    Projects{i-2,1} = files(i).name;
    Projects{i-2,2} = readmatrix(name);
end

%% all combinations of cross-project predictionss
% load('project_id.mat'); 
% k = 1;
% CrossProjects = cell(length(Projects)*(length(Projects)-1),7);
% for i=1:length(Projects)
%     target_id = project_id(i,1);
%     target_project = Projects{i,2};
%     source_id = find(project_id(:,1)~=target_id(1));
%     source_projects = Projects(project_id(:,1)~=target_id(1),:);
%     for j=1:size(source_projects,1)
%         CrossProjects{k,1} = i;                       
%         CrossProjects{k,2} = source_id(j);          
%         CrossProjects{k,3} = source_projects{j,2};     
%         CrossProjects{k,4} = target_project;         
%         CrossProjects{k,5} = target_project(:,11);     
%         [tca_src,tca_tar]  = standard(CrossProjects{k,3},CrossProjects{k,4});
%         CrossProjects{k,6} = tca_src;
%         CrossProjects{k,7} = tca_tar;
%         k = k + 1;
%     end
% end


    

