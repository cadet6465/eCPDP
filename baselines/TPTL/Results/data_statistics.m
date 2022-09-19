addpath('../Weka');

%% load 42 Promise projects
addr = '../Promise/';
files = dir(addr);
Projects = cell(length(files)-2,1);
for i=3:length(files)
    name = [addr,files(i).name];
    Projects{i-2,1} = files(i).name;
    data = WekaArff2Data(name);
    Projects{i-2,2} = data;
    Projects{i-2,3} = size(data,1);
    Projects{i-2,4} = sum(data(:,end)==1);
end