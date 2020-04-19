clear all;
E = [1,5,20];

e=1;
%filename = ['iid_e5.txt'];
filename = ['e1.txt'];

fileID = fopen(filename,'r');
data = fscanf(fileID,'%f');


name1 = ['iid_b10e',num2str(e),'.txt'];
%name1 = ['non_iid_b10e',num2str(e),'.txt'];
dlmwrite(name1,data(1:200));

name2 = ['iid_b50e',num2str(e),'.txt'];
%name2 = ['non_iid_b50e',num2str(e),'.txt'];
dlmwrite(name2,data(501:700));

name3 = ['iid_b600e',num2str(e),'.txt'];
%name3 = ['non_iid_b600e',num2str(e),'.txt'];
dlmwrite(name3,data(1001:1200));
