clear all;
E = [1,5,20];

for e = E
    filename = ['e',num2str(e),'.txt'];
    fileID = fopen(filename,'r');
    data = fscanf(fileID,'%f');
    
    
    name1 = ['ud_b10e',num2str(e),'.txt'];
    dlmwrite(name1,data(1:100));
    
    name2 = ['ud_b50e',num2str(e),'.txt'];
    dlmwrite(name2,data(101:200));

    name3 = ['ud_b600e',num2str(e),'.txt'];
    dlmwrite(name3,data(201:300));
end