%%
% reading tab:
% b10e1     b50e1   b600e1 
% b10e5     b50e5   b600e5
% b10e20    b50e20  b600e20
%% plotting paper fig 2-1 (iid)
clear all;
close all;

b_idx = [10,50,600];
e_idx = [1, 5, 20];

figure(87);

tab_iid = NaN(1,9);
t = 1:1000;
i = 1;
for b = b_idx
    if b == 10
        color = [.85 0 0];
    elseif b == 50
        color = [.9 .7 0];
    else
        color = [0 .5 .9];
    end
    
    for e = e_idx
        filename = ['iid_paper/iid_b',num2str(b),'e',num2str(e),'.txt'];
        fileID = fopen(filename,'r');
        data = fscanf(fileID,'%f');
        data = data(1:1000)./100;
        
        data = sgolayfilt(data,1,11);
        
        % find first number >= 0.99 accuracy
        if isempty(find(data>=0.99, 1))
            tab_iid(i) = NaN;
        else
            tab_iid(i) = find(data>=0.99, 1);
        end
        i = i + 1;
        
        % plot results in different lines
        hold on;
        
        if e == 1
            plot(t, data, '-', 'Color', color, 'LineWidth', 1.5);
        elseif e == 5
            plot(t, data, '-.', 'Color', color, 'LineWidth', 1.5);
        else
            plot(t, data, '--', 'Color', color, 'LineWidth', 1.5);
        end
    end
end    
plot([-50,1000],[0.99, 0.99], 'Color', [.7 .7 .7]);
tab_iid = reshape(tab_iid, 3, 3)

xlabel('Communication Rounds');
ylabel('Test Accuracy');
title('MNIST CNN IID (paper reproduction)');
legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
legend('boxoff');
axis([-50 1000 0.70 1.02]);
box on;
set(gca,'LineWidth',2,'FontSize',15)
set(findall(gca, 'Type', 'Line'),'LineWidth',2);


%% plotting paper fig 2-2 (non-iid)
clear all;
close all;

b_idx = [10,50,600];
e_idx = [1, 5, 20];

figure(88);

tab_non = NaN(1,9);
t = 1:1000;
i = 1;
for b = b_idx
    if b == 10
        color = [.85 0 0];
    elseif b == 50
        color = [.9 .7 0];
    else
        color = [0 .5 .9];
    end
    
    for e = e_idx
        filename = ['non_iid_paper/non_b',num2str(b),'e',num2str(e),'.txt'];
        fileID = fopen(filename,'r');
        data = fscanf(fileID,'%f');
        data = data(1:1000)/100;
        
        data = sgolayfilt(data,1,27);
        
        % find first number >= 0.99 accuracy
        if isempty(find(data>=0.99, 1))
            tab_non(i) = NaN;
        else
            tab_non(i) = find(data>=0.99, 1);
        end
        i = i + 1;
        
        % plot results in different lines
        hold on;
        
        if e == 1
            plot(t, data, '-', 'Color', color, 'LineWidth', 1.5);
        elseif e == 5
            plot(t, data, '-.', 'Color', color, 'LineWidth', 1.5);
        else
            plot(t, data, '--', 'Color', color, 'LineWidth', 1.5);
        end
    end
end    
plot([-100,1000],[0.99,0.99], 'Color', [.7 .7 .7]);
tab_non = reshape(tab_non, 3, 3)

xlabel('Communication Rounds');
ylabel('Test Accuracy');
title('MNIST CNN non-IID (paper reproduction)');
% legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
% legend('boxoff');
axis([-50 1000 0.70 1.02]);
box on;
set(gca,'LineWidth',2,'FontSize',15)
set(findall(gca, 'Type', 'Line'),'LineWidth',2);

%% plot uneven data distribution result
clear all;
close all;

b_idx = [10,50,600];
e_idx = [1, 5, 20];

figure(50);

tab_ud = NaN(1,9);
t = 1:1000;
i = 1;

for b = b_idx
    if b == 10
        color = [.85 0 0];
    elseif b == 50
        color = [.9 .7 0];
    else
        color = [0 .5 .9];
    end
    
    for e = e_idx
        filename = ['uneven_dd_iid/un_b',num2str(b),'e',num2str(e),'.txt'];
        fileID = fopen(filename,'r');
        data = fscanf(fileID,'%f');
        data = data(1:1000)/100;
        
        %data = sgolayfilt(data,1,15);
        
        % find first number >= 0.99 accuracy
        if isempty(find(data>=0.99, 1))
            tab_ud(i) = NaN;
        else
            tab_ud(i) = find(data>=0.99, 1);
        end
        i = i + 1;
        
        % plot results in different lines
        hold on;
        if e == 1
            plot(t, data, '-', 'Color', color, 'LineWidth', 1.5);
        elseif e == 5
            plot(t, data, '-.', 'Color', color, 'LineWidth', 1.5);
        else
            plot(t, data, '--', 'Color', color, 'LineWidth', 1.5);
        end
    end
end
plot([-100,1000], [0.99, 0.99], 'Color', [.7 .7 .7]);

tab_ud = reshape(tab_ud, 3, 3)

xlabel('Communication Rounds');
ylabel('Test Accuracy');

title({'MNIST CNN IID (uneven D.D)';'w/o weight distribution from clients'});
% legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
% legend('boxoff');
axis([-50 1000 0.70 1.02]);
box on;
set(gca,'LineWidth',2,'FontSize',15)
set(findall(gca, 'Type', 'Line'),'LineWidth',2);


%% Plot Stefan

clear all;
close all;
b_idx = [10, 50, 600];
e_idx = [1, 5, 20];

figure(51);

tab_sh = NaN(1,9);
t = 1:1000;
i = 1;

for b = b_idx
    if b == 10
        color = [.85 0 0];
    elseif b == 50
        color = [.9 .7 0];
    else
        color = [0 .5 .9];
    end
    
    for e = e_idx
        filename = ['uneven_with_weight/sh_b',num2str(b),'e',num2str(e),'.txt'];
        fileID = fopen(filename,'r');
        data = fscanf(fileID,'%f');
        data = data./100;
        
%         apply Savitzky-Golay filtering
        data = sgolayfilt(data,1,11);
        
        
        % find first number >= 0.99 accuracy
        if isempty(find(data>=0.99, 1))
            tab_sh(i) = NaN;
        else
            tab_sh(i) = find(data>=0.99, 1);
        end
        i = i + 1;
        
        % plot results in different lines
        hold on;
        if e == 1
            plot(t, data, '-', 'Color', color, 'LineWidth', 1.5);
        elseif e == 5
            plot(t, data, '-.', 'Color', color, 'LineWidth', 1.5);
        else
            plot(t, data, '--', 'Color', color, 'LineWidth', 1.5);
        end
    end
end    
plot([-100,100], [0.99,0.99], 'Color', [.7 .7 .7]);

tab_sh = reshape(tab_sh, 3, 3)

xlabel('Communication Rounds');
ylabel('Test Accuracy');
title({'MNIST CNN IID (uneven D.D.)';'w/ weight distribution from clients'});
% legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
% legend('boxoff');
axis([-50 1000 0.70 1.02]);
box on;
set(gca,'LineWidth',2,'FontSize',15)
set(findall(gca, 'Type', 'Line'),'LineWidth',2);

%% Plot Mirza

clear all;
close all;
b_idx = [10];
e_idx = [202, 203];

figure(51);

tab_mm = NaN(1,9);
t = 1:1000;
i = 1;

for b = b_idx
    if b == 10
        color = [.85 0 0];
    elseif b == 50
        color = [.9 .7 0];
    else
        color = [0 .5 .9];
    end
    
    for e = e_idx
        filename = ['noise/mm_b',num2str(b),'e',num2str(e),'.txt'];
        fileID = fopen(filename,'r');
        data = fscanf(fileID,'%f');
        data = data./100;
        
%         apply Savitzky-Golay filtering
         data = sgolayfilt(data,1,9);
        
        
        % find first number >= 0.99 accuracy
        if isempty(find(data>=0.99, 1))
            tab_mm(i) = NaN;
        else
            tab_mm(i) = find(data>=0.99, 1);
        end
        i = i + 1;
        
        % plot results in different lines
        hold on;
        if e == 202
            plot(t, data, '-', 'Color', [.9 .7 0], 'LineWidth', 1.5);
        elseif e == 203
            plot(t, data, '-', 'Color', [.85 0 0], 'LineWidth', 1.5);
        else
            plot(t, data, '--', 'Color', color, 'LineWidth', 1.5);
        end
    end
end    
plot([-100,1000], [0.99,0.99], 'Color', [.7 .7 .7]);

tab_mm = reshape(tab_mm, 3, 3)

xlabel('Communication Rounds');
ylabel('Test Accuracy');
title({'MNIST CNN IID';'w/ Gaussian noise on updates'});
% legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
% legend('boxoff');
axis([-50 1000 0.0 1.02]);
box on;
set(gca,'LineWidth',2,'FontSize',15)
set(findall(gca, 'Type', 'Line'),'LineWidth',2);