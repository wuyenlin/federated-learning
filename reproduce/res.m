%% plotting paper fig 2-1 (iid)
clear all;
b_idx = [10,50,600];
e_idx = [1, 5, 20];

figure(87);

tab_iid = zeros(1,9);
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
        filename = ['iid\b',num2str(b),'e',num2str(e),'.txt'];
        fileID = fopen(filename,'r');
        data = fscanf(fileID,'%f');
        data = repelem(data,5);
        
        % find first number >= 0.99 accuracy
        if isempty(find(data>=99, 1))
            tab_iid(i) = NaN;
        else
            tab_iid(i) = find(data>=99, 1);
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
tab_iid = reshape(tab_iid, 3,3)

xlabel('Communication Rounds');
ylabel('Test Accuracy (%)');
title('MNIST CNN IID (paper reproduction)');
legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
legend('boxoff');
axis([0 1000 80 100]);


%% plotting paper fig 2-2 (non-iid)
b_idx = [10,50,600];
e_idx = [1, 5, 20];

figure(88);

tab_non = zeros(1,9);
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
        filename = ['non_iid\non_b',num2str(b),'e',num2str(e),'.txt'];
        fileID = fopen(filename,'r');
        data = fscanf(fileID,'%f');
        data = repelem(data,5);
        
        % find first number >= 0.99 accuracy
        if isempty(find(data>=99, 1))
            tab_non(i) = NaN;
        else
            tab_non(i) = find(data>=99, 1);
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
tab_non = reshape(tab_non, 3,3)

xlabel('Communication Rounds');
ylabel('Test Accuracy (%)');
title('MNIST CNN non-IID (paper reproduction)');
legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
legend('boxoff');

%% plot uneven data distribution result
%clear all;
b_idx = [10,50,600];
e_idx = [1, 5, 20];

figure(50);

tab_ud = zeros(1,9);
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
        filename = ['uneven_mnist_cnn_iid\ud_b',num2str(b),'e',num2str(e),'.txt'];
        fileID = fopen(filename,'r');
        data = fscanf(fileID,'%f');
        data = repelem(data,10);
        
        % find first number >= 0.99 accuracy
        if isempty(find(data>=99, 1))
            tab_ud(i) = NaN;
        else
            tab_ud(i) = find(data>=99, 1);
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

tab_ud = reshape(tab_ud, 3,3)

xlabel('Communication Rounds');
ylabel('Test Accuracy (%)');
title('MNIST CNN IID (uneven data distribution)');
legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
legend('boxoff');
axis([0 1000 80 100]);

