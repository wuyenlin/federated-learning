%%
clear all;

%% plotting paper fig 2-1 (iid)
b_idx = [10,50,600];
e_idx = [1, 5, 20];

figure(87);

t = 1:1000;

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


xlabel('Communication Rounds');
ylabel('Test Accuracy (%)');
title('MNIST CNN IID');
legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
legend('boxoff');
axis([0 1000 90 100]);


%% plotting paper fig 2-2 (non-iid)
b_idx = [10,50,600];
e_idx = [1, 5];
%e_idx = [1, 5, 200];

figure(88);

t = 1:1000;

for b = b_idx
    if b == 10
        color = [.85 0 0];
    elseif b == 50
        color = [.9 .7 0];
    else
        color = [0 .5 .9];
    end
    
    for e = e_idx
        filename = ['non_iid\b',num2str(b),'e',num2str(e),'.txt'];
        fileID = fopen(filename,'r');
        data = fscanf(fileID,'%f');
        data = repelem(data,5);
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


xlabel('Communication Rounds');
ylabel('Test Accuracy (%)');
title('MNIST CNN IID');
%legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
%legend('boxoff');

%% plot result

b_idx = [10,50,600];
e_idx = [1, 5, 20];

figure(50);

t = 1:1000;

for b = b_idx
    if b == 10
        color = [.85 0 0];
    elseif b == 50
        color = [.9 .7 0];
    else
        color = [0 .5 .9];
    end
    
    for e = e_idx
        filename = ['mnist_cnn_iid_result\ud_b',num2str(b),'e',num2str(e),'.txt'];
        fileID = fopen(filename,'r');
        data = fscanf(fileID,'%f');
        data = repelem(data,10);
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


xlabel('Communication Rounds');
ylabel('Test Accuracy (%)');
title('MNIST CNN IID');
legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
legend('boxoff');
axis([0 1000 90 100]);