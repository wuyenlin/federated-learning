% clear all
% load('result.mat');
t = 1:1000;

figure(1);
plot([0 1000], [0.99 0.99], 'Color', [0 .5 .2], 'LineWidth', 2);
hold on;
plot(t, b10e1, '-', 'Color', [.85 0 0], 'LineWidth', 1.5);
plot(t, b10e5, '-.', 'Color', [.85 0 0], 'LineWidth', 1.5);
plot(t, b10e20, '--', 'Color', [.85 0 0], 'LineWidth', 1.5);

plot(t, b50e1, '-', 'Color', [.9 .7 0], 'LineWidth', 1.5);
plot(t, b50e5, '-.', 'Color', [.9 .7 0], 'LineWidth', 1.5);
plot(t, b50e20, '--', 'Color', [.9 .7 0], 'LineWidth', 1.5);

plot(t, b600e1, '-', 'Color', [0 .5 .9], 'LineWidth', 1.5);
plot(t, b600e5, '-.', 'Color', [0 .5 .9], 'LineWidth', 1.5);
plot(t, b600e20, '--', 'Color', [0 .5 .9], 'LineWidth', 1.5);
hold off;

xlabel('Communication Rounds');
ylabel('Test Accuracy (%)');
title('MNIST CNN IID');
legend({'B=10 E=1','B=10 E=5','B=10 E=20','B=50 E=1','B=50 E=5','B=50 E=20','B=\infty E=1','B=\infty E=5','B=\infty E=20'},'Location','southeast');
legend('boxoff');

axis([0 1000 0.9 1]);
