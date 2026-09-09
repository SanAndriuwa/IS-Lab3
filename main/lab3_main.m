% RBF approximation: fixed centers and widths, trained output layer.
clear; clc; close all;
x = 0.1:1/22:1; % Exactly 20 points; the last is about 0.9636.
% The complete sum is divided by 2 (the assignment has unmatched brackets).
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;
c = [0.20; 0.60];
r = [0.15; 0.15];
w = [0.1; -0.1];
b = 0;
eta = 0.05;
epochs = 5000;
loss = zeros(1, epochs);
F = exp(-(x-c).^2 ./ (2*r.^2));
for epoch = 1:epochs
    for i = 1:length(x)
        f = F(:, i);
        y = w'*f + b; % Linear output, no sign function.
        e = d(i) - y;
        w = w + eta*e*f;
        b = b + eta*e;
    end
    loss(epoch) = mean((d - (w'*F+b)).^2);
end
fprintf('Training MSE: %.6f\n', loss(end));
disp(table(c,r,w,'VariableNames',{'Center','Width','Weight'}));
fprintf('Output bias: %.6f\n', b);
xx = linspace(0.1, 1, 201);
dd = (1 + 0.6*sin(2*pi*xx/0.7) + 0.3*sin(2*pi*xx))/2;
yy = w'*exp(-(xx-c).^2 ./ (2*r.^2)) + b;
fprintf('Dense-grid MSE: %.6f\n', mean((dd-yy).^2));
figure; plot(xx,dd,'b-',xx,yy,'r--',x,d,'ko'); grid on;
xlabel('x'); ylabel('Output'); legend('Target','RBF','Training points');
title('Two RBF neurons: fixed centers and widths');
figure; semilogy(loss); grid on; xlabel('Epoch'); ylabel('Training MSE');

