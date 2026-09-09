% RBF approximation: train weights, bias, centers and widths.
clear; clc; close all;
x = 0.1:1/22:1;
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;
c = [0.20; 0.60];
r = [0.15; 0.15];
w = [0.1; -0.1];
b = 0;
eta = 0.05;
etaShape = 0.005;
epochs = 5000;
loss = zeros(1, epochs);
history = zeros(4, epochs);
for epoch = 1:epochs
    for i = 1:length(x)
        distance = x(i)-c;
        f = exp(-distance.^2 ./ (2*r.^2));
        y = w'*f + b;
        e = d(i)-y;
        % All derivatives use the same OLD parameters.
        dc = e*w.*f.*distance ./ r.^2;
        dr = e*w.*f.*distance.^2 ./ r.^3;
        w = w + eta*e*f;
        b = b + eta*e;
        c = c + etaShape*dc;
        r = r + etaShape*dr;
        % Keep widths positive and prevent extreme center movement.
        c = min(max(c,0),1);
        r = min(max(r,0.03),1);
    end
    F = exp(-(x-c).^2 ./ (2*r.^2));
    loss(epoch) = mean((d-(w'*F+b)).^2);
    history(:,epoch) = [c;r];
end
fprintf('Training MSE: %.6f\n', loss(end));
disp(table(c,r,w,'VariableNames',{'Center','Width','Weight'}));
fprintf('Output bias: %.6f\n', b);
xx = linspace(0.1,1,201);
dd = (1 + 0.6*sin(2*pi*xx/0.7) + 0.3*sin(2*pi*xx))/2;
yy = w'*exp(-(xx-c).^2 ./ (2*r.^2)) + b;
fprintf('Dense-grid MSE: %.6f\n', mean((dd-yy).^2));
figure; plot(xx,dd,'b-',xx,yy,'r--',x,d,'ko'); grid on;
xlabel('x'); ylabel('Output'); legend('Target','Adaptive RBF','Training points');
figure; semilogy(loss); grid on; xlabel('Epoch'); ylabel('Training MSE');
figure; plot(history'); grid on; xlabel('Epoch'); ylabel('Parameter');
legend('c1','c2','r1','r2'); title('Centers and widths during training');

