clear;
clc;
close all;
x = 0.1:1/22:1;
target = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;
epoch = 5000;
eta = 0.05;

%% Hidden neurons: fixed centers and widths
c1 = 0.20;
r1 = 0.15;
c2 = 0.60;
r2 = 0.15;

%% Output weights and bias
w11_2 = rand(1);
w21_2 = rand(1);
b1_2 = 0;

%% Training
for j = 1:epoch
    for i = 1:length(x)
        % Gaussian hidden neurons
        y1_1 = exp(-(x(i)-c1)^2/(2*r1^2));
        y2_1 = exp(-(x(i)-c2)^2/(2*r2^2));

        % Linear output neuron
        v1_2 = y1_1*w11_2 + y2_1*w21_2 + b1_2;
        y1_2 = v1_2;

        % Error and output delta
        e = target(i) - y1_2;
        delta1_2 = e;

        % Update only the output layer
        w11_2 = w11_2 + eta*delta1_2*y1_1;
        w21_2 = w21_2 + eta*delta1_2*y2_1;
        b1_2 = b1_2 + eta*delta1_2;
    end
end

%% Training predictions with the final weights
y_mokymas = zeros(size(x));
for i = 1:length(x)
    y1_1 = exp(-(x(i)-c1)^2/(2*r1^2));
    y2_1 = exp(-(x(i)-c2)^2/(2*r2^2));
    y_mokymas(i) = y1_1*w11_2 + y2_1*w21_2 + b1_2;
end
figure;
plot(x, target, '-o');
hold on;
plot(x, y_mokymas, '-*');
legend('Target', 'Network');
title('Training');

%% Testing: no weight updates
x_naujas = 0.05:1/22:1;
target_naujas = (1 + 0.6*sin(2*pi*x_naujas/0.7) + 0.3*sin(2*pi*x_naujas))/2;
Y = zeros(size(x_naujas));
for i = 1:length(x_naujas)
    y1_1 = exp(-(x_naujas(i)-c1)^2/(2*r1^2));
    y2_1 = exp(-(x_naujas(i)-c2)^2/(2*r2^2));
    v1_2 = y1_1*w11_2 + y2_1*w21_2 + b1_2;
    y1_2 = v1_2;
    Y(i) = y1_2;
end
figure;
plot(x_naujas, target_naujas, '-o');
hold on;
plot(x_naujas, Y, '-*');
legend('Target', 'Network');
title('Testing');
fprintf('Test MSE: %.6f\n', mean((target_naujas-Y).^2));
