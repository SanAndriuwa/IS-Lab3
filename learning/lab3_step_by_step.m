% Step-by-step RBF example.
% One input and two Gaussian RBF neurons are calculated by hand.
clear; clc; close all;

x = 0.35;                % one input value
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;
c = [0.20; 0.60];        % RBF centers
r = [0.15; 0.15];        % RBF widths
w = [0.1; -0.1];         % output weights
b = 0;                   % output bias
eta = 0.05;              % learning rate

% 1. Distance from the input to each center.
distance = x - c;

% 2. Gaussian activation: close to a center means a large response.
f = exp(-distance.^2 ./ (2*r.^2));

% 3. Linear output and error.
y = w' * f + b;
e = d - y;

fprintf('RBF values: [%.4f %.4f]\n', f(1), f(2));
fprintf('Output y = %.4f, target d = %.4f, error e = %.4f\n', y, d, e);

% 4. With fixed centers and widths, update only the output layer.
w = w + eta*e*f;
b = b + eta*e;

fprintf('Updated weights: [%.4f %.4f], bias = %.4f\n', w(1), w(2), b);

% The main Lab3 file repeats this calculation for all x values and epochs.
