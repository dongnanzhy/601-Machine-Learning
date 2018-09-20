clear;

point = load('HW8_Problem3.txt');
N = length(point);
T = log(point)/log(exp(1));
s = sum(T(:));
rst = N / (0.5-s) - 1;
%beta = -1*N/s - 1;
% b = 1-s;
% c = -1*(N + s);
% beta = (-1 *b + sqrt(b^2 - 4*c))/2;

% PA = 0.003*0.997;
% PB = 0.997*0.015;
% prior = PA /(PA + PB);
% PA2 = prior*0.997;
% PB2 = (1-prior)*0.015;
% rst = PB2 /(PA2 + PB2);

