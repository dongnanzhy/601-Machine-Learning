clear;

point = load('HW8_Problem6.txt');
N = length(point);
total = sum(point(:));
miu1 = total /N;
miu2 = (total + 1) /(N + 1);
miu3 = (1 + 4*total)/(4*N + 1);
miu4 = total /(N + 1);