%%%Data
close all
clc;
clear;

load steamgen.dat

depthu=40;
depthy=40;

dataSet1 = steamgen(1:9600, 2:9)';
r = size(dataSet1, 1);
N = size(dataSet1, 2);
N_trn = 0.75*N;
N_tst = 0.25*N;

dataSetu = [zeros(r/2, depthu-1), dataSet1(1:r/2, :)];
dataSety = [zeros(r/2, depthy), dataSet1(r/2+1:r, :)];

U = dataSetu(1, 1:N);
for i = 2:depthu
    U = [U; dataSetu(1, i:N+i-1)];   
end

Y = dataSety(1, 1:N);
for i = 2:depthy
    Y = [Y; dataSety(1, i:N+i-1)];
end
p = [U; Y];
t = dataSety(1, depthy+1:depthy+N);

% ========================================================================
% normalizing data
% ========================================================================
[pn, minp, maxp] = premnmx(p);
[tn, mint, maxt] = premnmx(t);
range{1}=minp; 
range{2}=maxp;
range{3}=mint;
range{4}=maxt;
% ========================================================================

p_trn = pn(:, 1:N_trn);
t_trn = tn(:, 1:N_trn);
p_tst = pn(:, 1+N_trn:N_trn+N_tst);
t_tst = tn(:, 1+N_trn:N_trn+N_tst);