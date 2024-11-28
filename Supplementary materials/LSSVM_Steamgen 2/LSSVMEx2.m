%in the name of god
%LSSVM Alghoritm for system identification

clc;
clear;
close all

load ballbeam.dat
X1=ballbeam(:,1);
Y1=ballbeam(:,2);
gam = 10;
sig2 = 2;
type = 'function approximation';
[alpha,b] = trainlssvm({X1,Y1,type,gam,sig2,'RBF_kernel'});

