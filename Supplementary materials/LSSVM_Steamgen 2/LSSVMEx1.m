%in the name of god
%LSSVM Alghoritm for system identification

clc;
clear;
close all

load ballbeam.dat
X1=ballbeam(:,1);
Y1=ballbeam(:,2);
type = 'function approximation';
kernel = 'RBF_kernel';
gam =0.2;
sig2 = 10;
%A model is de?ned
model = initlssvm(X1,Y1,type,gam,sig2,kernel);
model = trainlssvm(model);
plotlssvm(model);
[alpha,b] = trainlssvm({X1,Y1,type,gam,sig2,'RBF_kernel','original'});
Ytest = simlssvm({X1,Y1,type,gam,sig2,'RBF_kernel'},{alpha,b},Y1);
plot(1:1000,alpha,'r',1:1000,Y1,'b')


