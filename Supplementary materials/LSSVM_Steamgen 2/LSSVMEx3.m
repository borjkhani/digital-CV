clc;
clear;
close all

%insert sig2
gam =3;
sig2 = 40;

% A simple example shows how to start using the toolbox for a classi?cation task. We start with
% constructing a simple example dataset according to the correct formatting. Data are represented
% as matrices where each row of the matrix contains one datapoint:

load ballbeam.dat
X1=ballbeam(1:700,1);
Y1=ballbeam(1:700,2);
Xt=ballbeam(701:1000,1);
Y2=ballbeam(701:1000,2);
type = 'function estimation';
kernel = 'RBF_kernel';
sig3=sig2-35;
%A model is de?ned
model = initlssvm(X1,Y1,type,gam,sig2,kernel);
model = trainlssvm(model);
plotlssvm(model);
[alpha,b] = trainlssvm({X1,Y1,type,gam,sig2,'RBF_kernel','original'});
Ytrn=(alpha/gam+b);
[Ytest,b2] = trainlssvm({Xt,Y2,type,gam,sig3,'RBF_kernel','original'});
plot(1:700,alpha/gam+b,'r',1:700,Y1,'b')
plot(1:300,Y2,'r',1:300,Ytest/gam+b2,'b')

Y_tst=Ytest/gam+b2;

e=(Y1-Ytrn).^2;
MSE=sum(e)/1000

e=(Y2-Y_tst).^2;
MSE_tst=sum(e)/300

