% Part d
clc
clear all
load X_trainlc
load X_testlc
X_train=X_trainlc;
X_test=X_testlc;
[n Q C]=size(X_train);
[nn,Q_test,mm]=size(X_test);
Xmin_train=min(min(X_train,[],2),[],3);
Xmax_train=max(max(X_train,[],2),[],3);
Xmin_test=min(min(X_test,[],2),[],3);
Xmax_test=max(max(X_test,[],2),[],3);
Xtrain=2*(X_train-repmat(Xmin_train,[1,Q,C]))./(repmat(Xmax_train,[1,Q,C])-repmat(Xmin_train,[1,Q,C]))-1;
Xtest=2*(X_test-repmat(Xmin_test,[1,Q_test,C]))./(repmat(Xmax_test,[1,Q_test,C])-repmat(Xmin_test,[1,Q_test,C]))-1;
clear X_train X_test
Epoch=100;
M=120;
K=M/C;
sigma=1;
CCR_train=zeros(1,Epoch);
CCR_test=zeros(1,Epoch);
J_train=zeros(1,Epoch);
J_test=zeros(1,Epoch);
U=(rand(M,C)-0.5);
Xs=zeros(n,M);
IDX_train=zeros(Q,C);
C_train=zeros(K,n,C);
for c = 1:C
    [IDX_train(:,c),C_train(:,:,c)] = kmeans(Xtrain(:,:,c)',K);
    Xs(:,(c-1)*K+1:c*K)=C_train(:,:,c)';
end
tic
for ep=1:Epoch
    r=0.05;
    U_new=RBF(U,r,Xtrain,Xs,sigma);
    U=U_new;
    [CCR_train(ep) Confusion_train J_train(ep)]=ClassifyData(U,Xtrain,Xs,sigma);
    [CCR_test(ep) Confusion_test J_test(ep)]=ClassifyData(U,Xtest,Xs,sigma);
end
time=toc;
% Plot CCR
ep=1:Epoch;
figure;
plot(ep,CCR_train,'b',ep,CCR_test,'r');xlabel('Epoch');ylabel('CCR');title('CCR for Train and Test Data');
grid on;
% Plot J
ep=1:Epoch;
figure;
plot(ep,J_train,'b',ep,J_test,'r');xlabel('Epoch');ylabel('Cost Function (J)');title('Cost Function for Train and Test Data');
grid on;