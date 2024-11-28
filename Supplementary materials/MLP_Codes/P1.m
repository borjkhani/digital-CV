% #6
% Part 1
clc
clear all
load X_trainlc
load X_testlc
X_train=X_trainlc;
X_test=X_testlc;
[n Q C]=size(X_train);
[nn Q_test mm]=size(X_test);
M=n;
Xmin_train=min(min(X_train,[],2),[],3);
Xmax_train=max(max(X_train,[],2),[],3);
Xmin_test=min(min(X_test,[],2),[],3);
Xmax_test=max(max(X_test,[],2),[],3);
Xtrain=2*(X_train-repmat(Xmin_train,[1,Q,C]))./(repmat(Xmax_train,[1,Q,C])-repmat(Xmin_train,[1,Q,C]))-1;
Xtest=2*(X_test-repmat(Xmin_test,[1,Q_test,C]))./(repmat(Xmax_test,[1,Q_test,C])-repmat(Xmin_test,[1,Q_test,C]))-1;
clear X_train X_test
X_train=zeros(n+1,Q,C);
X_test=zeros(n+1,Q_test,C);
for c=1:C
    X_train(:,:,c)=[Xtrain(:,:,c);ones(1,Q)];
    X_test(:,:,c)=[Xtest(:,:,c);ones(1,Q_test)];
end
W=(rand(n+1,M)-0.5);
U=(rand(M+1,C)-0.5);
Epoch=50;
CCR_train=zeros(1,Epoch);
CCR_test=zeros(1,Epoch);
J_train=zeros(1,Epoch);
J_test=zeros(1,Epoch);
tic
for ep=1:Epoch
    r=0.01/log(ep+1);
    [W_new,U_new]=BackPropagation(W,U,r,X_train);
    W=W_new;
    U=U_new;
    [CCR_train(ep) Confusion_train J_train(ep)]=ClassifyData(W,U,X_train);
    [CCR_test(ep) Confusion_test J_test(ep)]=ClassifyData(W,U,X_test);
end
time=toc;
% save Part_a CCR_test CCR_train Epoch Confusion_test Confusion_train J_test J_train time
% Plot CCR
ep=1:Epoch;
figure;
plot(ep,CCR_train,'b',ep,CCR_test,'r');xlabel('Epoch');ylabel('CCR');title('CCR for Train and Test Data');
% Plot J
ep=1:Epoch;
figure;
plot(ep,J_train,'b',ep,J_test,'r');xlabel('Epoch');ylabel('Cost Function (J)');title('Cost Function for Train and Test Data');