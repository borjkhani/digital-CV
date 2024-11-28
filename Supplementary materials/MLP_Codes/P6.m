% #6
tic
clc
clear all
close all
load X_trainlc
load X_testlc
%-------set parameter-----------------
X_train=X_trainlc;
X_test=X_testlc;
final=50;
neuron_layer1=5;
neuron_layer2=8;
a=size(X_train);
feature_size=a(1,1);
sample_size_train=a(1,2);
class_lable=a(1,3);
clear a
a=size(X_test);
sample_size_test=a(1,2);
clear a

X_trainlc = zeros(feature_size+1,sample_size_train,class_lable);
X_testlc =  zeros(feature_size+1,sample_size_test ,class_lable);
for i = 1:class_lable
    A = 3*X_train(14,:,i)+2*X_train(22,:,i)-2*X_train(36,:,i);
    X_trainlc(:,:,i) = cat(1,X_train(:,:,i),A);
    B=3*X_test(14,:,i)+2*X_test(22,:,i)-2*X_test(36,:,i);
    X_testlc(:,:,i) = cat(1,X_test(:,:,i),B);
    
    ss=sample_size_train*(i-1)+1:sample_size_train*(i);
    Y_n(:,ss)=X_trainlc(:,:,i);
end
clear A B i ss

%------------------------------------PCA---------------------------
Mu_T=mean(Y_n(:,:),2);
N=class_lable*sample_size_train;
a=0;Cov_x=0;
for q=1:N
    a=(Y_n(:,q)-Mu_T)*(Y_n(:,q)-Mu_T)';
    Cov_x=Cov_x+a;
end
Cov_x=Cov_x/(N-1);
clear q a N Y_n
[A B C]=svd(Cov_x);
bb=diag(B);
cc=find(bb <10^-12);
feature_selected=1:cc(1,1)-1;
clear cc bb

a=size(feature_selected);
feature_size_sel=a(1,2);
clear a

D=A(:,feature_selected)'*Cov_x*A(:,feature_selected);
D_inv=D^(-0.5);

X_W=zeros(feature_size_sel,sample_size_train,class_lable);
Y_W=zeros(feature_size_sel,sample_size_test,class_lable);
for i=1:class_lable
    for k=1:sample_size_train      
        X_W(:,k,i)=D_inv*A(:,feature_selected)'*(X_trainlc(:,k,i)-Mu_T);
    end
end
for i=1:class_lable
    for k=1:sample_size_test        
        Y_W(:,k,i)=D_inv*A(:,feature_selected)'*(X_testlc(:,k,i)-Mu_T);
    end
end

% SW and SB
SW=zeros(feature_size_sel,feature_size_sel);
SB=zeros(feature_size_sel,feature_size_sel);
for i=1:class_lable
    Mu_I=0;
    Mu_I=mean(X_W(:,:,i),2);
    SI=zeros(feature_size_sel,feature_size_sel);
    for k=1:sample_size_train
        SI=SI+(X_W(:,k,i)-Mu_I)*(X_W(:,k,i)-Mu_I)';
    end
    SW=SW+SI;
    SB=SB+(Mu_I-Mu_T(feature_selected,1))*(Mu_I-Mu_T(feature_selected,1))';   
end
clear i k Mu_I 
clear A B C
Sep_matrix=SW\SB;
[A B C]=svd(Sep_matrix);
bb=diag(B);
count=feature_size_sel+1;
SJ0=sum(bb);
SJ=1;
while( SJ>0.95 && count ~=1)
    count=count-1;
    SJ=(sum(bb(1:count,1)))./SJ0;
end
count=count+1;
SJ=(sum(bb(1:count,1)))./SJ0;
count=7;
X_W_R=zeros(count,sample_size_train,class_lable);
for i=1:class_lable
    for k=1:sample_size_train
        X_W_R(:,k,i)=A(:,1:count)'*X_W(:,k,i);
    end
end
Y_W_R=zeros(count,sample_size_test,class_lable);
for i=1:class_lable
    for k=1:sample_size_test
        Y_W_R(:,k,i)=A(:,1:count)'*Y_W(:,k,i);
    end
end
clear a
X_n=X_W_R;
a=size(X_n);
feature_size=a(1,1);
sample_size_train=a(1,2);
class_lable=a(1,3);
Y_n=Y_W_R;
clear a

for f=1:feature_size
    X_min(f,1)=min(min(X_n(f,:,:)));
    X_max(f,1)=max(max(X_n(f,:,:)));
    X_nn(f,:,:)=2*(X_n(f,:,:)-X_min(f,1))/((X_max(f,1)-X_min(f,1)))-1;
end

for f=1:feature_size
    Y_min(f,1)=min(min(Y_n(f,:,:)));
    Y_max(f,1)=max(max(Y_n(f,:,:)));
    Y_nn(f,:,:) =2*(Y_n(f,:,:)-Y_min(f,1))/((Y_max(f,1)-Y_min(f,1)))-1;
end
clear Y_min Y_max X_min X_max


for i=1:class_lable
    for k=1:sample_size_test
        vector_test(:,k,i)=[Y_nn(:,k,i);1];
    end
end
for i=1:class_lable
    for k=1:sample_size_train
        vector_train(:,k,i)=[X_nn(:,k,i);1];
    end
end

W=rand(feature_size+1,neuron_layer1)-0.5;
Q=rand(neuron_layer1+1,neuron_layer2)-0.5;
U=rand(neuron_layer2+1,class_lable)-0.5;



confusion_matrix_test=zeros(class_lable,class_lable,final);
confusion_matrix_train=zeros(class_lable,class_lable,final);
J_error_test=zeros(final,1);
J_error_train=zeros(final,1);
RO_initial=0.0005;
n_train=sample_size_train*class_lable;
for epoch=1:final
    epoch
    RO=RO_initial./(log(epoch)+1);
    diff_W=zeros(feature_size+1,neuron_layer1);
    diff_Q=zeros(neuron_layer1+1,neuron_layer2);
    diff_U=zeros(neuron_layer2+1,class_lable);
    index_random=randperm(n_train);
    
    for sample_data=1:n_train
        re_index(1,sample_data)=floor((index_random(sample_data)-1)/sample_size_train)+1;
        re_index(2,sample_data)=index_random(sample_data)-(re_index(1,sample_data)-1)*sample_size_train;
    end
    for sample_data=1:n_train
        i=re_index(1,sample_data);
        q=re_index(2,sample_data);
        
        Z1_ready(:,q,i)=W'*vector_train(:,q,i);
        Z1_next(:,q,i)=tanh(Z1_ready(:,q,i));
        Z1_offset(:,q,i)=[Z1_next(:,q,i);1];
        
        Z2_ready(:,q,i)=Q'*Z1_offset(:,q,i);
        Z2_next(:,q,i)=tanh(Z2_ready(:,q,i));
        Z2_offset(:,q,i)=[Z2_next(:,q,i);1];
        
        
        Y_ready(:,q,i)=U'*Z2_offset(:,q,i);
        Y_next(:,q,i)=tanh(Y_ready(:,q,i));
        
        Target(:,1)=-0.9*ones(class_lable,1);
        Target(i,1)=0.9;
        Error(:,q,i)=Y_next(:,q,i)-Target(:,1);
        Error_prime(:,q,i)=Error(:,q,i).*(1-Y_next(:,q,i).^2);
        
        for n1=1:neuron_layer1
                for f=1:feature_size+1
                    slack2=0;
                    for c=1:class_lable
                        slack1=0;
                        for n2=1:neuron_layer2
                            slack1=slack1+(U(n2,c)*(1-Z2_offset(n2,q,i)^2)*Q(n1,n2));
                        end
                        slack2=slack2+(Error(c,q,i))*(1-Y_next(c,q,i)^2)*slack1;
                    end
                    diff_W=slack2*(1-Z1_offset(n1,q,i)^2)*(vector_train(f,q,i));
                    W(f,n1)=W(f,n1)-RO*diff_W;
                end
        end
        for u=1:neuron_layer2
            diff_Q=(U(u,:)*Error_prime(:,q,i))*(1-Z2_offset(u,q,i)^2)*Z1_offset(:,q,i);
            Q(:,u)=Q(:,u)-RO.*diff_Q;
        end
        for u=1:class_lable
                diff_U=(Error(u,q,i)*(1-Y_next(u,q,i)^2).*Z2_offset(:,q,i));
                U(:,u)=U(:,u)-RO.*diff_U;
            end

    end
%------------------------------first train data calculated-----------------
    toc
    for i=1:class_lable
        for k=1:sample_size_train
            Z1_ready(:,k,i)=W'*vector_train(:,k,i);
            Z1_next(:,k,i)=tanh(Z1_ready(:,k,i));
            Z1_offset(:,k,i)=[Z1_next(:,k,i);1];

            Z2_ready(:,k,i)=Q'*Z1_offset(:,k,i);
            Z2_next(:,k,i)=tanh(Z2_ready(:,k,i));
            Z2_offset(:,k,i)=[Z2_next(:,k,i);1];


            Y_ready(:,k,i)=U'*Z2_offset(:,k,i);
            Y_next(:,k,i)=tanh(Y_ready(:,k,i));

            Target(:,1)=-0.9*ones(class_lable,1);
            Target(i,1)=0.9;
            Error(:,k,i)=Y_next(:,k,i)-Target(:,1);
            J_error_train(epoch)=J_error_train(epoch)+norm(Error(:,k,i))^2;
            switch max(Y_next(:,k,i))
                case Y_next(1,k,i)
                    confusion_matrix_train(i,1,epoch)=confusion_matrix_train(i,1,epoch)+1;
                case Y_next(2,k,i)
                    confusion_matrix_train(i,2,epoch)=confusion_matrix_train(i,2,epoch)+1;
                case Y_next(3,k,i)
                    confusion_matrix_train(i,3,epoch)=confusion_matrix_train(i,3,epoch)+1;
                case Y_next(4,k,i)
                    confusion_matrix_train(i,4,epoch)=confusion_matrix_train(i,4,epoch)+1;
                case Y_next(5,k,i)
                    confusion_matrix_train(i,5,epoch)=confusion_matrix_train(i,5,epoch)+1;
                case Y_next(6,k,i)
                    confusion_matrix_train(i,6,epoch)=confusion_matrix_train(i,6,epoch)+1;
                case Y_next(7,k,i)
                    confusion_matrix_train(i,7,epoch)=confusion_matrix_train(i,7,epoch)+1;
                case Y_next(8,k,i)
                    confusion_matrix_train(i,8,epoch)=confusion_matrix_train(i,8,epoch)+1;
            end
        end
    end
    CCR_train(epoch)=trace(confusion_matrix_train(:,:,epoch))/n_train;
%------------------------------secound test data calculated----------------
    n_test=sample_size_test*class_lable;
    for i=1:class_lable
        for k=1:sample_size_test
            Z1_ready(:,k,i)=W'*vector_test(:,k,i);
            Z1_next(:,k,i)=tanh(Z1_ready(:,k,i));
            Z1_offset(:,k,i)=[Z1_next(:,k,i);1];

            Z2_ready(:,k,i)=Q'*Z1_offset(:,k,i);
            Z2_next(:,k,i)=tanh(Z2_ready(:,k,i));
            Z2_offset(:,k,i)=[Z2_next(:,k,i);1];


            Y_ready(:,k,i)=U'*Z2_offset(:,k,i);
            Y_next(:,k,i)=tanh(Y_ready(:,k,i));

            Target(:,1)=-0.9*ones(class_lable,1);
            Target(i,1)=0.9;
            Error(:,k,i)=Y_next(:,k,i)-Target(:,1);

            J_error_test(epoch)=J_error_test(epoch)+norm(Error(:,k,i))^2;
            switch max(Y_next(:,k,i))
                case Y_next(1,k,i)
                    confusion_matrix_test(i,1,epoch)=confusion_matrix_test(i,1,epoch)+1;
                case Y_next(2,k,i)
                    confusion_matrix_test(i,2,epoch)=confusion_matrix_test(i,2,epoch)+1;
                case Y_next(3,k,i)
                    confusion_matrix_test(i,3,epoch)=confusion_matrix_test(i,3,epoch)+1;
                case Y_next(4,k,i)
                    confusion_matrix_test(i,4,epoch)=confusion_matrix_test(i,4,epoch)+1;
                case Y_next(5,k,i)
                    confusion_matrix_test(i,5,epoch)=confusion_matrix_test(i,5,epoch)+1;
                case Y_next(6,k,i)
                    confusion_matrix_test(i,6,epoch)=confusion_matrix_test(i,6,epoch)+1;
                case Y_next(7,k,i)
                    confusion_matrix_test(i,7,epoch)=confusion_matrix_test(i,7,epoch)+1;
                case Y_next(8,k,i)
                    confusion_matrix_test(i,8,epoch)=confusion_matrix_test(i,8,epoch)+1;
            end
        end
    end
    CCR_test(epoch)=trace(confusion_matrix_test(:,:,epoch))/n_test;
end
run_time=toc