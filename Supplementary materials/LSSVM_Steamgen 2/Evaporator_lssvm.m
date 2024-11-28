clc;
clear;
close all

load evaporator.dat

%%%%%%%Data Initializing with Delay%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


depthu=2;
depthy=2;

dataSet1 = evaporator(1:6305, 1:6)';
r = size(dataSet1, 1);
N = size(dataSet1, 2);
N_trn = 0.75*N;
N_tst = 0.25*N;

dataSetu = [zeros(r/2, depthu), dataSet1(1:r/2, :)];
dataSety = [zeros(r/2, depthy), dataSet1(r/2+1:r, :)];

%U=zeros(8,1:N);


U1 = dataSetu(1, 1:N);
for i = 2:depthu
    U1 = [U1; dataSetu(1, i:N+i-1)];   
end

U2 = dataSetu(2, 1:N);
for i = 2:depthu
    U2 = [U2; dataSetu(2, i:N+i-1)];   
end


U3 = dataSetu(3, 1:N);
for i = 2:depthu
    U3 = [U3; dataSetu(3, i:N+i-1)];   
end




%
U=[U1;U2;U3];


Y1 = dataSety(1, 1:N)
for i =2:depthy
    Y1 = [Y1; dataSety(1, i:N+i-1)];
end


Y2 = dataSety(2, 1:N)
for i =2:depthy
    Y2 = [Y2; dataSety(2, i:N+i-1)];
end

Y3 = dataSety(3, 1:N)
for i =2:depthy
    Y3 = [Y3; dataSety(3, i:N+i-1)];
end



Y=[Y1;Y2;Y3];


p = [U; Y];
t = dataSety(1, depthy+1:depthy+N);

%%%%%%%Normalization%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


%%%%%%%%%%%%%%%%%%%%%%%%%%LS-SVM%%%%%%%%%%%%



gam =20;
sig2 = 40000;
%sig3=sig2-1e5;
%passes=40;


% A simple example shows how to start using the toolbox for a classi?cation task. We start with
% constructing a simple example dataset according to the correct formatting. Data are represented
% as matrices where each row of the matrix contains one datapoint
% Choose the Function type

%%outputs: 10, 12 , 14 , 16
%%% Train


 type = 'function estimation';
 kernel = 'MLP_kernel';
  


        [alpha_trn,b] = trainlssvm({p_trn',p_trn([8 10 12 ],1:4728)',type,gam,sig2,'RBF_kernel','original'});
       
        for i=1:3
        Ytrn(:,i)=(alpha_trn(:,i)/gam+b(i));
        mse_trn(i)=sum((Ytrn(:,i)'-p_trn(6+2*i,1:4728)).^2)/4728
        end
        
        
        
     
%%%% Test
       
       alpha_tst = simlssvm({p_trn',p_trn([8 10 12],1:4728)',type,gam,sig2,'RBF_kernel','preprocess'},{alpha_trn,b},p_tst');
 
      
      % Ytst(:,1)=(alpha_tst/gam+b);
      
        
      for j=1:3
       Ytst(:,j)=(alpha_tst(:,j));
       mse_tst(j)=sum((Ytst(:,j)'-p_tst(6+j*2,1:1576)).^2)/1576
      end
        
     figure 
subplot(3,1,1)
plot(1:4728,p_trn(8,1:4728),'r',1:4728,Ytrn(:,1),'b')
legend ('Real','Trained Output')
xlabel('Samples')
ylabel('Output num 1')

 
subplot(3,1,2)
plot(1:4728,p_trn(10,1:4728),'r',1:4728,Ytrn(:,2),'b')
legend ('Real','Trained Output')
xlabel('Samples')
ylabel('Output num 2')
 
 subplot(3,1,3)
 plot(1:4728,p_trn(12,1:4728),'r',1:4728,Ytrn(:,3),'b')
legend ('Real','Trained Output')
xlabel('Samples')
ylabel('Output Num 3')
 

 
     figure
      
   subplot(3,1,1)
plot(1:1576,p_tst(8,1:1576),'r',1:1576,Ytst(:,1),'b')
legend ('Real','Test Output')
xlabel('Samples')
ylabel('Output num 1')

 
subplot(3,1,2)
plot(1:1576,p_tst(10,1:1576),'r',1:1576,Ytst(:,2),'b')
legend ('Real','Test Output')
xlabel('Samples')
ylabel('Output num 2')
 
 subplot(3,1,3)
 plot(1:1576,p_tst(12,1:1576),'r',1:1576,Ytst(:,3),'b')
legend ('Real','Test Output')
xlabel('Samples')
ylabel('Output Num 3')
 

      
      
      

        
    