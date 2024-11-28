clc;
clear;
close all

load evaporator.dat

%%%%%%%Data Initializing with Delay%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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


%%%%%%%%%%%%%%%%%%%%%%%%%%LS-SVM%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%reg=[1 10 15 20 25];


% reg size must be equal to ker size



mse_trn=zeros(3,30);
mse_tst=zeros(3,30);
count=0;



gam =20;

     for l=1:30
    
     count=count+1

     sig2 =2000*l+10000;
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
        mse_trn(i,count)=sum((Ytrn(:,i)'-p_trn(6+2*i,1:4728)).^2)/4728
        end
        
        
        
     
%%%% Test
       
       alpha_tst = simlssvm({p_trn',p_trn([8 10 12],1:4728)',type,gam,sig2,'RBF_kernel','preprocess'},{alpha_trn,b},p_tst');
 
      
      % Ytst(:,1)=(alpha_tst/gam+b);
      
        
      for j=1:3
       Ytst(:,j)=(alpha_tst(:,j));
       mse_tst(j,count)=sum((Ytst(:,j)'-p_tst(6+j*2,1:1576)).^2)/1576
      end
      
      
      
        
end

  figure 
subplot(3,1,1)
plot(1:30,mse_tst(1,:),'r',1:30,mse_trn(1,:),'b')
legend ('Test','Train')
xlabel('Samples')
ylabel('Error')

 
subplot(3,1,2)
plot(1:30,mse_tst(2,:),'r',1:30,mse_trn(2,:),'b')
legend ('Test','Train')
xlabel('Samples')
ylabel('Error')
 
 subplot(3,1,3)
plot(1:30,mse_tst(3,:),'r',1:30,mse_trn(3,:),'b')
legend ('Test','Train')
xlabel('Samples')
ylabel('Error')
 




      
      
      
%         figure
%         plot(1:7200,p_trn(out,1:7200),'r',1:7200,Ytrn,'b')
%         figure
%         plot(1:2400,p_tst(out,1:2400),'r',1:2400,Ytst,'b')
        
        
    