
clc;
clear;
close all

load steamgen.dat

depthu=0;
depthy=0;

dataSet1 = steamgen(1:9600, 2:9)';
r = size(dataSet1, 1);
N = size(dataSet1, 2);
N_trn = 0.75*N;
N_tst = 0.25*N;

dataSetu = [zeros(r/2, depthu), dataSet1(1:r/2, :)];
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
%insert sig2
gam =3;
sig2 = 40;
sig3=sig2-35;
passes=40;
% A simple example shows how to start using the toolbox for a classi?cation task. We start with
% constructing a simple example dataset according to the correct formatting. Data are represented
% as matrices where each row of the matrix contains one datapoint:


N_trn = length(t_trn);
N_tst = length(t_tst);

    for i = 40:N_trn
        
%         X(:,i)=p_trn(1:40,i);
%         Y(:,i)=p_trn(41:80,i);
        type = 'function estimation';
        kernel = 'RBF_kernel';
        [alpha,b] = trainlssvm({X(:,i),Y(:,i),type,gam,sig2,'RBF_kernel','original'});
        Ytrn(:,i)=(alpha/gam+b);
        
    end
    
%     for j=1:190
%         Ytrain(:,190)=Ytrn(:,40*j);  
        
        
       

            
            
            
     
    
   


