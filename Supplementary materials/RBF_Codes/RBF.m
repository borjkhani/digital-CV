function U_new=RBF(U,r,X,Xs,sigma)
    [nn,Q,C]=size(X);
    d_U=zeros(size(Xs,2),C);
    N=Q*C;
    Xrand=randperm(N);
    for s=1:N
        c=fix((Xrand(s)-1)/Q)+1;
        q=Xrand(s)-(c-1)*Q;        
        Z=sum((repmat(X(:,q,c),1,size(Xs,2))-Xs).^2,1);
        P=exp(-Z./sigma);
        Y=tanh(U'*P');
        T=-0.9*ones(C,1);
        T(c)=0.9;
        e=Y-T;            
        for u=1:C
            d_U(:,u)=d_U(:,u)+(e(u)*(1-Y(u)^2).*P');
        end
    end    
    U_new=U-r*d_U;
end