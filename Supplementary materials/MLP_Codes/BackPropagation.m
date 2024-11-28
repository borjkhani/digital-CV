function [W_new,U_new]=BackPropagation(W,U,r,X)
    [nn Q C]=size(X);
    Z=zeros(size(W,2)+1,Q,C);
    Y=zeros(size(U,2),Q,C);
    T=zeros(size(U,2),Q,C);
    e=zeros(size(U,2),Q,C);
    e1=zeros(size(U,2),Q,C);
    N=Q*C;
    Xrand=randperm(N);
    for s=1:N
        c=fix((Xrand(s)-1)/Q)+1;
        q=Xrand(s)-(c-1)*Q;
        Z(:,q,c)=[tanh(W'*X(:,q,c));1];
        Y(:,q,c)=tanh(U'*Z(:,q,c));
        T(:,q,c)=-0.9*ones(size(U,2),1);
        T(c,q,c)=0.9;
        e(:,q,c)=(Y(:,q,c)-T(:,q,c));
        e1(:,q,c)=e(:,q,c).*(1-Y(:,q,c).^2);
        for m=1:size(W,2)
             d_W=(U(m,:)*e1(:,q,c))*(1-Z(m,q,c)^2).*X(:,q,c);
             W(:,m)=W(:,m)-r.*d_W;
        end
        for m=1:size(U,2)
            d_U=(e(m,q,c)*(1-Y(m,q,c)^2).*Z(:,q,c));
            U(:,m)=U(:,m)-r.*d_U;
        end        
    end
    U_new=U;
    W_new=W;
end