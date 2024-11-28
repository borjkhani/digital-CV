function [W_new,U_new]=FullPropagation(W,U,r,X)
    [nn,Q,C]=size(X);
    Z=zeros(size(W,2)+1,Q,C);
    Y=zeros(size(U,2),Q,C);
    T=zeros(size(U,2),Q,C);
    e=zeros(size(U,2),Q,C);
    e1=zeros(size(U,2),Q,C);
    d_W=zeros(size(W));
    d_U=zeros(size(U));
    for q=1:Q
        for c=1:C            
            Z(:,q,c)=[tanh(W'*X(:,q,c));1];
            Y(:,q,c)=tanh(U'*Z(:,q,c));
            T(:,q,c)=-0.9*ones(size(U,2),1);
            T(c,q,c)=0.9;
            e(:,q,c)=(Y(:,q,c)-T(:,q,c));
            e1(:,q,c)=e(:,q,c).*(1-Y(:,q,c).^2);
            for m=1:size(W,2)
                 d_W(:,m)=d_W(:,m)+(U(m,:)*e1(:,q,c))*(1-Z(m,q,c)^2).*X(:,q,c);
            end
            for m=1:size(U,2)
                d_U(:,m)=d_U(:,m)+(e(m,q,c)*(1-Y(m,q,c)^2).*Z(:,q,c));
            end
        end
    end
    W_new=W-r.*d_W;
    U_new=U-r.*d_U;
end