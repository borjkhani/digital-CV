function [CCR Confusion J]=ClassifyData(W,U,X)
    [nn,Q,C]=size(X);
    Z=zeros(size(W,2)+1,Q,C);
    Y=zeros(size(U,2),Q,C);
    T=zeros(size(U,2),Q,C);
    e=zeros(size(U,2),Q,C);
    J=0;
    Confusion=zeros(C,C);
    for c=1:C
        for q=1:Q
            Z(:,q,c)=[tanh(W'*X(:,q,c));1];
            Y(:,q,c)=tanh(U'*Z(:,q,c));
            T(:,q,c)=-1*ones(size(U,2),1);
            T(c,q,c)=1;
            e(:,q,c)=Y(:,q,c)-T(:,q,c);
            J=J+norm(e(:,q,c))^2;
            [mm index]=max(Y(:,q,c));
            Confusion(c,index)=Confusion(c,index)+1;
        end
    end
    CCR=trace(Confusion)./(Q*C);
end
    