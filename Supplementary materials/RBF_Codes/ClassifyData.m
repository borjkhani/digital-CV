function [CCR Confusion J]=ClassifyData(U,X,Xs,sigma)
    [nn,Q,C]=size(X);
    J=0;
    Confusion=zeros(C,C);
    for c=1:C
        for q=1:Q
            Z=sum((repmat(X(:,q,c),1,size(Xs,2))-Xs).^2,1);
            P=exp(-Z./sigma);
            Y=tanh(U'*P');
            T=-0.9*ones(C,1);
            T(c)=0.9;
            e=Y-T;
            J=J+norm(e)^2;
            [mm,index]=max(Y);
            Confusion(c,index)=Confusion(c,index)+1;
        end
    end
    CCR=trace(Confusion)./(Q*C);
    J=J/Q;
end