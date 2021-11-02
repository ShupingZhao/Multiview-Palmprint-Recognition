function [Q,E] = DC_MDPR(X,T,W,Train_Lab,opts)
% double-cohesion multiview palmprint recognition, code is written by Shuping

miu = opts.miu;
rho = opts.rho;
max_miu = 1e8;
lambda1 = opts.lambda1;
lambda2 = opts.lambda2;
nnClass   = opts.nnClass;
max_iter  = opts.maxIter;

iv = length(X);

for i = 1:iv
    Q{i}  = rand(nnClass,size(X{i},1));
    E{i}  = zeros(nnClass,size(X{i},2));
end
D = diag(sum(W,2));

[maxORY, indexORY]=max(T,[],1);

U=T;
% U(U==0)=-1;
for iter = 1:max_iter
    
    M = zeros(size(T));
    for j = 1:iv
         M = M+Q{j}*X{j};
    end
    M = M/iv;
         
    [maxF, index]=max(M,[],1);
    for i=1:length(index)
        [mm,inde]=max(M(:,i));
        M=normalize1(M);
        if indexORY(i)==inde
            U(indexORY(i),i)=1;
        else
            U(indexORY(i),i)=2;
        end
    end

    
    for j = 1:iv
        Y=T+E{j}+U.*M;
        Q{j} = Y*X{j}'*inv(X{j}*X{j}'+X{j}*(D-W)'*X{j}'+X{j}*(D-W)*X{j}'+lambda1*eye(size(X{j},1)));
    end
    
    for j = 1:iv
        K{j} = Q{j}*X{j}-T-U.*M;
        E{j} = solve_l1l2(K{j}',lambda2)';
    end
     
end

end
    
    