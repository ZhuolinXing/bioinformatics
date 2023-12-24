function [nmi,ACC,AR,f,p,r,RI,Z_all,pre,cov_val,Z1,Z2] = C_solver(X,gt,paras)

% ---------------------------------- data pre-processing
V = size(X,2);      % number of views
N = size(X{1},2);   % number of samples
cls_num  = size(unique(gt),1);
% cls_num = 60;
lambda = paras.lambda;
Ns     = paras.Ns;

% for i = 1:V
%     X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+10e-10);
% end
% ---------------------------------------------- initialize variables
for i = 1:V
    Eh{i} = zeros(Ns,N);
    Yh{i} = zeros(Ns,N);
    Ys{i} = zeros(N,N);
    Zv{i} = zeros(N,N); 
    T{i} = zeros(N,N);
end

Z_tensor = cat(3,Zv{:,:});
T_tensor = cat(3,T{:,:});
Ys_tensor = cat(3,Ys{:,:});
sX = [N,N,V];
% ----------------------------------------------
IsConverge = 0;
mu         = 1e-4;
pho        = 2;
% pho        = 1.5;
max_mu     = 1e6;
max_iter   = 100;
iter       = 1;
thresh     = 1e-6;
% -------------------------------------------------------------------------
tic
while (IsConverge == 0&&iter<max_iter+1)
    
    B = [];
    d = 1;
    for i = 1:V
        % ---------------------------------- update Pv
        P{i} = updatePP(Yh{i},mu,Eh{i}, X{i}-X{i}*Zv{i});
           
        % ---------------------------------- update Zv
        A = P{i}*X{i};
        Zv{i} = (A'*A+eye(N))\(A'*Yh{i}/mu+A'*(A-Eh{i})+T{i}-Ys{i}/mu);
%         Zv{i} = Zv{i} - diag(diag(Zv{i}));
        Zv{i} = (Zv{i}+Zv{i}') / 2;
        
        % ---------------------------------- update Ev
        G = P{i}*X{i} - P{i}*X{i}*Zv{i} + Yh{i}/mu;
        B = [B;G];
        E = solve_l1l2(B,lambda/mu);
                
        Eh{i} = E(d:i*Ns,:);
        d = i*Ns + 1;              
    end
    % ---------------------------------- update T_tensor
    Z_tensor = cat(3,Zv{:,:});
    T_tensor = cat(3,T{:,:});
    Ys_tensor = cat(3,Ys{:,:});
    
    [t_tensor, objV] = wshrinkObj(Z_tensor + 1/mu*Ys_tensor,1/mu,sX,0,3);
    T_tensor = reshape(t_tensor, sX);
    
    for i = 1:V
        Zv{i} = Z_tensor(:,:,i);
        T{i} = T_tensor(:,:,i);
        Ys{i} = Ys_tensor(:,:,i);
    end
    
    % ---------------------------------- uconstructW_PKNpdata multipliers
    for i = 1:V
        Yh{i} = Yh{i} + mu*(P{i}*X{i} - P{i}*X{i}*Zv{i} - Eh{i});
        Ys{i} = Ys{i} + mu*(Zv{i}-T{i});
    end
    mu = min(pho*mu, max_mu);
    
    % ----------------------------------- convergence conditions
    min_err = 0;
    for i = 1:V
        errp(i) = norm(P{i}*X{i}  - P{i}*X{i}*Zv{i} - Eh{i},inf);
        errs(i) = norm(Zv{i} - T{i} , inf);   
    end
    
    max_err = max(errp(:)+errs(:));
    
    % -----------------------------------
    if max_err < thresh
        IsConverge = 1;
    end
    cov_val(iter) = max_err;

    iter = iter + 1; 
end
time = toc;
% -------------------------------------------------------------------------
Z_all = zeros(N,N);
for j = 1:V  
    Z_all = Z_all + (abs(Zv{i}) + abs(Zv{i}'));
end
Z_all = Z_all / V;
Z1 = Zv{1};
Z2 = Zv{2};
[nmi,ACC,AR,f,p,r,RI,pre] = clustering(Z_all, cls_num, gt);
