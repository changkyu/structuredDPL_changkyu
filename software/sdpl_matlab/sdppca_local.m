function [W_new, MU_new, PREC_new, EZn, EZnZnt] = sdppca_local(...
    Xi, M, idx, Bj, ETAi, Wi, MUi, PRECi, LAMBDAi, GAMMAi, BETAi)
% DPPCA_LOCAL  Distributed Probablistic PCA (D-PPCA) Local Node
% 
% Description
%  model = dppca_local(.) solves local optimization problem using iteration 
% forumla and return consensus-enforced parameters so that they can be 
% broadcasted to neighbors. For simpler implementation, this function can 
% access all parameters in the network although they are only using the
% values actually accessible in the real environments.
%
% Input
% Xi    : D x Ni matrix for data from idx-th node (N=Ni)
% M     : Projected dimension
% idx   : Current node index
% Bj    : List of indexes in the ball Bi (one-hop neighbors of node i)
% ETAi  : List of Scalar Learning ratio from the ball Bi
% Wi, MUi, VARi, LAMBDAi, GAMMAi, BETAi: All parameters in the network
%
% Output
% W_new    : D x M projection matrix
% MU_new   : D x 1 vector sample means
% PREC_new : 1 x 1 scalar estimated variance
% EZ_new   : M x N matrix containing mean of latent space
% EZZt_new : M x M x N cube containing covariances of latent space
%~F_new    : 1 x 1 scalar computed optimization forumla (first term only)
%
% Modified from dppca_local.m of [1]
%  by     Sejong Yoon (sjyoon@cs.rutgers.edu)
%  on     2011.10.07 (last modified on 2012/02/01)
%
% References
%  [1] S. Yoon and V. Pavlovic. Distributed Probabilistic Learning
%      for Camera Networks with Missing Data. In NIPS, 2012.

% Get size of this samples and ball of this node
[D, Ni] = size(Xi);
cBj = length(Bj);
sumETAi = sum(ETAi(Bj));

% Initialize latent variables (for loop implementation)
EZn = zeros(M, Ni);
EZnZnt = zeros(M, M, Ni);

%% E-step

% Compute Mi^(-1) = (Wi'Wi + VARi*I)^(-1) first
Miinv = inv( Wi(:,:,idx)' * Wi(:,:,idx) + 1/PRECi(idx) * eye(M) );

% E[Zn] = Mi^(-1) * Wi' * (Xin - MUi)
% Currently M x N
EZn = Miinv * Wi(:,:,idx)' * (Xi - repmat(MUi(:,idx), [1, Ni]));

for n = 1:Ni
    % E[z_n z_n'] = VAR * Minv + E[z_n]E[z_n]'
    % Currently M x M
    EZnZnt(:,:,n) = 1/PRECi(idx) * Miinv + EZn(:,n) * EZn(:,n)';
end
    
%% M-step

% Update Wi
% [ORG] 
%W_new1 = sum(EZnZnt, 3) * PRECi(idx) + 2*ETA*cBj*eye(M);
% [NEW]
W_new1 = sum(EZnZnt, 3) * PRECi(idx) + 2*sumETAi*eye(M);
W_new2 = zeros(D, M);
for n = 1:Ni
    W_new2 = W_new2 + (Xi(:,n) - MUi(:,idx)) * EZn(:,n)';
end
W_new2 = W_new2 * PRECi(idx);
W_new3 = 2 * LAMBDAi(:,:,idx);
W_new4 = zeros(D, M);

% [ORG]
%for jn = 1:cBj
%    W_new4 = W_new4 + (Wi(:,:,idx) + Wi(:,:,Bj(jn)));
%end
%W_new = (W_new2 - W_new3 + ETA * W_new4) / W_new1 ;
% [NEW]
for jn = 1:cBj
    W_new4 = W_new4 + ETAi(Bj(jn))*(Wi(:,:,idx) + Wi(:,:,Bj(jn)));
end
W_new = (W_new2 - W_new3 + W_new4) / W_new1 ;

% Update MUi
% [ORG]
%MU_new1 = Ni * PRECi(idx) + 2*ETA*cBj;
% [NEW]
MU_new1 = Ni * PRECi(idx) + 2*sumETAi;
MU_new2 = zeros(D, 1);
for n = 1:Ni
    MU_new2 = MU_new2 + (Xi(:,n) - W_new*EZn(:,n));
end
MU_new2 = PRECi(idx) * MU_new2;
MU_new3 = 2 * GAMMAi(:,idx);
MU_new4 = zeros(D, 1);
% [ORG]
%for jn = 1:cBj
%    MU_new4 = MU_new4 + MUi(:,idx) + MUi(:,Bj(jn));
%end
%MU_new = MU_new1^(-1) * (MU_new2 - MU_new3 + ETA * MU_new4);
for jn = 1:cBj
    MU_new4 = MU_new4 + ETAi(Bj(jn))*(MUi(:,idx) + MUi(:,Bj(jn)));
end
MU_new = MU_new1^(-1) * (MU_new2 - MU_new3 + MU_new4);

% Solve for VARi^(-1)
% [ORG]
%PREC_new1 = 2*ETA*cBj;
PREC_new1 = 2*sumETAi;
PREC_new21 = 2*BETAi(idx);
PREC_new22 = 0;
for jn = 1:cBj
    % [ORG] 
    %PREC_new22 = PREC_new22 + ETA*(PRECi(idx) + PRECi(Bj(jn)));
    % [NEW]
    PREC_new22 = PREC_new22 + ETAi(Bj(jn))*(PRECi(idx) + PRECi(Bj(jn)));
end
PREC_new23 = 0;
PREC_new24 = 0;
for n = 1:Ni
    PREC_new23 = PREC_new23 + EZn(:,n)'*W_new'*(Xi(:,n)-MU_new);
    PREC_new24 = PREC_new24 + 0.5 * ( norm(Xi(:,n)-MU_new,2)^2 ...
        + trace( EZnZnt(:,:,n) * W_new' * W_new ) );
end
PREC_new2 = PREC_new21 - PREC_new22 - PREC_new23 + PREC_new24;
PREC_new3 = -Ni * D / 2;
PREC_new = roots([PREC_new1, PREC_new2, PREC_new3]);

% We follow larger, real solution
if length(PREC_new) > 1
    PREC_new = max(PREC_new);
end
if abs(imag(PREC_new)) ~= 0i
    error('No real solution!');
    % We shouldn't reach here since both solutions are not real...
end
if PREC_new < 0
    error('Negative precicion!');
end

%% Compute local objective formula
% [ORG]
% S = cov(Xi');
% C = W_new * W_new' + (1/PREC_new) * eye(D);
% if D>M
%     logDetC = (D-M)*log(1/PREC_new) + log(det(W_new'*W_new + (1/PREC_new)*eye(M)));
% else
%     logDetC = log(det(C));
% end
% F_new = (Ni/2)*(logDetC + trace(C\S));
% [NEW] moved to the outside of this function
