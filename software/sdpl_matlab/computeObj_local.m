function F = computeObj_local( Xi, M, W, PREC )
% computeObj_local        
% Compute local objective formula
% 
% Description
% Xi      : D x Ni matrix for data from idx-th node (N=Ni)
% M       : Projected dimension
% W, PREC : All parameters in the network
%
% Output
% F       : 1 x 1 scalar computed optimization forumla (first term only)
%
% Modified from dppca_local.m of [1]
%  by     Changkyu Song (changkyu.song@cs.rutgers.edu)
%  on     2014.11.07 (last modified on 2014.11.07)
%
% References
%  [1] S. Yoon and V. Pavlovic. Distributed Probabilistic Learning
%      for Camera Networks with Missing Data. In NIPS, 2012.

[D, Ni] = size(Xi);

S = cov(Xi');
C = W * W' + (1/PREC) * eye(D);
if D>M
    logDetC = (D-M)*log(1/PREC) + log(det(W'*W + (1/PREC)*eye(M)));
else
    logDetC = log(det(C));
end
F = (Ni/2)*(logDetC + trace(C\S));

end

