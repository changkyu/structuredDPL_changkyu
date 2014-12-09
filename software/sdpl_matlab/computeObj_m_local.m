function F = computeObj_m_local( Xi, MissIDXi, W_new, MU_new, PREC_new, EZn, EZnZnt)
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

obj_val1 = 0;
obj_val2 = 0;
obj_val3 = 0;
obj_val4 = 0;
obj_val5 = 0;
for n = 1:Ni
    DcI = (MissIDXi(:,n) == 1);
    Xc = Xi(DcI,n);
    MUc = MU_new(DcI);
    Wc = W_new(DcI,:);

    obj_val1 = obj_val1 + 0.5 * sum(DcI) * log(2 * pi / PREC_new);
    obj_val2 = obj_val2 + 0.5 * trace(EZnZnt(:,:,n));
    obj_val3 = obj_val3 + (0.5*PREC_new) * norm(Xc - MUc, 2).^2;
    obj_val4 = obj_val4 + PREC_new * EZn(:,n)' * Wc' * (Xc - MUc);
    obj_val5 = obj_val5 + (0.5*PREC_new) * trace(EZnZnt(:,:,n) * (Wc' * Wc));
end
F = (obj_val1 + obj_val2 + obj_val3 - obj_val4 + obj_val5);

end

