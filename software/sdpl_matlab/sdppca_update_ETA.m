function [ ETA_new ] = sdppca_update_ETA( ETA_old, Fpn, type_update )
% update_ETA        
% Update adaptive penalty ETA with Fpn
% 
% Description
% ETA_old     : Previous ETA
% Fpn         : The first term of local objective value 
%               F[row:parameter learned from each node, col:node data]
% type_update : Updating type
%
% Output
% ETA_new     : new penalty ETA
%
% Implemented 
%  by     Changkyu Song (changkyu.song@cs.rutgers.edu)
%  on     2014.11.07 (last modified on 2014.11.07)
%
ETA_new = zeros(size(ETA_old));

J = size(Fpn, 1);

maxFn  = max(Fpn);
minFn  = min(Fpn);
distFn = maxFn - minFn;

zFpn = (Fpn - repmat(minFn,J,1)) ./ repmat(distFn,J,1) + 1;

for idx=1:J
    if( distFn(idx) > 10 )
        ratioFp = zFpn(idx,idx) ./ zFpn(:,idx);
        
        switch type_update
            case 'bd75'
                ETA_new(idx,:) = ETA_old(idx,:) .* max(min(ratioFp,1.25),0.75)';
            case 'bd50'                
                ETA_new(idx,:) = ETA_old(idx,:) .* max(min(ratioFp,1.50),0.50)';
            case 'unbd'
                ETA_new(idx,:) = ETA_old(idx,:) .* ratioFp';
            otherwise
                error('[Error] Undefined update type');
        end    
    else
        ETA_new(idx,:) = ETA_old(idx,:);
    end
end

end

