function [Networks] = get_adj_graph(J)
%--------------------------------------------------------------------------
% GET_ADJ_GRAPH     Get a set of square adjacent graphs for some topologies
%
% INPUT
%   J       Number of nodes
%
% OUTPUT
%   Earr    Set of adjacent graphs
%
% Implemented/Modified from [1]
%  by     Changkyu Song (changkyu.song@rutgers.edu)
%  on     2015.02.04 
%
% References
%  [1] S. Yoon and V. Pavlovic. Distributed Probabilistic Learning
%      for Camera Networks with Missing Data. In NIPS, 2012.
%
%--------------------------------------------------------------------------

if J > 4 && mod(J, 4) == 0
    n_Networks = 4;
else    
    n_Networks = 5;
end
Networks = cell(n_Networks,1);

% E: Network topology
E1 = ones(J) - eye(J); % complete
Networks{1} = struct('name', 'complete', 'adj', E1);

E2 = diag(ones(1,J-1),1); E2(J,1) = 1; E2 = E2+E2'; % ring
Networks{2} = struct('name', 'ring', 'adj', E2);

E3 = [ones(1,J);zeros(J-1,J)]; E3 = E3+E3'; % star
Networks{3} = struct('name', 'star', 'adj', E3);

E4 = diag(ones(1,J-1),1); E4 = E4+E4'; % chain
Networks{4} = struct('name', 'chain', 'adj', E4);

if J > 4 && mod(J, 4) == 0
    E5 = [ones(1,J/2) zeros(1,J/2); zeros(J/2-1,J); ...
          zeros(1,J/2) ones(1,J/2) ; zeros(J/2-1,J)]; 
    E5(1,J/2+1)=1; E5=E5+E5'; % cluster
    
    Networks{5} = struct('name', 'cluster', 'adj', E5);
end

end
