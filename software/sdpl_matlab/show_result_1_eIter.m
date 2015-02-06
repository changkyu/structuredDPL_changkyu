%% Experiments Description
clearvars;

name_experiment = 'synth';
dir_result = './results';
dir_experiment = fullfile( dir_result, name_experiment );
path_desc = fullfile(dir_experiment, [name_experiment '_description.mat']);

load( path_desc );

linetype = {'-', '--', ':', '-.'};
color = {'r', 'm', 'g', 'c', 'b', 'k' };

%% Read Result Files

n_networks = 5;
iters = zeros(length(NVarr),n_networks,length(idxes_model_dm));

for idx_model = idxes_model_dm
    
    model_desc = models_desc{idx_model};    
    % result directory
    dir_result = sprintf(format_dir_result,model_desc.name);
    
    for idx_NV = 1:length(NVarr)
        NV = NVarr(idx_NV);
        
        % Network topology
        Networks = get_adj_graph(NV);
            
        for idx_Network=1:n_networks
            for idx_ETA = 1:length(ETAarr)                
                ETA = ETAarr(idx_ETA);
                
                % result mat file
                mat_result = sprintf(model_desc.format_result, Networks{idx_Network}.name, NV, ETA);
                path_result = fullfile(dir_experiment, model_desc.name, mat_result);

                res = load(path_result);
                iters(idx_NV,idx_Network,idx_m) = res.model.eITER;
            end
        end
        
    end
end

%% Show Result (# of Nodes vs. iters)
close all;

str_xlabel = cell(length(NVarr),1);
for idx_NV=1:length(NVarr)
    str_xlabel{idx_NV} = num2str(NVarr(idx_NV));
end

for idx_Network=1:n_networks

    figure();
    hold on;
    hold off;
    
    iter = sqeeze(iters(:,idx_Network,:));
    bar(iter);

    set(gca, 'XTickLabel',str_xlabel, 'XTick',1:length(NVarr))

    xlabel('Number of Nodes');
    ylabel('Number of Iteration');
    legend({'DPPCA','Ours'});
    title('Number of Nodes vs. Iterations');

end

%% Show Result