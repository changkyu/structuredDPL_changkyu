linetype = {'-', '--', ':', '-.'};
color = {'r', 'm', 'g', 'c', 'b', 'k' };

%% Experiments Description
models_desc = { 
                %struct('name','cppca', 'type','SVD',   'fn',@cppca   ,'skip',0), ...
                %struct('name','cppca', 'type','EM',    'fn',@cppca_em,'skip',0), ...
                struct('name','dppca', 'type','',      'fn',@dppca   ,'skip',0), ...
                %struct('name','sdppca','type','bd75',  'fn',@sdppca  ,'skip',0), ...
                struct('name','sdppca','type','bd50',  'fn',@sdppca  ,'skip',0), ...
                %struct('name','sdppca','type','unbd',  'fn',@sdppca  ,'skip',0), ...
              };
          
n_models = numel(models_desc);
%idxes_model_cm = [1,2];
idxes_model_dm = [1,2];

format_dir_result   = 'result/synth_%s';
format_mat_result_cppca = '%s_%s.mat';
format_mat_result_dppca = '%s_N%02d_G%03d_E%02d.mat';
format_mat_result_sdppca = '%s_N%02d_G%03d_E%02d_%s.mat';

%%

Varr = [5, 10, 20, 40 ];
ETA = 10;

figure();
hold on;

iters = zeros(length(Varr),length(idxes_model_dm));

for idx_model = idxes_model_dm
    idx_m = idx_model-idxes_model_dm(1)+1;
    model_desc = models_desc{idx_model};    
    % result directory
    dir_result = sprintf(format_dir_result,model_desc.name);
    
    
    for idk = 1:length(Varr)
    
        NV = Varr(idk);

        for idx = 1
            % result mat file
            if strcmp(model_desc.name,'sdppca')
                mat_result = sprintf(format_mat_result_sdppca, model_desc.name, NV, idx, ETA, model_desc.type);
            else
                mat_result = sprintf(format_mat_result_dppca, model_desc.name, NV, idx, ETA);
            end
            path_result = [dir_result '/' mat_result];

            res = load(path_result);
            iters(idk,idx_m) = res.model.eITER;
        end
        
    end
        
    %plot(Varr, iters, [color{idx_m} linetype{idx_m}]);    
    %hist(iters, Varr);    
    %bar(Varr, iters);
    bar(iters);
end
hold off;

set(gca, 'XTickLabel',{'5','10','20','40'}, 'XTick',1:4)

xlabel('Number of Nodes');
ylabel('Number of Iteration');
legend({'DPPCA','SDPPCA (Ours)'});
title('Number of Nodes vs. Iterations');