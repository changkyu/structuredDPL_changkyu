
%%

models_desc = { 
                struct('name','cppca', 'type','SVD',   'fn',@cppca   ,'skip',1), ...
                struct('name','cppca', 'type','EM',    'fn',@cppca_em,'skip',1), ...
                struct('name','dppca', 'type','',      'fn',@dppca   ,'skip',1), ...
                struct('name','sdppca','type','bd75',  'fn',@sdppca  ,'skip',0), ...
                struct('name','sdppca','type','bd50',  'fn',@sdppca  ,'skip',0), ...
                struct('name','sdppca','type','unbd',  'fn',@sdppca  ,'skip',0), ...
              };

n_models = numel(models_desc);
n_models = 5;
idxes_model_cm = [2];
idxes_model_dm = [3,4,5,6];

format_dir_result   = 'result/synth_%s';
format_mat_result_cppca = '%s_%s.mat';
format_mat_result_dppca = '%s_N%02d_G%03d_E%02d.mat';
format_mat_result_sdppca = '%s_N%02d_G%03d_E%02d_%s.mat';

%% Synthetic Result #1

ETA = 10;
NV  = 20;
type_G = 1;

idx = 1;
path_results = cell(n_models,1);
models = cell(n_models,1);

for idx_model=idxes_model_cm

    model_desc = models_desc{idx_model};
    
    dir_result = sprintf(format_dir_result,model_desc.name);
    mat_result = sprintf(format_mat_result_cppca, model_desc.name, model_desc.type);
    path_results{idx} = [dir_result '/' mat_result];
    models{idx} = load(path_results{idx});
    
    idx = idx + 1;
end

for idx_model=idxes_model_dm
    
    model_desc = models_desc{idx_model};

    dir_result = sprintf(format_dir_result,model_desc.name);
    if strcmp(model_desc.name,'sdppca')
        mat_result = sprintf(format_mat_result_sdppca, model_desc.name, NV, type_G, ETA, model_desc.type);
    else
        mat_result = sprintf(format_mat_result_dppca, model_desc.name, NV, type_G, ETA);
    end
    path_results{idx} = [dir_result '/' mat_result];
    models{idx} = load(path_results{idx});
    
    idx = idx + 1;
end

%show_result_1_synth_sdppca( path_results{:} );