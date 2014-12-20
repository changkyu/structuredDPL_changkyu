%--------------------------------------------------------------------------
% Synthetic data demonstration
%
% Implemented/Modified
%  by     Changkyu Song (changkyu.song@cs.rutgers.edu)
%  on     2014.11.07 (last modified on 2014/11/07)
%--------------------------------------------------------------------------

%% Reset/Clear
clearvars;
env_setting;

%% Experiments Description
models_desc = { 
                struct('name','cppca', 'type','SVD',   'fn',@cppca   ,'skip',0), ...
                struct('name','cppca', 'type','EM',    'fn',@cppca_em,'skip',0), ...
                struct('name','dppca', 'type','',      'fn',@dppca   ,'skip',0), ...
                struct('name','sdppca','type','bd50',  'fn',@sdppca  ,'skip',0), ...
                %struct('name','sdppca','type','bd75',  'fn',@sdppca  ,'skip',0), ...
                %struct('name','sdppca','type','unbd',  'fn',@sdppca  ,'skip',0), ...
              };
          
n_models = numel(models_desc);
idxes_model_cm = [1,2];
idxes_model_dm = [3,4];

format_dir_result   = 'result/synth_%s';
format_mat_result_cppca = '%s_%s.mat';
format_mat_result_dppca = '%s_N%02d_G%03d_E%02d.mat';
format_mat_result_sdppca = '%s_N%02d_G%03d_E%02d_%s.mat';

%% Initial Setting

% Frequency for intermediate object function value output
objfreq_c = 100;
objfreq_d1 = 100;
objfreq_dj = 100;
THRESHc = 1e-5; % convergence precision (default: 1e-5)
THRESHd = 1e-5;

% Choose random seed. It is same with the S. Yoon's setting
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
reset(s,0);

% 50(D) dimensional, 100(N) samples searching, 5(M) dimensional subspace
N = 100; D = 50; M = 5; VAR = 1;
% Z (dim=M) comes from N(0,I)
Z = randn(s,N,M);
% W (dim=M) comes from N(0,I)
% NOTE: W can be an arbitrary matrix, i.e. W = rand(s,D,M);
W = randn(s,D,M);
% E (dim=D) comes from N(0,VAR*I)
E = (sqrt(VAR) .* randn(s,N,D))';
% Our PPCA model is X = W * Z + E
measurement_matrix = W * Z' + E;
datastr = '';

% get dimension of measurement matrix
[D, N] = size(measurement_matrix);

mm_trans = measurement_matrix;

% initializes with a random projection, zero mean and a small variance
m_init_cppca = get_init_value_ex('cppca', mm_trans, M, 1);


%% Experiments - Centralized
disp('*** Centralized Setting ***');

for idx_model=idxes_model_cm

    model_desc = models_desc{idx_model};
    disp(['PPCA ('  model_desc.type ')']);

    % result directory
    dir_result = sprintf(format_dir_result,model_desc.name);
    if ~exist(dir_result,'dir')
        mkdir(dir_result);
    end
    
    % result mat file
    mat_result = sprintf(format_mat_result_cppca, model_desc.name, model_desc.type);
    path_result = [dir_result '/' mat_result];
    
    % skip if it already exist
    if( model_desc.skip )
        if( exist( path_result, 'file' ) )        
            disp('already exist... skip');
            continue;
        end
    end    
    
    if isequal(model_desc.fn, @cppca)
        model = cppca(mm_trans, M);            
    elseif isequal(model_desc.fn, @cppca_em)        
        model = cppca_em(mm_trans, M, THRESHc, m_init_cppca, objfreq_c);
    else
        error(['[Error] Undefined Function - ' model_desc.name]);
    end
    save_parfor_model(path_result, model);
    disp('saved the result... done');
end

%% Experiments - Distributed
disp('*** Distributed Setting ***');

% V: Node assignment of samples
%Varr = {1, 2, 5, 8, 10};
%Varr = {1, 2, 5, 8, 20, 50};
%Varr = {5, 10, 20, 40, 80};
%Varr = {15, 30};
Varr = {5, 10, 20};

% ETA: Learning rate
%ETAarr = {8, 10, 12, 16};
ETAarr = {10};

% Various number of nodes, network topology and ETA
% NOTE: We used parallel computation toolbox of MATLAB here but one can
%  simply change [parfor] with [for] to run the code without the toolbox.
%parfor idk = 1:length(Varr)
for idk = 1:length(Varr)
    % Node assignment to each sample
    NV = Varr{idk};
    Vp = get_sample_assign(NV, N, 0);
    
    m_init_ddup = get_init_value_ex('d_dup', mm_trans, M, 10, NV, m_init_cppca);
    
    % E: Network topology
    Earr = get_adj_graph(NV);

    %for idx = 1:length(Earr)
    for idx = 1
        E = Earr{idx};

        for idy = 1:length(ETAarr)
            ETA = ETAarr{idy};
            
            for idx_model = idxes_model_dm

                model_desc = models_desc{idx_model};
                if( strcmp(model_desc.name,'sdppca') )
                    if( idx ~= 1 )
                        % SDPPCA starts with fully connected graph
                        continue;
                    end
                end
                
                disp(['PPCA ('  model_desc.name ' ' model_desc.type ')']);

                % result directory
                dir_result = sprintf(format_dir_result,model_desc.name);
                if ~exist(dir_result,'dir')
                    mkdir(dir_result);
                end

                % result mat file
                if strcmp(model_desc.name,'sdppca')
                    mat_result = sprintf(format_mat_result_sdppca, model_desc.name, NV, idx, ETA, model_desc.type);
                else
                    mat_result = sprintf(format_mat_result_dppca, model_desc.name, NV, idx, ETA);
                end
                path_result = [dir_result '/' mat_result];
    
                if( model_desc.skip )
                    if( exist( path_result, 'file' ) )
                        model = load(path_result);
                        model = model.model;
                        
                        disp('already exist... skip');
                        fprintf('Iter = %d:  Cost = %f\n', model.eITER, model.objArray(end,end));
                
                        continue;
                    end
                end
                    
                if isequal(model_desc.fn, @dppca)
                    model = model_desc.fn(mm_trans, Vp, E, M, ETA, THRESHd, m_init_ddup, objfreq_dj);
                elseif isequal(model_desc.fn, @sdppca)
                    model = model_desc.fn(mm_trans, Vp, E, M, ETA, model_desc.type, THRESHd, m_init_ddup, objfreq_dj, '');
                else
                    error(['[Error] Undefined Function - ' model_desc.name]);
                end
                
                fprintf('Iter = %d:  Cost = %f\n', model.eITER, model.objArray(end,end));
                save_parfor_model(path_result, model);
            end
        end
    end
end

% Show Result
%show_result_1_synth_sdppca( savepath{:} );
