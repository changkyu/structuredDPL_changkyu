%--------------------------------------------------------------------------
% Synthetic data demonstration
%
% Implemented/Modified from [1]
%  by     Changkyu Song (changkyu.song@cs.rutgers.edu)
%  on     2014.11.07 (last modified on 2014/11/07)
%
% References
%  [1] S. Yoon and V. Pavlovic. Distributed Probabilistic Learning
%      for Camera Networks with Missing Data. In NIPS, 2012.
%
%--------------------------------------------------------------------------

%% Reset/Clear
clearvars;
env_setting;

%% Experiments Description
name_experiment = 'synth';
models_desc = { 
                struct( 'name',          'cppca',     ...
                        'type',          'SVD',       ...
                        'fn',            @cppca,      ...
                        'format_result', 'cppca_SVD.mat', ...
                        'skip',          0), ...
                struct( 'name',          'cppca',     ...
                        'type',          'EM',        ...
                        'fn',            @cppca_em,   ...
                        'format_result', 'cppca_EM.mat', ...
                        'skip',          0), ...
                struct( 'name',          'dppca',     ...
                        'type',          '',          ...
                        'fn',            @dppca,      ...
                        'format_result', 'dppca__%s_NV%d_ETA%d.mat', ...
                        'skip',          0), ...
                struct( 'name',          'adppca',    ...
                        'type',          'bd50',      ...
                        'fn',            @sdppca  ,   ...
                        'format_result', 'adppca_bd50__%s_NV%d_ETA%d.mat', ...
                        'skip',          0), ...
              };
          
idxes_model_cm  = [1,2];
idxes_model_dm  = [3,4];
idxes_model_all = [idxes_model_cm idxes_model_dm];

dir_result = './results';
if ~exist(dir_result,'dir')
    mkdir(dir_result);
end

dir_experiment = fullfile( dir_result, name_experiment );
if ~exist(dir_experiment,'dir')
    mkdir(dir_experiment);
end

for idx_model = idxes_model_all
    model_desc = models_desc{idx_model};
    dir_model = fullfile(dir_experiment, model_desc.name);
    if ~exist(dir_model,'dir')
        mkdir(dir_model);
    end
end

disp(['============ Experiment: ' name_experiment '============']);   

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

%% Initial Setting for DPPCA and ADPPCA
% NV: the number of vertices (nodes) in the network
NVarr = [5, 10, 20, 40];

% ETA: Learning rate
ETAarr = [10];

%% Experiments - Centralized

for idx_model=idxes_model_cm

    model_desc = models_desc{idx_model};
    fprintf(['\nModel: ' model_desc.name ' ' model_desc.type '\n']);
   
    % result mat file (skip if it already exist)
    mat_result = model_desc.format_result;
    path_result = fullfile(dir_experiment, model_desc.name, mat_result);
    if( model_desc.skip )
        if( exist( path_result, 'file' ) )        
            fprintf('[Skip] already exist...\n');
            continue;
        end
    end    
    
    if isequal(model_desc.fn, @cppca)
        model = model_desc.fn(mm_trans, M);            
    elseif isequal(model_desc.fn, @cppca_em)        
        model = model_desc.fn(mm_trans, M, THRESHc, m_init_cppca, objfreq_c);
    else
        error(['[Error] Undefined Function - ' model_desc.name]);
    end
    model.W_GT = W;
    save_parfor_model(path_result, model);
    fprintf('[Done] saved the result.\n');
end

%% Experiments - Distributed

for idx_model = idxes_model_dm

    model_desc = models_desc{idx_model};
    fprintf(['\nModel: ' model_desc.name ' ' model_desc.type '\n']);

    % Various number of nodes, network topology and ETA
    % NOTE: We used parallel computation toolbox of MATLAB here but one can
    %  simply change [parfor] with [for] to run the code without the toolbox.
    %parfor idk = 1:length(Varr)
    for idx_NV = 1:length(NVarr)
        % Node assignment to each sample
        NV = NVarr(idx_NV);
        fprintf(['# of nodes: ' num2str(NV) '\n']);
        
        Vp = get_sample_assign(NV, N, 0);

        m_init_ddup = get_init_value_ex('d_dup', mm_trans, M, 10, NV, m_init_cppca);

        % Network topology
        Networks = get_adj_graph(NV);

        %for idx = 1:length(Networks)
        for idx_Network = 1
            fprintf(['Network: ' Networks{idx_Network}.name '\n']);

            for idx_ETA = 1:length(ETAarr)
                ETA = ETAarr(idx_ETA);
                fprintf(['ETA: ' num2str(ETA) '\n']);

                % result mat file
                mat_result = sprintf(model_desc.format_result, Networks{idx_Network}.name, NV, ETA);
                path_result = fullfile(dir_experiment, model_desc.name, mat_result);

                if( model_desc.skip )
                    if( exist( path_result, 'file' ) )
                        model = load(path_result);
                        model = model.model;

                        disp('[Skip] already exist...');
                        fprintf('Iter = %d\n', model.eITER);

                        continue;
                    end
                end

                if isequal(model_desc.fn, @dppca)
                    model = model_desc.fn(mm_trans, Vp, Networks{idx_Network}.adj, M, ETA, THRESHd, m_init_ddup, objfreq_dj);
                elseif isequal(model_desc.fn, @sdppca)
                    model = model_desc.fn(mm_trans, Vp, Networks{idx_Network}.adj, M, ETA, model_desc.type, THRESHd, m_init_ddup, objfreq_dj, '');
                else
                    error(['[Error] Undefined Function - ' model_desc.name]);
                end

                model.W_GT = W;    
                fprintf('Iter = %d\n', model.eITER);
                save_parfor_model(path_result, model);
            end                
        end
    end
end
