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
                struct('name','cppca_miss', 'type','EM',    'fn',@cppca_em_m,'skip',1), ...
                struct('name','dppca_miss', 'type','',      'fn',@dppca_m   ,'skip',1), ...
                struct('name','sdppca_miss','type','bd75',  'fn',@sdppca_m  ,'skip',0), ...
                struct('name','sdppca_miss','type','bd50',  'fn',@sdppca_m  ,'skip',0), ...
                struct('name','sdppca_miss','type','unbd',  'fn',@sdppca_m  ,'skip',0), ...
              };
          
n_models = numel(models_desc);
idxes_model_cm = [1,2];
idxes_model_dm = [3,4,5];

format_dir_result   = 'result/synth_%s';
format_mat_result_cppca  = '%s_%s%2.2f_i%02d_%s.mat';
format_mat_result_dppca  = '%s_%s%2.2f_i%02d.mat';
format_mat_result_sdppca = '%s_%s%2.2f_i%02d_%s.mat';

% Frequency for intermediate object function value output
objfreq_c = 100;
objfreq_d1 = 100;
objfreq_dj = 100;

% Choose random seed: optional setting to reproduce numbers. You should be
% able to obtain consistent result without this although numbers may be
% different from those reported in the paper. In some cases, the D-PPCA
% objective function value may fluctuate near the stationary point and it
% may take a while to stabilize. However, they will converge in the end.
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
reset(s,0);

% 50 dimensional, 100 samples searching, 5 dimensional subspace
N = 100; D = 50; M = 5; VAR = 0.2;
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
m_init_cppca = get_init_value('cppca', mm_trans, M, 1);

% Set thresholds
THRESHc = 1e-5; % convergence precision (default: 1e-5)
THRESHd = 1e-5; % convergence precision (default: 1e-5)

%% MAR --------------------------------------------------------------------
disp('*** Different missing value rate (MAR) ***');

RATEArr = {0.01, 0.05, 0.1, 0.2, 0.3};

%parfor idk = 1:length(RATEArr)
for idk = 1:length(RATEArr)
    RATE = RATEArr{idk};
    
    for typearr_miss={'R','NR'}
        
        type_miss = cell2mat(typearr_miss);
        if strcmp( type_miss, 'R' )
            % We simply remove values randomly
            MissIDX = ones(D, N);
            MissIDX(randi(D*N, 1, ceil(D*N*RATE))) = 0;
        elseif strcmp( type_miss, 'NR')
            % We get band matrix: correlated missing values at non-diagonal
            MissIDX = gen_sfm_missing_mnar(mm_trans, RATE);
        else
            error('[Error] Undefined Missing Index Type');
        end

        for idr = 1:5
            for idx_model=1:n_models

                model_desc = models_desc{idx_model};
                if( strcmp(model_desc.name,'cppca') && strcmp(model_desc.type,'SVD') )
                    continue;
                end
                disp(['PPCA (' model_desc.name ' ' model_desc.type ')']);

                % result directory
                dir_result = sprintf(format_dir_result,model_desc.name);
                if ~exist(dir_result,'dir')
                    mkdir(dir_result);
                end

                % result mat file
                if strcmp(model_desc.type, '')
                    mat_result = sprintf(format_mat_result_dppca, model_desc.name, type_miss, RATE, idr);
                else
                    mat_result = sprintf(format_mat_result_cppca, model_desc.name, type_miss, RATE, idr, model_desc.type);
                end
                path_result = [dir_result '/' mat_result];

                % skip if it already exist
                if( model_desc.skip )
                    if( exist( path_result, 'file' ) )        
                        disp('already exist... skip');

                        tmp = load( path_result );
                        MissIDX = tmp.MissIDX;
                        continue;
                    end
                end    

                if isequal(model_desc.fn, @cppca_em_m)

                    % Run centralized
                    model = model_desc.fn(mm_trans, MissIDX, M, THRESHc, m_init_cppca, objfreq_c);
                    save_parfor_model_missidx(path_result, model, MissIDX);

                elseif isequal(model_desc.fn, @dppca_m) || isequal(model_desc.fn, @sdppca_m)

                    % Run distributed
                    ETA = 10;
                    NV = 5;
                    Earr = get_adj_graph(NV);
                    E = Earr{2};
                    Vp = get_sample_assign(NV, N, 0);

                    % To remove initialization effect, we duplicate centralized
                    % initialization parameters; Note that our interest here is dealing
                    % with missing values, not initialization. Moreover, it is
                    % possible for all nodes to have the same initialization values.
                    m_init_ddup = get_init_value('d_dup', mm_trans, M, 10, NV, m_init_cppca);

                    if isequal(model_desc.fn, @dppca_m )
                        model = dppca_m(mm_trans, MissIDX, Vp, E, M, ETA, THRESHd, m_init_ddup, objfreq_dj);
                    else
                        model = sdppca_m(mm_trans, MissIDX, Vp, E, M, ETA, model_desc.type ,THRESHd, m_init_ddup, objfreq_dj);
                    end
                    save_parfor_model_missidx(path_result, model, MissIDX);

                else
                    error('[Error] Undefined Model');
                end
            end
        end
    end
end

%% ------------------------------------------------------------------------
% Plot result
show_result_2_synth_dppca_m
