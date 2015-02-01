%--------------------------------------------------------------------------
% SfM Caltech Turntable data demonstration
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

env_setting();

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
idxes_model_dm = [4];

format_dir_result   = 'result/sfm_caltech_%s';
format_mat_result_cppca = '%s_%s.mat';
format_mat_result_dppca = '%s_n%03d_r%03d_r%03d.mat';
format_mat_result_sdppca = '%s_n%03d_r%03d_r%03d_%s.mat';

%%
% Frequency for intermediate object function value output.
% We suppress both of them by default.
objfreq_c = 0;
objfreq_d = 0;

load('../../dataset/caltech_turntable/data.mat');
measurement_matrices = {BallSander, BoxStuff, Rooster, Standing, StorageBin};
datastr = {'BallSander', 'BoxStuff', 'Rooster', 'Standing', 'StorageBin'};

repeats = 1;%20;
SAs = zeros(length(datastr), 5, repeats);
SAs_sdppca = zeros(length(datastr), repeats, length(idxes_model_dm));

% Create result output directory if needed
if ~exist('result/sfm_real_caltech', 'dir');
    mkdir('result/sfm_real_caltech');
end

for idm = 1:length(measurement_matrices)
    disp(['[[ Current Model: ' datastr{idm} ' ]]']);
    mm_mat = measurement_matrices{idm};
    mm_mat_t = mm_mat';

    % get dimension of measurement matrix
    [D, N] = size(mm_mat);

    % Measurement matrix should be in the specific form
    if mod(D, 2) ~= 0
        error('Measurement matrix should be (2 x #frames)x(# points) form!');
    end

    % translate origin of measurement matrix to zero
    centroid = mean(mm_mat, 2);
    mm_trans = mm_mat - repmat(centroid, [1, N]);
    mm_trans_t = mm_trans';

    %--------------------------------------------------------------
    % Centralized setting
    [sfm_M, sfm_S, sfm_b, U3, W3, V3] = affine_sfm(mm_mat);
    clearvars U3 W3 V3;

    M = 3; % latent space dimension
    THRESHc = 1e-3; % convergence precision

    cm1 = cppca(mm_trans, M);

    m_init1 = get_init_value('cppca', mm_trans, M, 1);
    cm2 = cppca_em(mm_trans, M, THRESHc, m_init1, objfreq_c);

    m_init2 = get_init_value('sfm_c', mm_trans, M, 1, 0, 1, ones(size(mm_trans)));
    cm3 = cppca_em(mm_trans_t, M, THRESHc, m_init2, objfreq_c);

    % Print Result
    fprintf('*** Result ***\n');
    
    GT = sfm_S'; % We consider SVD SfM as ground truth

    fprintf(' ### Subspace angle (S/A)                 :1e-123456789    15\n');

    % Centralized settting result
    fprintf('SVD: (1) GT and (2) SVM SfM               : %.15f\n', ...
        subspace(GT, sfm_S'));
    fprintf('cm1: (1) GT and (2) PPCA (SVD)            : %.15f\n', ...
        subspace(GT, cm1.EZ'));
    fprintf('cm2: (1) GT and (2) PPCA (EM)             : %.15f\n', ...
        subspace(GT, cm2.EZ'));
    fprintf('cm3: (1) GT and (2) PPCA (EM) (Transposed): %.15f\n', ...
        subspace(GT, cm3.W));

    %--------------------------------------------------------------
    % Distributed setting
    ETA = 10; % learning parameter
    THRESHd = 1e-3; % convergence precision

    % Run with 5 nodes
    J = 5; % number of nodes

    %parfor idr2 = 1:repeats % independent runs (initial value)
    for idr2 = 1:repeats % independent runs (initial value)
        % 'sfm_d'  option fixes the variance initialization value
        % 'sfm_dr' option gives variable variance
        % NOTE: We reported best run using sfm_d but the performance may
        % vary a little depending on systems (up to 5 degrees). With fixed
        % value given here, one can see the angles actually better than we
        % reported. In all cases, as the final result video demonstrates,
        % the estimated 3D structure can be reasonably reconstructed.
        m_init4 = get_init_value('sfm_dr', mm_trans_t', M, 2, J, 0.5);

        % D-PPCA
        V = get_sample_assign(J, D, 1); % node assignment of samples(=frames)
        E = get_adj_graph(J); E = E{1}; % graph topology (1 = compete, 2 = ring)
        cm4 = dppca_t(mm_trans_t, V, E, M, ETA, THRESHd, m_init4, objfreq_d);

        % Distributed settting result: we report maximum (worst) angle
        max_angle = subspace(GT, cm4.W(:,:,1));
        for idj = 2:J
            if max_angle < subspace(GT, cm4.W(:,:,idj))
                max_angle = subspace(GT, cm4.W(:,:,idj));
            end
        end
        fprintf('cm4: (1) GT and (2) D-PPCA    (Nodes:%d)   : %.15f (Init: %d)\n', ...
            J, max_angle, idr2);

        save_var_parfor2(sprintf('result/sfm_real_caltech/tmp_%03d_%03d.mat', idm, idr2), ...
            cm4, max_angle);
        
        % SPPCA
        model_dm = cell(length(idxes_model_dm),1);
        for idx_model = idxes_model_dm
            model_desc = models_desc{idx_model};
            % result directory
            dir_result = sprintf(format_dir_result,model_desc.name);
            if ~exist(dir_result,'dir')
                mkdir(dir_result);
            end

            % result mat file
            if strcmp(model_desc.name,'sdppca')
                mat_result = sprintf(format_mat_result_sdppca, model_desc.name, idm, idr2, model_desc.type);
            else
                mat_result = sprintf(format_mat_result_dppca, model_desc.name, idm, idr2);
            end
            path_result = [dir_result '/' mat_result];

            if isequal(model_desc.fn, @dppca)
                model_dm{idx_model - idxes_model_dm(1) + 1} = model_desc.fn(mm_trans_t, V, E, M, ETA, THRESHd, m_init4);
            elseif isequal(model_desc.fn, @sdppca)
                model_dm{idx_model - idxes_model_dm(1) + 1} = model_desc.fn(mm_trans_t, V, E, M, ETA, model_desc.type, THRESHd, m_init4, 10000, 'fix_mu');
            else
                error(['[Error] Undefined Function - ' model_desc.name]);
            end

            % Distributed settting result
            max_angle = subspace(GT, model_dm{idx_model - idxes_model_dm(1) + 1}.W(:,:,1));
            for idj = 2:J
                if max_angle < subspace(GT, model_dm{idx_model - idxes_model_dm(1) + 1}.W(:,:,idj))
                    max_angle = subspace(GT, model_dm{idx_model - idxes_model_dm(1) + 1}.W(:,:,idj));
                end
            end

            save_parfor_model_init_angle(path_result, model_dm{idx_model - idxes_model_dm(1) + 1}, m_init4, max_angle);    
        end

    end % parfor: 20 independent runs (initial value)

    %----------------------------------------------------------
    % Store Result
    for idr2 = 1:repeats
        SAs(idm, 1, idr2) = subspace(GT, sfm_S');
        SAs(idm, 2, idr2) = subspace(GT, cm1.EZ');
        SAs(idm, 3, idr2) = subspace(GT, cm2.EZ');
        SAs(idm, 4, idr2) = subspace(GT, cm3.W);
        load(sprintf('result/sfm_real_caltech/tmp_%03d_%03d.mat', idm, idr2));
        SAs(idm, 5, idr2) = cm_init;
%         delete(sprintf('result/sfm_real_caltech/tmp_%03d_%03d.mat', idm, idr2));
        
        for idx_model = idxes_model_dm     
            model_desc = models_desc{idx_model};
            % result directory
            dir_result = sprintf(format_dir_result,model_desc.name);
            if ~exist(dir_result,'dir')
                mkdir(dir_result);
            end

            % result mat file
            if strcmp(model_desc.name,'sdppca')
                mat_result = sprintf(format_mat_result_sdppca, model_desc.name, idm, idr2, model_desc.type);
            else
                mat_result = sprintf(format_mat_result_dppca, model_desc.name, idm, idr2);
            end
            path_result = [dir_result '/' mat_result];

            tmp = load(path_result);
            SAs_sdppca(idm, idr2, idx_model - idxes_model_dm(1) + 1) = tmp.max_angle;
            clear tmp;
        end
    end

    tmp = mean(SAs, 3);
    fprintf('cm4: (1) GT and (2) D-PPCA                : %.15f (Mean of %d runs)\n', ...
        tmp(idm, 5), repeats);
    
    tmp = mean(SAs_sdppca, 2);
    for idx_model = idxes_model_dm     
        fprintf('cm4: (1) GT and (2) SD-PPCA               : %.15f (Mean of %d runs)\n', ...
        tmp(idm, idx_model - idxes_model_dm(1) + 1), repeats);
    end
    
    %--------------------------------------------------------------
    % Save the model for future usage
    ms_path = sprintf('result/sfm_real_caltech/result_%s.mat', datastr{idm});
%     save(ms_path, 'sfm_M', 'sfm_S', 'sfm_b', 'cm2', 'cm3');
    save(ms_path, 'sfm_M', 'sfm_S', 'sfm_b', 'cm2', 'cm3', 'cm4', 'model_dm');
end

save('result/sfm_real_caltech/result_sfm_caltech_all.mat','SAs');
save('result/sfm_real_caltech/result_sfm_caltech_sdppca_all.mat','SAs_sdppca');

clearvars -except SAs;

% In order to see these, you need to save cm4 above separately as 'dppca_*'
%show_result_4_caltech;

% Compute mean and variance of angles
meanSAs = mean(SAs,3) * 180 / pi;
varSAs = var(SAs,1,3) * 180 / pi;

format LONG;
