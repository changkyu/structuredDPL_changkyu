%--------------------------------------------------------------------------
% SfM Synthetic data demonstration (Cube)
%
% Implemented/Modified
%  by     Changkyu Song (changkyu.song@rutgers.edu)
%  on     2014.12.05 (last modified on 2014/12/15)
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
idxes_model_dm = [4];

format_dir_result   = 'result/sfm_cube_nodenoise_%s';
format_mat_result_cppca = '%s_%s.mat';
format_mat_result_dppca = '%s_%s%s_r%03d.mat';
format_mat_result_sdppca = '%s_%s%s_r%03d_%s.mat';

%%

model = 'cube';
type = 'nodenoise';
% noises = [
%     0.10, 0.01, 0.01, 0.01, 0.01;
%     0.01, 0.10, 0.01, 0.01, 0.01;
%     0.01, 0.01, 0.10, 0.01, 0.01;
%     0.01, 0.01, 0.01, 0.10, 0.01;
%     0.01, 0.01, 0.01, 0.01, 0.10;
% ];
noises = [
    1.00, 0.01, 0.01, 0.01, 0.01;
    0.01, 1.00, 0.01, 0.01, 0.01;
    0.01, 0.01, 1.00, 0.01, 0.01;
    0.01, 0.01, 0.01, 1.00, 0.01;
    0.01, 0.01, 0.01, 0.01, 1.00;
];

if ~exist(sprintf('data/%s/%s', model,type), 'dir')
    error('Please run cube data generation script first!');
elseif ~exist(sprintf('result/sfm_%s_%s', model, type), 'dir')
    mkdir(sprintf('result/sfm_%s_%s', model, type));
end

datagens = 20; % Number of independent runs of perturbed data
repeats = 1; % Number of independent runs of different initial values

% Run with 5 nodes
J = 5; % number of nodes

SAs = zeros(size(noises,2), 2, datagens, repeats);
SAs_sdppca = zeros(size(noises,2), datagens, repeats, length(idxes_model_dm));

for idn = 1:size(noises,2) % noise level
    %--------------------------------------------------------------
    % Data
    % Frequency for intermediate object function value output
    objfreq_c = 0;
    objfreq_d = 0;

    measurement_matrices = cell(20, 1);
    for idr = 1:datagens
        
        dir_data = sprintf('data/%s/%s/',model, type);
        prefix_data = sprintf('N%03d_idx%03d', J, idr);
        str_noise = '';
        for r=1:J
            str_noise = [str_noise, sprintf('_%.2f',noises(r,idn))];
        end
        format_save = [dir_data prefix_data, str_noise, '.mat'];
        load(format_save);
        measurement_matrices{idr} = measurement_matrix;
    %end
    %
    %for idr = 1:datagens % independent runs (noise level)
        mm_mat = measurement_matrices{idr};
        mm_mat_t = mm_mat';

        % get dimension of measurement matrix
        [D, N] = size(mm_mat);

        % Measurement matrix should be in the specific form
        if mod(D, 2) ~= 0
            error('Measurement matrix should be (2 x #frames)x(# points) form!');
        end

        % translate origin of measurement matrix to zero
        % NOTE: In the real setting, the mean over columns are actually
        %  the mean over rows as we transposed. Thus, there's no difference
        %  in actual implementation as we have access to all points.
        centroid = mean(mm_mat, 2);
        mm_trans = mm_mat - repmat(centroid, [1, N]);
        mm_trans_t = mm_trans';

        %--------------------------------------------------------------
        % Centralized setting
        [sfm_M, sfm_S, sfm_b, U3, W3, V3] = affine_sfm(mm_mat);

        M = 3; % latent space dimension
        THRESHc = 1e-3; % convergence precision

        % Print Result
        fprintf('*** Result (Run: %d) ***\n', idr);

        % We consider centralized SVD solution as ground truth
        fprintf('### Subspace angle (S/A)             :1e-123456789    15\n');

        % Centralized settting result
        fprintf('(1) GT and (2) SVD SfM               : %.15f\n', ...
            subspace(GT, sfm_S'));

        %--------------------------------------------------------------
        % Distributed setting
        ETA = 10; % learning parameter
        THRESHd = 1e-3; % convergence precision

        %parfor idr2 = 1:repeats % independent runs (initial value)
        for idr2 = 1:repeats % independent runs (initial value)
            m_init = get_init_value('sfm_dr', mm_trans_t', M, 10, J, 0.5);

            % D-PPCA
            V = get_sample_assign(J, D, 1); % node assignment of samples(=frames)
            E = get_adj_graph(J); E = E{1}; % graph topology (1 = compete, 2 = ring)
            cm_dppca = dppca_t(mm_trans_t, V, E, M, ETA, THRESHd, m_init, objfreq_d);

            % Distributed settting result
            max_angle = subspace(GT, cm_dppca.W(:,:,1));
            for idj = 2:J
                if max_angle < subspace(GT, cm_dppca.W(:,:,idj))
                    max_angle = subspace(GT, cm_dppca.W(:,:,idj));
                end
            end

            fprintf('(1) GT and (2) D-PPCA    (Nodes:%d)   : %.15f (Init: %02d / %f)\n', ...
                J, max_angle, idr2, m_init.VAR);

            save_var_parfor1(sprintf('result/sfm_%s_%s/angle_%s%s_%03d.mat', ...
                model, type, prefix_data, str_noise, idr2), max_angle);
            save_var_parfor2(sprintf('result/sfm_%s_%s/result_%s%s_%03d.mat', ...
                model, type, prefix_data, str_noise, idr2), cm_dppca, m_init);
            
            % SPPCA
            for idx_model = idxes_model_dm
                model_desc = models_desc{idx_model};
                % result directory
                dir_result = sprintf(format_dir_result,model_desc.name);
                if ~exist(dir_result,'dir')
                    mkdir(dir_result);
                end

                % result mat file
                if strcmp(model_desc.name,'sdppca')
                    mat_result = sprintf(format_mat_result_sdppca, model_desc.name, prefix_data, str_noise, idr2, model_desc.type);
                else
                    mat_result = sprintf(format_mat_result_dppca, model_desc.name, prefix_data, str_noise, idr2);
                end
                path_result = [dir_result '/' mat_result];
                
                if isequal(model_desc.fn, @dppca)
                    model_dm = model_desc.fn(mm_trans_t, V, E, M, ETA, THRESHd, m_init);
                elseif isequal(model_desc.fn, @sdppca)
                    model_dm = model_desc.fn(mm_trans_t, V, E, M, ETA, model_desc.type, THRESHd, m_init, 10000, 'fix_mu');
                else
                    error(['[Error] Undefined Function - ' model_desc.name]);
                end
                
                % Distributed settting result
                max_angle = subspace(GT, model_dm.W(:,:,1));
                for idj = 2:J
                    if max_angle < subspace(GT, model_dm.W(:,:,idj))
                        max_angle = subspace(GT, model_dm.W(:,:,idj));
                    end
                end
                                
                save_parfor_model_init_angle(path_result, model_dm, m_init, max_angle);    
            end
        end % parfor: independent runs (initial value)

        %----------------------------------------------------------
        % Store Result
        for idr2 = 1:repeats
            SAs(idn, 1, idr, idr2) = subspace(GT, sfm_S');
            load(sprintf('result/sfm_%s_%s/angle_%s%s_%03d.mat', ...
                model, type, prefix_data, str_noise, idr2));
            SAs(idn, 2, idr, idr2) = cm;
            delete(sprintf('result/sfm_%s_%s/angle_%s%s_%03d.mat', ...
                model, type, prefix_data, str_noise, idr2));
            
            for idx_model = idxes_model_dm     
                model_desc = models_desc{idx_model};
                % result directory
                dir_result = sprintf(format_dir_result,model_desc.name);
                if ~exist(dir_result,'dir')
                    mkdir(dir_result);
                end

                % result mat file
                if strcmp(model_desc.name,'sdppca')
                    mat_result = sprintf(format_mat_result_sdppca, model_desc.name, prefix_data, str_noise, idr2, model_desc.type);
                else
                    mat_result = sprintf(format_mat_result_dppca, model_desc.name, prefix_data, str_noise, idr2);
                end
                path_result = [dir_result '/' mat_result];
                
                tmp = load(path_result);
                SAs_sdppca(idn, idr, idr2, idx_model - idxes_model_dm(1) + 1) = tmp.max_angle;
                
            end            
        end
    end % for: independent runs (noise)
end % noise level

save(sprintf('result/sfm_%s_%s/result_cube_%s_all.mat', model,type, prefix_data), 'SAs');

%%
for idx_model = idxes_model_dm     
    model_desc = models_desc{idx_model};
    SAs = SAs_sdppca(:,:,:, idx_model - idxes_model_dm(1) + 1);
    
    % result directory
    dir_result = sprintf(format_dir_result,model_desc.name);
    if ~exist(dir_result,'dir')
        mkdir(dir_result);
    end
    
    % result mat file
    if strcmp(model_desc.name,'sdppca')
        mat_result = sprintf('result_cube_%s_%s_%s.mat', model_desc.name, prefix_data, model_desc.type);
    else
        mat_result = sprintf('result_cube_%s_%s.mat', model_desc.name, prefix_data);
    end
    path_result = [dir_result '/' mat_result];
    
    save(path_result, 'SAs');    
end

show_result_5_sfm_cube;
