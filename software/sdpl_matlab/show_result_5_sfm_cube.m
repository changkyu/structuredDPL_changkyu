%--------------------------------------------------------------------------
% SfM Synthetic data result plot (Cube)
%
% Implemented/Modified
%  by     Changkyu Song (changkyu.song@rutgers.edu)
%  on     2014.12.05 (last modified on 2014/12/15)
%--------------------------------------------------------------------------

%% Experiments Description
models_desc = { 
                struct('name','cppca', 'type','SVD',   'fn',@cppca   ,'skip',0), ...
                struct('name','cppca', 'type','EM',    'fn',@cppca_em,'skip',0), ...
                struct('name','dppca', 'type','',      'fn',@dppca   ,'skip',0), ...
                struct('name','sdppca','type','bd50',  'fn',@sdppca  ,'skip',0), ...
                %struct('name','sdppca','type','unbd',  'fn',@sdppca  ,'skip',0), ...
                %struct('name','sdppca','type','bd75',  'fn',@sdppca  ,'skip',0), ...                
              };
          
n_models = numel(models_desc);
idxes_model_cm = [1,2];
idxes_model_dm = [4];

format_dir_result   = 'result/sfm_cube_nodenoise_%s';
format_mat_result_cppca = '%s_%s.mat';
format_mat_result_dppca = '%s_%s%s_r%03d.mat';
format_mat_result_sdppca = '%s_%s%s_r%03d_%s.mat';

%%

fontsize = 14;

load('result/sfm_cube_nodenoise/result_cube_N005_idx020_all.mat');

size_SAs = size(SAs);
SAs_sdppca = zeros(5, 20,1, length(idxes_model_dm));
for idx_model = idxes_model_dm     
    model_desc = models_desc{idx_model};    
    
    % result directory
    dir_result = sprintf(format_dir_result,model_desc.name);
    if ~exist(dir_result,'dir')
        mkdir(dir_result);
    end
    
    % result mat file
    if strcmp(model_desc.name,'sdppca')
        mat_result = sprintf('result_cube_%s_N005_idx020_%s.mat', model_desc.name, model_desc.type);
    else
        mat_result = sprintf('result_cube_%s_N005_idx020.mat', model_desc.name);
    end
    path_result = [dir_result '/' mat_result];
    
    sv = load(path_result);
    SAs_sdppca(:,:,:,idx_model - idxes_model_dm(1) + 1) = sv.SAs;
end

% Get average over runs w/ different initializations
SAx = mean(SAs, 4);
SAx_sdppca = mean(SAs_sdppca, 3);

% Separate out each angles
SAx1 = SAx(:,1,:);
SAx2 = SAx(:,2,:);

% Prepare matrix to plot - each one has (noises) x (runs)
SAx1t = zeros(20, 5);
SAx2t = zeros(20, 5);
SAx_sdppca_t = zeros(20,5,length(idxes_model_dm));
for idn = 1:size(SAx,1)
    for idr = 1:size(SAx,3)
        SAx1t(idr, idn) = SAx1(idn,1,idr) * 180 / pi;
        SAx2t(idr, idn) = SAx2(idn,1,idr) * 180 / pi;
        
        for idx_model = idxes_model_dm     
            SAx_sdppca_t(idr,idn,idx_model - idxes_model_dm(1) + 1) =  SAx_sdppca(idn,idr,1,idx_model - idxes_model_dm(1) + 1) * 180 / pi;
        end
    end
end
    
% Prepare vectors to plot - mean of each box (noises) x 1
SAx1c = mean(SAx1t, 1);
SAx2c = mean(SAx2t, 1);

%%
% Prepare figure
figure;
h = axes;
hold on;
hl = xlabel('Noise level','FontSize',30); 
set(hl, 'FontSize',fontsize);
hl = ylabel('Subspace angle (degree)'); 
set(hl, 'FontSize',fontsize);
set(h,'FontSize',fontsize);

% Plot Boxes
boxplot(SAx1t,...
    'colors','k','symbol','g+','Position',1:1:5,'widths',0.2); 
set(gca,'XTickLabel',{' '});
boxplot(SAx2t,...
    'colors','b','Position',1.3:1:5.3,'widths',0.2); 

for idx_model = 1
    boxplot(SAx_sdppca_t(:,:,idx_model),...
    'colors','r','symbol','r+','Position',(1+0.3*(2)):1:(5+0.3*(2)),'widths',0.2); 
    set(gca,'XTickLabel',{' '});
end

%axis([0,6,0,1.6]);
title('Subspace Angle vs. Ground Truth');

hLegend = legend(findall(gca,'Tag','Box'), {'Centralized','DPPCA','S-DPPCA (Ours)'});
hChildren = findall(get(hLegend,'Children'), 'Type','Line');
% Set the horizontal lines to the right colors
set(hChildren(6),'Color','k')
set(hChildren(4),'Color','b')
set(hChildren(2),'Color','r')

drawnow;
