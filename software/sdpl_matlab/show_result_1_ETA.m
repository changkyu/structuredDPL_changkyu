close all;

%linetype = {'-', '--', ':', '-.','-'};
linetype = {'-', '-', '-', '-','-'};
color = {'r', 'm', 'g', 'b', 'k', 'c' };

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
idxes_model_dm = [2];

format_dir_result   = 'result/synth_%s';
format_mat_result_cppca = '%s_%s.mat';
format_mat_result_dppca = '%s_N%02d_G%03d_E%02d.mat';
format_mat_result_sdppca = '%s_N%02d_G%03d_E%02d_%s.mat';

%%

%Varr = [5, 10, 20, 40 ];
Varr = [5];
ETA = 10;

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
            
            lgd = cell(NV,1);
            ETAij = res.model.ETAij_history;
            
            figure(1);
            for i=1:NV
                lgd{i} = sprintf('Node %d',i);
                subplot(NV,1,i);
                hold on;
                for j=1:NV
                    plot(1:res.model.eITER, ETAij(:,i,j), [color{j} linetype{j}]);
                end
                hold off;
                axis([1 15 5 15]);
            end
            
            subplot(NV,1,1);
            legend(lgd);
            title('Penalty Constraint Changes as Iterations');
            
            subplot(NV,1,NV);
            xlabel('Iteration');
            
            subplot(NV,1,ceil(NV/2));            
            ylabel('penalty constraint');
            
        end
        
    end
        
end

%%

iters = [1 2 3 4 5 6 7 8];
n_iters = length(iters);

NV = 5;
r = 10;
%angles = 0:0.01:2*pi;

figure(2);
clf;
for idx_iter=1:n_iters
iter = iters(idx_iter);
subplot(2,n_iters/2,idx_iter);
hold on;

    for i=1:NV    
        ci = cos((i-1)*(2*pi)/NV + pi/2);
        si = sin((i-1)*(2*pi)/NV + pi/2);

        xi = r*ci;
        yi = r*si;

        for j=1:NV
            if( i~=j && ETAij(iter,j,i) >= 10 )
                cj = cos((j-1)*(2*pi)/NV + pi/2);
                sj = sin((j-1)*(2*pi)/NV + pi/2);

                xj = r*cj;
                yj = r*sj;

                if( i < j )
                    %line([(xi+1/2*ci) (xj+1/2*cj)],[(yi+1/2*si) (yj+1/2*sj)],'color',color{i}, 'LineWidth', ETAij(iter,j,i)-9);
                    line([(xi) (xj)],[(yi) (yj)],'color',color{i}, 'LineWidth', ETAij(iter,j,i)-8);
                else
                    %line([(xi-1/2*ci) (xj-1/2*cj)],[(yi-1/2*si) (yj-1/2*sj)],'color',color{i}, 'LineWidth', ETAij(iter,j,i)-9);
                    line([(xi) (xj)],[(yi) (yj)],'color',color{i}, 'LineWidth', ETAij(iter,j,i)-8);
                end
            end        
        end

        %plot(xi+cos(angles), yi+sin(angles),color{i}, 'LineWidth', 2);
        h = plot(xi,yi,'o');
        set(h(1),'LineWidth',4,'MarkerEdgeColor',color{i},'MarkerFaceColor',color{i})
        text(xi+5*ci,yi+5*si,num2str(i),'color',color{i}, 'FontSize',12,'FontWeight','bold');
        set(gca,'XTick',[])
        set(gca,'XColor','k')
        set(gca,'YTick',[])
        set(gca,'YColor','w')    
    end
    xlabel(['iter = ' num2str(iter)]);
    hold off;
    axis([-15 16 -15 16]);
end

