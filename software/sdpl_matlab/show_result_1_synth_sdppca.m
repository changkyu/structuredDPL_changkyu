function show_result_1_synth_sdppca(varargin)

fontsize = 18;
linetype = {'-', '--', ':', '-.'};
color = {'r', 'm', 'g', 'c', 'b', 'k' };
n_color = numel(color);

%%
models = cell(nargin,1);

for idx=1:nargin
    models{idx} = load(varargin{idx});
end

%% Figure 1. Overall
close all;

figure(1);
h = axes;
hold on;
hl = xlabel('iterations'); 
set(hl, 'FontSize',fontsize);
hl = ylabel('Objective function value'); 
set(hl, 'FontSize',fontsize);

min_val = inf;
max_val = -inf;
max_iter = -inf;

for idx=1:nargin
    min_val = min([models{idx}.model.objArray(1:models{idx}.model.eITER,end); min_val]);
    max_val = max([models{idx}.model.objArray(1:models{idx}.model.eITER,end); max_val]);
    max_iter = max(models{idx}.model.eITER, max_iter);

    plot(models{idx}.model.objArray(1:models{idx}.model.eITER,end), [linetype{idx} color{idx}],'LineWidth',2,'MarkerSize',7);
end

set(h,'XScale','log');
axis([0,max_iter, min_val, max_val]);
set(h,'FontSize',fontsize);

%% Figure 2. Each component

figure(2);
h = axes;
hold on;
n_node = size(models{end}.model.objArray,2)-1;

min_val = inf;
max_val = -inf;
max_iter = -inf;

for idx=1:nargin
    for n=1:n_node
        
        if( size(models{idx}.model.objArray,2) == 1 )
            continue;
        end
        
        min_val = min([models{idx}.model.objArray(1:models{idx}.model.eITER,n); min_val]);
        max_val = max([models{idx}.model.objArray(1:models{idx}.model.eITER,n); max_val]);
        max_iter = max(models{idx}.model.eITER, max_iter);
        
        plot(models{idx}.model.objArray(1:models{idx}.model.eITER,n), [linetype{idx} color{mod(n,n_color)+1}],'LineWidth',2,'MarkerSize',7);
    end
end
hold off;

legend({'dppca','sdppca unbd','sdppca bd'})

set(h,'XScale','log');
axis([0,max_iter, min_val, max_val]);
set(h,'FontSize',fontsize);
