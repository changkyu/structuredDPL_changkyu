resize_factor = 0.5;
% load('result/sfm_real_caltech/result_sfm_caltech_all.mat');
load('result/sfm_real_caltech/result_sfm_caltech_sdppca_all.mat');
VARs = var(SAs, 1, 3);
SAs = mean(SAs, 3);

GTs = load('../../dataset/caltech_turntable/data.mat');
GTs = {GTs.BallSander, GTs.BoxStuff, GTs.Rooster, GTs.Standing, GTs.StorageBin};
CENTs = cell(length(GTs),1);
for idm = 1:length(GTs)
    CENTs{idm} = mean(GTs{idm}, 2);
    GTs{idm} = round(GTs{idm}*resize_factor);
end
PTs = {size(GTs{1},2), size(GTs{2},2), size(GTs{3},2), size(GTs{4},2), size(GTs{5},2),};
obj_name = {'BallSander', 'BoxStuff', 'Rooster', 'Standing', 'StorageBin'};

if ~exist('video/caltech', 'dir')
    mkdir('video/caltech');
end

for idm = 1:length(obj_name)
    % Print object's subspace angle info
    disp(['Object: ' obj_name{idm}]);
    fprintf('Mean     of maximum subspace angle vs. SVD-based SfM: %3.5f\n', (SAs(idm, 5)*180/pi));
    fprintf('Variance of maximum subspace angle vs. SVD-based SfM: %3.5f\n', (VARs(idm, 5)*180/pi));
    
    % Prepare window and recording
    close gcf;
    scrsz = get(0,'ScreenSize');
    figure('Position',[scrsz(3)/2-240 scrsz(4)/2-320 640 480]);
    winsize = get(gcf, 'Position');
    writerObj = VideoWriter(sprintf('video/caltech/%s.avi', obj_name{idm}));
    writerObj.FrameRate = 3; % second pause
    writerObj.Quality = 100;
    open(writerObj);
    set(gca, 'NextPlot', 'replacechildren');
    set(gcf, 'Renderer', 'zbuffer');

    % Load D-PPCA model
    DPPCA = load(sprintf('result/sfm_real_caltech/result_%s.mat', obj_name{idm}));
    DPPCA = DPPCA.cm4;

%     % Load SD-PPCA model
%     DPPCA = load(sprintf('result/sfm_real_caltech/result_%s.mat', obj_name{idm}));
%     DPPCA = DPPCA.model_dm{1};


    % Compute frames in each node
    FRAMEs = zeros(size(DPPCA.W, 3), 1);
    FRAMEs(1) = size(DPPCA.EZ{1},2);
    for idj = 2:size(DPPCA.W, 3)
        FRAMEs(idj) = FRAMEs(idj -1) + size(DPPCA.EZ{idj},2);
    end
    
    % Compute reprojected points
    Est3D = mean(DPPCA.W, 3);
    EstCam = [];
    RPPTs = [];
    for idj = 1:size(DPPCA.W,3)
        EstCam = [EstCam DPPCA.EZ{idj}];
        RPPTs = [RPPTs (DPPCA.W(:,:,idj) * DPPCA.EZ{idj})];
    end
    RPPTs = RPPTs' + repmat(CENTs{idm}, [1 PTs{idm}]);
    RPPTs = round(RPPTs*resize_factor);

    % Make homogeneous coordinates
    Est3Dh = [Est3D ones(size(Est3D, 1), 1)]';
    EstCamh = permute(reshape(EstCam', [2, size(EstCam, 2)/2, 3]), [1 3 2]);
    EstCamh(3, 3, :) = 1; EstCamh(3, 4, :) = 0; 
    EstCamh(1:2, 4, :) = reshape(CENTs{idm}, 2, 1, size(EstCam, 2)/2);
    
    % Compute world-coordinate projected 3D points
    frames = size(GTs{idm},1)/2;
    Est3Dhp = zeros(3, PTs{idm}, frames);
    for idf = 1:frames
        Est3Dhp(:,:,idf) = EstCamh(:,:,idf) * Est3Dh;
    end    
    Est3Dhp(1:2,:,:) = Est3Dhp(1:2,:,:) * resize_factor;
    
    disp('Loading images...')
    IMGs = cell(frames, 1);
    for idf = 1:frames
        filename = sprintf('../../dataset/caltech_turntable/%s/img_%04d.PNG', obj_name{idm}, idf);
        IMGs{idf} = repmat(rgb2gray(imresize(imread(filename), resize_factor)), [1 1 3]);
    end

    idj = 1; % starting node
    for idf = 1:frames
        clf;
        
        % prepare image (convert to grayscale)
        I = IMGs{idf};

        % put GT points and reprojected points
        for idp = 1:PTs{idm}
            pt_x = GTs{idm}(idf*2-1,idp);
            pt_y = GTs{idm}(idf*2,idp);

            I(pt_y,pt_x,1) = 0;
            I(pt_y,pt_x,2) = 255;
            I(pt_y,pt_x,3) = 0;
            I(pt_y,pt_x+1,1) = 0;
            I(pt_y,pt_x+1,2) = 255;
            I(pt_y,pt_x+1,3) = 0;
            I(pt_y,pt_x-1,1) = 0;
            I(pt_y,pt_x-1,2) = 255;
            I(pt_y,pt_x-1,3) = 0;
            I(pt_y-1,pt_x,1) = 0;
            I(pt_y-1,pt_x,2) = 255;
            I(pt_y-1,pt_x,3) = 0;
            I(pt_y+1,pt_x,1) = 0;
            I(pt_y+1,pt_x,2) = 255;
            I(pt_y+1,pt_x,3) = 0;
            
            pt_x = RPPTs(idf*2-1,idp);
            pt_y = RPPTs(idf*2,idp);

            I(pt_y,pt_x,1) = 255;
            I(pt_y,pt_x,2) = 0;
            I(pt_y,pt_x,3) = 255;
            I(pt_y,pt_x+1,1) = 255;
            I(pt_y,pt_x+1,2) = 0;
            I(pt_y,pt_x+1,3) = 255;
            I(pt_y,pt_x-1,1) = 255;
            I(pt_y,pt_x-1,2) = 0;
            I(pt_y,pt_x-1,3) = 255;
            I(pt_y-1,pt_x,1) = 255;
            I(pt_y-1,pt_x,2) = 0;
            I(pt_y-1,pt_x,3) = 255;
            I(pt_y+1,pt_x,1) = 255;
            I(pt_y+1,pt_x,2) = 0;
            I(pt_y+1,pt_x,3) = 255;
        end

        % create color map
        [h w d] = size(I);
        cmap = cast(reshape(I, [h*w 3]),'double')./255;
        indexes = 1:(w*h);
        I = reshape(indexes, [h w]);

        % generate mesh
        [X,Y] = meshgrid(1:size(I,2), 1:size(I,1));
        Z = zeros(size(I,1),size(I,2));

        % plot each slice as a texture-mapped surface
        surf('XData',X-(w/2), 'YData',-Z+15, 'ZData',-Y+(h/2), ...
            'CData',I(:,:), 'CDataMapping','direct', ...
            'EdgeColor','none', 'FaceColor','texturemap')
        colormap(cmap);
        hold on;

        % plot points
        plot3(Est3Dhp(1,:,idf)-(w/2), -Est3Dhp(3,:,idf), -Est3Dhp(2,:,idf)+(h/2), ...
            'r+', 'MarkerSize', 7);
        
        % which node are we in?
        if FRAMEs(idj) < (idf*2)
            idj = idj + 1;
        end

        % put title and wait...
        title(sprintf('%s (Total Frame: %d/%d, Camera %d)', ...
            obj_name{idm}, idf, frames, idj), ...
            'FontWeight', 'bold');
        xlabel('x');
        ylabel('z');
        zlabel('y');
        if idm == 1
            view([30 35]);
        else
            view([-25 35]);
        end
        pause(.05);
        
        % record movie
        writeVideo(writerObj, getframe(gcf));
    end
    
    close(writerObj);
end

close gcf;