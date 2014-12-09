function [model_init] = get_init_value_ex(varargin)
% GET_INIT_VALUE    Get initial value model for PPCA (extension)
%
% INPUT
%   TYPE       One of model types below
%              sdppca    - General SD-PPCA parameters
%              sdppca_t  - General SD-PPCA parameters (transposed)
%              sfm_sd    - SfM parameters for SD-PPCA
%              sfm_sdr   - SfM parameters for SD-PPCA (w/ random noise)
%              
%              (From 'S. Yoon NIPS 2012' code)
%              d_dup    - Simple duplicate for given centralized parameters
%              cppca    - General centralized PPCA parameters
%              dppca    - General D-PPCA parameters
%              sfm_c    - SfM parameters for PPCA (transposed)
%              cppca_t  - General centralized PCCA (transposed) parameters
%              dppca_t  - General D-PPCA parameters (transposed)
%              sfm_d    - SfM parameters for D-PPCA
%              sfm_dr   - SfM parameters for D-PPCA (w/ random noise)
%
%   MM_TRANS   Measurement matrix. For SfM, its size should be in the form 
%              of (twice #frames) x (#points)
%   M          Scalar value for latent space dimension
%   VARI       Variance factor for random variance
%   J          (Optional for distributed cases) Number of nodes 
%   D_PARAM    (Optional for distributed cases) Special parameter
%              If (TYPE = sfm_d),  it's offset for minimum variance
%              If (TYPE = sfm_dj), it's vector for minimum variance
%              If (TYPE = d_dup),  it's initial model to be duplicated
%   MissIDX    (Optional for missing values) Missing value index matrix
%
% OUTPUT
%   MODEL_INIT Model structure with appropriate initial parameters
%
% Implemented/Modified
%  by     Changkyu Song (changkyu.song@cs.rutgers.edu)
%  on     2014.11.07 (last modified on 2014/11/07)

if( nargin >= 4 )
    type = varargin{1};mm_trans = varargin{2};M = varargin{3};vari = varargin{4};
if( nargin >= 5 )
    J = varargin{5};
if( nargin >= 6 )
    D_PARAM = varargin{6};
if( nargin >= 7 )
    MissIDX = varargin{7};
end
end
end
else
    fprintf('[changkyu] Error: Not enough arguments\n');    
    return
end

if strcmp(type, 'sdppca')
    %----------------------------------------------------------------------
    % Structured Distributed PPCA
    %----------------------------------------------------------------------
    if nargin < 5
        error('Did not specified number of nodes!');
    end
    % get number of frames
    FF = size(mm_trans, 1);
    % W is initialized as stack of homogeneous coordiante camera matrices
    model_init.W   = repmat(rand(FF,M)/100,[1 1 J]);
    % MU should be initialized as zero
    model_init.MU  = zeros(FF, 1);
    % VAR is just a small number; enough variance needed for first hop
    model_init.VAR = 100;
elseif strcmp(type, 'sdppca_t')
    %----------------------------------------------------------------------
    % Structured Distributed PCCA
    %----------------------------------------------------------------------
    if nargin < 5
        error('Did not specified number of nodes!');
    end
    % calculate frames per node
    N = size(mm_trans, 2);
    % W is initialized x-y coordinates of the first frame AT EACH NODE
    model_init.W = rand(N, M, J); 
    % MU should be initialized as zero
    model_init.MU = zeros(N, 1);
    % VAR is just a small number
    model_init.VAR = rand(1);
elseif strcmp(type, 'sfm_sd')
    %----------------------------------------------------------------------
    % Structured Distributed PPCA for SfM
    %----------------------------------------------------------------------
    if nargin < 5
        error('Did not specified number of nodes!');
    elseif nargin < 6
        error('Please specify minimum variance!');
    end
    % calculate frames per node
    [FF, N] = size(mm_trans);
    fpnode = floor(FF / (2 * J)); 
    % W is initialized x-y coordinates of the first frame AT EACH NODE
    % with small variance from uniform distribution scaled by some value
    model_init.W = zeros(N, 3, J); 
    for idj = 1:J
        past_frames = (fpnode * 2) * (idj - 1);
        model_init.W(:,:,idj) = ...
            [ mm_trans( past_frames + 1 : past_frames + 2, : ) + ones(2,N)/vari; ones(1,N)]';
    end
    % MU should be initialized as zero
    model_init.MU = zeros(N, 1);
    % VAR is just a small number
    model_init.VAR = D_PARAM;
elseif strcmp(type, 'sfm_sdr')
    %----------------------------------------------------------------------
    % Structured Distributed PPCA for SfM (random noise - each node has different one)
    %----------------------------------------------------------------------
    if nargin < 5
        error('Did not specified number of nodes!');
    elseif nargin < 6
        error('Please specify minimum variance!');
    end
    % calculate frames per node
    [FF, N] = size(mm_trans);
    fpnode = floor(FF / (2 * J)); 
    % W is initialized x-y coordinates of the first frame AT EACH NODE
    % with small variance from uniform distribution scaled by some value
    model_init.W = zeros(N, 3, J); 
    for idj = 1:J
        past_frames = (fpnode * 2) * (idj - 1);
        model_init.W(:,:,idj) = ...
            [ mm_trans( past_frames + 1 : past_frames + 2, : ) + ones(2,N)/vari; ones(1,N)]';
    end
    % MU should be initialized as zero
    model_init.MU = zeros(N, 1);
    % VAR is just a small number
    model_init.VAR = rand(1)/vari + D_PARAM;
else
    model_init = get_init_value(varargin{:});
end

end

