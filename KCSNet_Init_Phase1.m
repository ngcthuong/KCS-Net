function net = KCSNet_Init_Phase2(ref_net)
global featureSize noLayer blkSize subRate isLearnMtx;

test = 0;
if test == 1
    featureSize = 64;
    noLayer = 7; 
    blkSize = 32; 
    subRate = 0.1; 
end

noMeas = round(sqrt(subRate) * blkSize); 

%%% 17 layers
b_min = 0.025;
lr11  = [1 1];
lr10  = [1 0];
lr00  = [0 0];
weightDecay = [1 0];
meanvar  =  [zeros(featureSize,1,'single'), 0.01*ones(featureSize,1,'single')];

% Define network
net.layers = {} ;

%% 1. Sampling layer - for gray image 
% Sampling network, with kernel size of blkSize x blkSize, do no use
% bias --> initialized as zero and learn rate = 0. 

% Load sensing matrix of size blkSizexBlkSize 
trial = 1; 
fileName = ['SensingMtxs\BlkSize' num2str(blkSize) '_trial' num2str(trial) '.mat' ];
if ~(exist(fileName))
    Phi_Full1 = orth(rand(blkSize, blkSize));
    Phi_Full2 = orth(rand(blkSize, blkSize));
    save(fileName, 'Phi_Full1', 'Phi_Full2'); 
else
    load(fileName); 
    Phi1 = single(Phi_Full1(1:noMeas, :)); 
    Phi2 = single(Phi_Full2(1:noMeas, :))'; 
end

%% KCS sampling 
% Step 1. Vertical Sampling
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{zeros(blkSize, 1, 1, noMeas,'single'), zeros(noMeas,1,'single')}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'dilate',1, ...
    'learningRate', isLearnMtx, ...
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;
% net.layers{end+1} = struct('type', 'relu','leak',0) ; -- do not use relu 

% assign the sampling matrix
W = zeros(blkSize, 1, 1, noMeas); 
for i = 1:1:noMeas
    W(:,1, 1, i) = Phi1(i, :); 
end
net.layers{end}.weights(1) = {single(W)}; 

% Step 2. Reshize vertical 
net.layers{end+1} = struct('type', 'reshape_ver');

% Step 3. Horizontal sampling
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{zeros(1, blkSize, 1, noMeas,'single'), zeros(noMeas,1,'single')}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'dilate',1, ...
    'learningRate', isLearnMtx, ...
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;

% assign the sampling matrix
W = zeros(1, blkSize, 1, noMeas); 
for i = 1:1:noMeas
    W(1, :, 1, i) = Phi2(:, i); 
end
net.layers{end}.weights(1) = {single(W)}; 

% reshape 
net.layers{end+1} = struct('type', 'reshape_hor');

%% 2. Initial reconstruction layer with 1x1 Convolution 
% Step 1. Vertical inverse
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{zeros(noMeas, 1, 1, blkSize,'single'), zeros(blkSize,1,'single')}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'dilate',1, ...
    'learningRate',lr10, ...
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;
% assign the sampling matrix
W = zeros(noMeas, 1, 1, blkSize); 
Phi1T = Phi1';
for i = 1:1:blkSize
    W(:, 1, 1, i) = Phi1T(i, :); 
end
net.layers{end}.weights(1) = {single(W)}; 

% Step 2. Reshape 
net.layers{end+1} = struct('type', 'reshape_ver');

% Step 3. Horizontal inverse
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{zeros(1, noMeas, 1, blkSize,'single'), zeros(blkSize,1,'single')}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'dilate',1, ...
    'learningRate',lr10, ...
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;

% assign the sampling matrix
W = zeros(1, noMeas, 1, blkSize); 
Phi2T = Phi2'; 
for i = 1:1:blkSize
    W(1, :, 1, i) = Phi2T(:, i); 
end
net.layers{end}.weights(1) = {single(W)}; 

% reshape 
net.layers{end+1} = struct('type', 'reshape_hor');

% %% 3. Reconstruction network - DnCNN 
% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {{sqrt(2/(9*featureSize))*randn(3,3,1,featureSize,'single'), zeros(featureSize,1,'single')}}, ...
%     'stride', 1, ...
%     'pad', 1, ...
%     'dilate',1, ...
%     'learningRate',lr11, ... 
%     'weightDecay',weightDecay, ...
%     'opts',{{}}) ;
% net.layers{end+1} = struct('type', 'relu','leak',0) ;
% 
% for i = 1:1:noLayer
%     
%     net.layers{end+1} = struct('type', 'conv', ...
%         'weights', {{sqrt(2/(9*featureSize))*randn(3,3,featureSize,featureSize,'single'), zeros(featureSize,1,'single')}}, ...
%         'stride', 1, ...
%         'learningRate', lr10, ...
%         'dilate',1, ...
%         'weightDecay',weightDecay, ...
%         'pad', 1, 'opts', {{}}) ;
%     
%     net.layers{end+1} = struct('type', 'bnorm', ...
%        'weights', {{clipping(sqrt(2/(9*64))*randn(64,1,'single'),b_min), zeros(64,1,'single'),meanvar}}, ...
%        'learningRate', [1 1 1], ...
%        'weightDecay', [0 0], ...
%        'opts', {{}}) ;
%     
%     net.layers{end+1} = struct('type', 'relu','leak',0) ;
%     
%     
%     
% end
% 
% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {{sqrt(2/(9*featureSize))*randn(3,3,featureSize,1,'single'), zeros(1,1,'single')}}, ...
%     'stride', 1, ...
%     'learningRate', lr11, ...
%     'dilate',1, ...
%     'weightDecay',weightDecay, ...
%     'pad', 1, 'opts', {{}}) ;

net.layers{end+1} = struct('type', 'loss') ; % make sure the new 'vl_nnloss.m' is in the same folder.

% Fill in default values
net = vl_simplenn_tidy(net);



function A = clipping(A,b)
A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;




