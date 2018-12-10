
%%% Note: run the 'GenerateTrainingPatches.m' to generate
%%% training data (clean images) first.
addpath('E:\matConvNet\matconvnet-1.0-beta25\matlab\mex');
addpath('E:\matConvNet\matconvnet-1.0-beta25\matlab\simplenn');

% addpath('E:\matConvNet\matconvnet-1.0-beta25\matlab');
addpath('utilities');

rng('default')

global featureSize noLayer subRate blkSize isLearnMtx; %%% noise level

featureSize = 64;
noLayer = 5;
subRate = 0.01;
blkSize = 512;
isLearnMtx = [1 0];
batSize = 16;

%%%-------------------------------------------------------------------------
%%% Configuration
%%%-------------------------------------------------------------------------
opts.modelName        = ['KCSNet_Phase2' num2str(noLayer) '_' num2str(featureSize) '_r' num2str(subRate) ...
    '_blk' num2str(blkSize) '_mBat' num2str(batSize) ...
    '_' num2str(isLearnMtx(1)) '_' num2str(isLearnMtx(2)) ]; %%% model name
opts.learningRate     = [logspace(-3, -3, 50) logspace(-3.5, -3.5, 50) logspace(-4, -4, 50) logspace(-4.5, -4.5, 30)];   %_ Rate 3%%% you can change the learning rate
% opts.learningRate     = [logspace(-3,-3,100) logspace(-3.5,-3.5,50) logspace(-4, -4, 50) logspace(-4.5, -4.5, 50)];   %_ Rate 2%%% you can change the learning rate

% opts.learningRate     = [logspace(-3,-3,40) logspace(-3.5,-3.5,40) logspace(-4, -4, 40) logspace(-4.5, -4.5, 30)];   _ Rate 1%%% you can change the learning rate
opts.batchSize        = batSize;
opts.gpus             = [1]; %%% this code can only support one GPU!

opts.numSubBatches    = 2;
opts.bnormLearningRate= 0;

%%% solver
opts.solver           = 'Adam';
opts.numberImdb       = 1;
%%
%
% $$ttttttttttttttttttttttttttttttttng g $$
%

opts.imdbDir          = ['../../../TrainingPatches/imdb_512_' num2str(64) '_stride256.mat'];

opts.gradientClipping = false; %%% set 'true' to prevent exploding gradients in the beginning.
opts.backPropDepth    = Inf;
%%%------------;-------------------------------------------------------------
%%%   Initialize model and load data
%%%-------------------------------------------------------------------------
%%%  model
refModelName        = ['KCSNet_Phase1_' num2str(noLayer) '_' num2str(featureSize) '_r' num2str(subRate) ...
    '_blk' num2str(blkSize) '_mBat' num2str(batSize) ...
    '_' num2str(isLearnMtx(1)) '_' num2str(isLearnMtx(2)) ]; %%% model name

iter = 123; 
load(['data\' refModelName '\' refModelName '-epoch-' num2str(iter) '.mat']); 
net  = KCSNet_Init_Phase2(net);

%%%  load data
opts.expDir  = fullfile('data', opts.modelName);

%%%-------------------------------------------------------------------------
%%%   Train
%%%-------------------------------------------------------------------------

[net, info] = KCSNet_train(net,  ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'bnormLearningRate',opts.bnormLearningRate, ...
    'numSubBatches',opts.numSubBatches, ...
    'numberImdb',opts.numberImdb, ...
    'backPropDepth',opts.backPropDepth, ...
    'imdbDir',opts.imdbDir, ...
    'solver',opts.solver, ...
    'gradientClipping',opts.gradientClipping, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;