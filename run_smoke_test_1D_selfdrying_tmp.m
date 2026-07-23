rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);

opts = struct();
opts.outputRoot = fullfile(rootDir, 'smoke_results_1D_selfdrying');
opts.executionEnvironment = "gpu";
opts.numEpochs = 1;
opts.numFolds = 1;
opts.hiddenLayersList = 2;
opts.neuronsList = 20;
opts.numInternalCollocationPoints = 20;
opts.miniBatchSize = 20;
opts.saveTrainedModels = false;

run_network_layer_sweep_1D_selfdrying(opts);
disp('SMOKE_1D_SELFDrying_DONE');
