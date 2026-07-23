rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);

opts = struct();
opts.outputRoot = fullfile(rootDir, 'results', '1D_selfdrying_BC');
opts.executionEnvironment = "gpu";
opts.numEpochs = 500;
opts.numFolds = 5;

run_network_layer_sweep_1D_selfdrying(opts);
disp('FULL_1D_SELFDrying_TRAINING_DONE');
