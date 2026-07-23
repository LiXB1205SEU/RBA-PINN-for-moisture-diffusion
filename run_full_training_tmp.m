rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);

opts = struct();
opts.outputRoot = fullfile(rootDir, 'results');
opts.executionEnvironment = "gpu";
opts.numEpochs = 500;
opts.numFolds = 5;

run_all_network_layer_sweeps(opts);
disp('FULL_NETWORK_LAYER_TRAINING_DONE');
