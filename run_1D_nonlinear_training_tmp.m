rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir, '-begin');

opts = struct();
opts.outputRoot = fullfile(rootDir, 'results', '1D_nonlinear_BC');
opts.executionEnvironment = "gpu";
opts.numEpochs = 500;
opts.numFolds = 5;

run_network_layer_sweep_1D_nonlinear(opts);
write_combined_network_layer_workbook();
disp('FULL_1D_NONLINEAR_TRAINING_DONE');
