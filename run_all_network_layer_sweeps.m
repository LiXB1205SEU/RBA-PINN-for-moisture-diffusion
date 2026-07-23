function results = run_all_network_layer_sweeps(opts)
%RUN_ALL_NETWORK_LAYER_SWEEPS Run the corrected network-size sweeps.
%
% This entry point is intentionally inert until called from MATLAB. It does
% not start training by being present in the folder.

if nargin < 1 || isempty(opts)
    opts = struct();
end

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir, '-begin');

oneDOpts = struct();
oneDNonlinearOpts = struct();
oneDSelfDryingOpts = struct();
twoDOpts = struct();
combinedWorkbookOpts = struct();

if isfield(opts, 'oneD') && ~isempty(opts.oneD)
    oneDOpts = opts.oneD;
end

if isfield(opts, 'oneDNonlinear') && ~isempty(opts.oneDNonlinear)
    oneDNonlinearOpts = opts.oneDNonlinear;
end

if isfield(opts, 'oneDSelfDrying') && ~isempty(opts.oneDSelfDrying)
    oneDSelfDryingOpts = opts.oneDSelfDrying;
end

if isfield(opts, 'twoD') && ~isempty(opts.twoD)
    twoDOpts = opts.twoD;
end

if isfield(opts, 'outputRoot') && ~isempty(opts.outputRoot)
    oneDOpts.outputRoot = fullfile(opts.outputRoot, '1D_linear_BC');
    oneDNonlinearOpts.outputRoot = fullfile(opts.outputRoot, '1D_nonlinear_BC');
    oneDSelfDryingOpts.outputRoot = fullfile(opts.outputRoot, '1D_selfdrying_BC');
    twoDOpts.outputRoot = fullfile(opts.outputRoot, '2D_selfdrying_BC');
    combinedWorkbookOpts.resultsRoot = opts.outputRoot;
end

if isfield(opts, 'executionEnvironment') && ~isempty(opts.executionEnvironment)
    oneDOpts.executionEnvironment = opts.executionEnvironment;
    oneDNonlinearOpts.executionEnvironment = opts.executionEnvironment;
    oneDSelfDryingOpts.executionEnvironment = opts.executionEnvironment;
    twoDOpts.executionEnvironment = opts.executionEnvironment;
end

if isfield(opts, 'numEpochs') && ~isempty(opts.numEpochs)
    oneDOpts.numEpochs = opts.numEpochs;
    oneDNonlinearOpts.numEpochs = opts.numEpochs;
    oneDSelfDryingOpts.numEpochs = opts.numEpochs;
    twoDOpts.numEpochs = opts.numEpochs;
end

if isfield(opts, 'numFolds') && ~isempty(opts.numFolds)
    oneDOpts.numFolds = opts.numFolds;
    oneDNonlinearOpts.numFolds = opts.numFolds;
    oneDSelfDryingOpts.numFolds = opts.numFolds;
    twoDOpts.numFolds = opts.numFolds;
end

results = struct();
results.oneD = run_network_layer_sweep_1D_linear(oneDOpts);
if isfield(opts, 'runOneDNonlinear') && opts.runOneDNonlinear
    results.oneDNonlinear = run_network_layer_sweep_1D_nonlinear(oneDNonlinearOpts);
end
if isfield(opts, 'runOneDSelfDrying') && opts.runOneDSelfDrying
    results.oneDSelfDrying = run_network_layer_sweep_1D_selfdrying(oneDSelfDryingOpts);
end
results.twoD = run_network_layer_sweep_2D_selfdrying(twoDOpts);
if isfield(opts, 'writeCombinedWorkbook') && opts.writeCombinedWorkbook
    results.combinedWorkbook = write_combined_network_layer_workbook(combinedWorkbookOpts);
end
end
