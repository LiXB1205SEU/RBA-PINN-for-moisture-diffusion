function results = run_network_layer_sweep_1D_nonlinear(opts)
%RUN_NETWORK_LAYER_SWEEP_1D_NONLINEAR Network-size sweep for 1D nonlinear BC PINN.
%
% Corrected scope:
%   PDE:        nonlinear diffusion without self-drying
%   IC/BC:      hard-constrained by the trial solution
%   Reference:  matching main-case reference data in Vary_dcdx.xlsx
%   Statistics: 5 folds by default, reported as mean +/- variance
%
% The delegated summary keeps Variance_R2 for fold-to-fold variability.

if nargin < 1 || isempty(opts)
    opts = struct();
end

rootDir = fileparts(mfilename('fullpath'));

D.caseName = "1D_nonlinear_BC";
D.hiddenLayersList = [2 4 8];
D.neuronsList = [20 40 80];
D.numFolds = 5;
D.baseSeed = 31000;

D.numInternalCollocationPoints = 1000;
D.numEpochs = 500;
D.miniBatchSize = 1000;
D.initialLearnRate = 0.01;
D.decayRate = 0.005;
D.executionEnvironment = "gpu";
D.progressEvery = [];

D.lengthX = 2.0;
D.Tlimit = 1.0;
D.diffusionC = 1.0;
D.Uc = 0.8;
D.includeSelfDrying = false;
D.tTest = [0.0 0.25 0.75 1.0];
D.referenceFile = fullfile(fileparts(rootDir), 'Vary_dcdx.xlsx');
D.referenceSheet = 'BC_dcdh';
D.referenceRows = 1:26;
D.referenceXColumn = 21;
D.referenceValueColumns = [23 25 27 29];
D.activation = "tanh";

D.outputRoot = fullfile(rootDir, 'results', '1D_nonlinear_BC');
D.saveTrainedModels = true;
D.resultFileName = "network_layer_sweep_1D_nonlinear_BC_results.mat";

opts = applyDefaults(opts, D);
results = run_network_layer_sweep_1D_selfdrying(opts);
end

function opts = applyDefaults(opts, defaults)
names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = defaults.(name);
    end
end
end
