function results = run_vanilla_RBA_2D_GPU_integrated(opts)
%RUN_VANILLA_RBA_2D_GPU_INTEGRATED
%
% Vanilla-RBA PINN for the 2D diffusion problem.
%
% Based on the uploaded 2D code structure:
%   x in [0,2], y in [0,2], t in [0,0.1]
%   FEM_data.mat is used for validation
%   PDE residual follows the uploaded modelGradients.m:
%       f = Ut - c*(Uxx + Uyy)
%
% Difference from BC/RBA-BC:
%   U = N(x,y,t) directly; IC/BC are enforced by soft loss terms.
%   RBA is applied to the PDE residual loss.
%
% Usage:
%   results = run_vanilla_RBA_2D_GPU_integrated;
%
% Optional:
%   opts = struct();
%   opts.numEpochs = 500;
%   opts.executionEnvironment = "gpu";
%   results = run_vanilla_RBA_2D_GPU_integrated(opts);

if nargin < 1
    opts = struct();
end

runtimeCleanup = activateLocalRuntime(); %#ok<NASGU>

%% ===================== Defaults =====================
D.outputRoot = fullfile(pwd, 'vanilla_RBA_2D_results');
D.femDataFile = fullfile(pwd, 'FEM_data.mat');

D.numInternalCollocationPoints = 10000;
D.Tlimit = 1.0;
D.length_x = 2.0;
D.length_y = 2.0;
D.c = 1.0;

D.numLayers = 8;
D.numNeurons = 20;
D.activation = "sin";

D.numEpochs = 500;
D.miniBatchSize = 1000;
D.initialLearnRate = 0.01;
D.decayRate = 0.005;
D.executionEnvironment = "gpu";  % "gpu" or "cpu"

% IC/BC soft constraint points
D.numICPerDim = 26;
D.numBCSpace = 26;
D.numBCTime = 26;

% RBA parameters, consistent with the code logic you provided
D.rba_gamma = 0.999;
D.rba_eta_init = 1.0;
D.rba_eta = 0.01;
D.rba_add_eps = 1e-12;

% Evaluation
D.tTest = [0.0 0.25 0.75 1.0];
D.numPredictions = 26;
D.FEM_interval = 26;
D.predictionCLim = [0.6 1.0];

D.rngSeed = 202601;
D.collocationSeed = 20260710;
D.networkSeed = 20260711;
D.icbcSeed = 20260712;
D.saveFigures = true;
D.saveModel = true;
D.verbose = true;
D.progressRoot = "";
D.caseName = "";
D.modelName = "";
D.progressEvery = [];

opts = applyDefaultsLocal(opts, D);

if ~exist(opts.outputRoot, 'dir')
    mkdir(opts.outputRoot);
end

figDir = fullfile(opts.outputRoot, 'figures');
if opts.saveFigures && ~exist(figDir, 'dir')
    mkdir(figDir);
end

if ~isfile(opts.femDataFile)
    error('FEM data file not found: %s', opts.femDataFile);
end

%% ===================== Device setup =====================
executionEnvironment = string(opts.executionEnvironment);
useGPU = false;

if executionEnvironment == "gpu"
    gpuDevice(1);
    useGPU = true;
elseif executionEnvironment == "cpu"
    useGPU = false;
else
    error('executionEnvironment must be "gpu" or "cpu".');
end

if opts.verbose
    fprintf('\n========================================\n');
    fprintf('2D Vanilla-RBA PINN\n');
    fprintf('Execution environment: %s\n', executionEnvironment);
    if useGPU
        fprintf('GPU: %s\n', gpuDevice().Name);
    end
    fprintf('PDE: Ut - c*(Uxx+Uyy) = 0\n');
    fprintf('Layers = %d, Neurons = %d, Activation = %s\n', ...
        opts.numLayers, opts.numNeurons, string(opts.activation));
    fprintf('Epochs = %d, Collocation points = %d, Batch size = %d\n', ...
        opts.numEpochs, opts.numInternalCollocationPoints, opts.miniBatchSize);
    fprintf('RBA: gamma = %.6g, eta_init = %.6g, eta = %.6g\n', ...
        opts.rba_gamma, opts.rba_eta_init, opts.rba_eta);
    fprintf('========================================\n');
end

%% ===================== Collocation points =====================
rng(opts.collocationSeed, 'twister');

points = lhsdesign(opts.numInternalCollocationPoints, 3);

dataX = opts.length_x * points(:,1);
dataY = opts.length_y * points(:,2);
dataT = opts.Tlimit   * points(:,3);

ds = arrayDatastore([dataX dataY dataT]);

%% ===================== IC and BC points =====================
rng(opts.icbcSeed, 'twister');
[ICBC, ICBC_gpu] = createICBCPoints(opts, useGPU);

%% ===================== Network initialization =====================
rng(opts.networkSeed, 'twister');
parameters = initializeNetworkLocal(opts.numLayers, opts.numNeurons, 3);
initialParameters = parameters;

if useGPU
    parameters = dlupdate(@gpuArray, parameters);
end

%% ===================== Mini-batch queue =====================
mbq = minibatchqueue(ds, ...
    'MiniBatchSize', opts.miniBatchSize, ...
    'MiniBatchFormat', 'BC', ...
    'OutputEnvironment', executionEnvironment);

numBatches = ceil(opts.numInternalCollocationPoints / opts.miniBatchSize);
rbaWeights = cell(numBatches, 1);

averageGrad = [];
averageSqGrad = [];
iteration = 0;

loss_hist_total = zeros(opts.numEpochs, 1);
loss_hist_weightedF = zeros(opts.numEpochs, 1);
loss_hist_unweightedF = zeros(opts.numEpochs, 1);
loss_hist_U = zeros(opts.numEpochs, 1);
elapsed_hist = zeros(opts.numEpochs, 1);
rbaScale_hist = zeros(opts.numEpochs, 1);

%% ===================== Training =====================
if useGPU
    wait(gpuDevice);
end

startTime = tic;

for epoch = 1:opts.numEpochs

    reset(mbq);

    epochLossTotal = 0.0;
    epochLossWeightedF = 0.0;
    epochLossUnweightedF = 0.0;
    epochLossU = 0.0;
    epochSampleCount = 0;
    batchCount = 0;
    rbaScale = computeRbaMemoryRmsLocal(rbaWeights);
    rbaScale_hist(epoch) = rbaScale;

    while hasdata(mbq)
        iteration = iteration + 1;
        batchCount = batchCount + 1;

        dlXYT = next(mbq);
        dlX = dlXYT(1,:);
        dlY = dlXYT(2,:);
        dlT = dlXYT(3,:);

        if batchCount > numel(rbaWeights) || isempty(rbaWeights{batchCount}) || ...
                ~isequal(size(extractdata(rbaWeights{batchCount})), size(extractdata(dlX)))
            rsum_old = dlarray(zeros(size(extractdata(dlX)), 'like', extractdata(dlX)), 'CB');
        else
            rsum_old = rbaWeights{batchCount};
        end

        [gradients, lossTotal, lossWeightedF, lossUnweightedF, lossU, rsum_new] = dlfeval( ...
            @modelGradientsVanillaRBALocal, ...
            parameters, dlX, dlY, dlT, ...
            ICBC_gpu, opts.c, opts.numLayers, opts.activation, ...
            epoch, rsum_old, opts.rba_gamma, opts.rba_eta_init, opts.rba_eta, ...
            opts.rba_add_eps, rbaScale);

        rbaWeights{batchCount} = rsum_new;

        learningRate = opts.initialLearnRate / (1 + opts.decayRate * iteration);

        [parameters, averageGrad, averageSqGrad] = adamupdate( ...
            parameters, gradients, averageGrad, averageSqGrad, iteration, learningRate);

        batchSize = numel(dlX);
        epochLossTotal = epochLossTotal + double(gather(extractdata(lossTotal))) * batchSize;
        epochLossWeightedF = epochLossWeightedF + double(gather(extractdata(lossWeightedF))) * batchSize;
        epochLossUnweightedF = epochLossUnweightedF + double(gather(extractdata(lossUnweightedF))) * batchSize;
        epochLossU = epochLossU + double(gather(extractdata(lossU))) * batchSize;
        epochSampleCount = epochSampleCount + batchSize;
    end

    if useGPU
        wait(gpuDevice);
    end

    loss_hist_total(epoch) = epochLossTotal / epochSampleCount;
    loss_hist_weightedF(epoch) = epochLossWeightedF / epochSampleCount;
    loss_hist_unweightedF(epoch) = epochLossUnweightedF / epochSampleCount;
    loss_hist_U(epoch) = epochLossU / epochSampleCount;
    elapsed_hist(epoch) = toc(startTime);

    if strlength(string(opts.progressRoot)) > 0 && exist('paperProgressUpdate', 'file') == 2
        progressEvery = opts.progressEvery;
        if isempty(progressEvery)
            progressEvery = max(1, floor(opts.numEpochs / 20));
        end
        if epoch == 1 || epoch == opts.numEpochs || mod(epoch, progressEvery) == 0
            paperProgressUpdate(opts.progressRoot, opts.caseName, opts.modelName, ...
                "RUNNING", epoch, opts.numEpochs, loss_hist_total(epoch), toc(startTime), "");
        end
    end

    if opts.verbose && (epoch == 1 || epoch == opts.numEpochs || mod(epoch, max(1, floor(opts.numEpochs/10))) == 0)
        fprintf('Epoch %5d/%d, total = %.6e, weightedF = %.6e, unweightedF = %.6e, U = %.6e, elapsed = %.2f s\n', ...
            epoch, opts.numEpochs, loss_hist_total(epoch), loss_hist_weightedF(epoch), ...
            loss_hist_unweightedF(epoch), loss_hist_U(epoch), elapsed_hist(epoch));
    end
end

%% ===================== Evaluation =====================
FEM_data = loadFEMDataLocal(opts.femDataFile);
evalData = prepareEvaluationData(FEM_data, opts);

evalResult = evaluateModel(parameters, evalData, opts, useGPU);

%% ===================== Save results =====================
LossTable = table((1:opts.numEpochs)', loss_hist_total, loss_hist_weightedF, ...
    loss_hist_unweightedF, loss_hist_U, elapsed_hist, ...
    'VariableNames', {'Epoch','TotalLoss','WeightedResidualLoss','UnweightedResidualLoss','SoftICBCLoss','ElapsedTime_s'});

writetable(LossTable, fullfile(opts.outputRoot, 'loss_history.xlsx'));
writetable(evalResult.MetricsByTime, fullfile(opts.outputRoot, 'metrics_by_time.xlsx'));
writetable(evalResult.MetricsTotal, fullfile(opts.outputRoot, 'metrics_total.xlsx'));

if opts.saveFigures
    exportLossFigure(LossTable, fullfile(figDir, 'loss_curve.png'));
    exportPredictionAndErrorFigures(evalData, evalResult, opts, figDir);
end

if useGPU
    parameters_to_save = dlupdate(@gather, parameters);
    averageGrad_to_save = gatherDlStruct(averageGrad);
    averageSqGrad_to_save = gatherDlStruct(averageSqGrad);
    rbaWeights_to_save = gatherRBACell(rbaWeights);
else
    parameters_to_save = parameters;
    averageGrad_to_save = averageGrad;
    averageSqGrad_to_save = averageSqGrad;
    rbaWeights_to_save = rbaWeights;
end

results = struct();
results.opts = opts;
results.parameters = parameters_to_save;
results.initialParameters = initialParameters;
results.averageGrad = averageGrad_to_save;
results.averageSqGrad = averageSqGrad_to_save;
results.rbaWeights = rbaWeights_to_save;
results.rba_gamma = opts.rba_gamma;
results.rba_eta_init = opts.rba_eta_init;
results.rba_eta = opts.rba_eta;
results.rbaScale_hist = rbaScale_hist;
results.iteration = iteration;
results.LossTable = LossTable;
results.MetricsByTime = evalResult.MetricsByTime;
results.MetricsTotal = evalResult.MetricsTotal;
results.predictions = evalResult.predictions;
results.FEM = evalResult.FEM;
results.errors = evalResult.errors;
results.ICBC = ICBC;
results.Legacy_L2_all = evalResult.MetricsByTime.Legacy_L2;
results.collocationSeed = opts.collocationSeed;
results.networkSeed = opts.networkSeed;
results.icbcSeed = opts.icbcSeed;
results.dataX = dataX;
results.dataY = dataY;
results.dataT = dataT;
results.numInternalCollocationPoints = opts.numInternalCollocationPoints;
results.numEpochs = opts.numEpochs;
results.numLayers = opts.numLayers;
results.numNeurons = opts.numNeurons;
results.miniBatchSize = opts.miniBatchSize;
results.initialLearnRate = opts.initialLearnRate;
results.decayRate = opts.decayRate;
results.executionEnvironment = opts.executionEnvironment;
results.tTest = opts.tTest;

save(fullfile(opts.outputRoot, 'vanilla_RBA_2D_results.mat'), 'results', '-v7.3');

if opts.saveModel
    save(fullfile(opts.outputRoot, 'vanilla_RBA_2D_trained_model.mat'), ...
        'parameters_to_save', 'averageGrad_to_save', 'averageSqGrad_to_save', ...
        'rbaWeights_to_save', 'iteration', 'opts', '-v7.3');
end

if opts.verbose
    fprintf('\nDone. Files saved to:\n');
    fprintf('  %s\n', opts.outputRoot);
end

end

%% ========================================================================
function [ICBC, ICBC_gpu] = createICBCPoints(opts, useGPU)

% Initial condition over a 2D grid at t = 0.
xIC = opts.length_x * linspace(0,1,opts.numICPerDim);
yIC = opts.length_y * linspace(0,1,opts.numICPerDim);
[XIC, YIC] = meshgrid(xIC, yIC);

TIC = zeros(size(XIC));
UIC = 0.6 + 0.4 .* sin(pi() .* XIC ./ 2) .* sin(pi() .* YIC ./ 2);

% Four Dirichlet boundaries h = 0.6.
s = linspace(0,1,opts.numBCSpace);
tt = linspace(0,opts.Tlimit,opts.numBCTime);
[S, TT] = meshgrid(s, tt);

% x = 0
XBC1 = zeros(size(S));
YBC1 = opts.length_y .* S;
TBC1 = TT;
UBC1 = 0.6 .* ones(size(S));

% x = 2
XBC2 = opts.length_x .* ones(size(S));
YBC2 = opts.length_y .* S;
TBC2 = TT;
UBC2 = 0.6 .* ones(size(S));

% y = 0
XBC3 = opts.length_x .* S;
YBC3 = zeros(size(S));
TBC3 = TT;
UBC3 = 0.6 .* ones(size(S));

% y = 2
XBC4 = opts.length_x .* S;
YBC4 = opts.length_y .* ones(size(S));
TBC4 = TT;
UBC4 = 0.6 .* ones(size(S));

ICBC = struct();

ICBC.XIC = XIC(:)';
ICBC.YIC = YIC(:)';
ICBC.TIC = TIC(:)';
ICBC.UIC = UIC(:)';

ICBC.XBC = [XBC1(:); XBC2(:); XBC3(:); XBC4(:)]';
ICBC.YBC = [YBC1(:); YBC2(:); YBC3(:); YBC4(:)]';
ICBC.TBC = [TBC1(:); TBC2(:); TBC3(:); TBC4(:)]';
ICBC.UBC = [UBC1(:); UBC2(:); UBC3(:); UBC4(:)]';

if useGPU
    ICBC_gpu.XIC = dlarray(gpuArray(ICBC.XIC), 'CB');
    ICBC_gpu.YIC = dlarray(gpuArray(ICBC.YIC), 'CB');
    ICBC_gpu.TIC = dlarray(gpuArray(ICBC.TIC), 'CB');
    ICBC_gpu.UIC = dlarray(gpuArray(ICBC.UIC), 'CB');

    ICBC_gpu.XBC = dlarray(gpuArray(ICBC.XBC), 'CB');
    ICBC_gpu.YBC = dlarray(gpuArray(ICBC.YBC), 'CB');
    ICBC_gpu.TBC = dlarray(gpuArray(ICBC.TBC), 'CB');
    ICBC_gpu.UBC = dlarray(gpuArray(ICBC.UBC), 'CB');
else
    ICBC_gpu.XIC = dlarray(ICBC.XIC, 'CB');
    ICBC_gpu.YIC = dlarray(ICBC.YIC, 'CB');
    ICBC_gpu.TIC = dlarray(ICBC.TIC, 'CB');
    ICBC_gpu.UIC = dlarray(ICBC.UIC, 'CB');

    ICBC_gpu.XBC = dlarray(ICBC.XBC, 'CB');
    ICBC_gpu.YBC = dlarray(ICBC.YBC, 'CB');
    ICBC_gpu.TBC = dlarray(ICBC.TBC, 'CB');
    ICBC_gpu.UBC = dlarray(ICBC.UBC, 'CB');
end

end

%% ========================================================================
function [gradients, lossTotal, lossWeightedF, lossUnweightedF, lossU, rsum_new] = modelGradientsVanillaRBALocal( ...
    parameters, dlX, dlY, dlT, ICBC, c, numLayers, activation, epoch, rsum_old, gamma, eta_init, eta_main, epsDenom, rbaScale)

% Network output directly represents U.
U = modelULocal(parameters, dlX, dlY, dlT, activation, numLayers);

% PDE derivatives
gradientsU = dlgradient(sum(U, 'all'), {dlX, dlY, dlT}, ...
    'EnableHigherDerivatives', true);

Ux = gradientsU{1};
Uy = gradientsU{2};
Ut = gradientsU{3};

Uxx = dlgradient(sum(Ux, 'all'), dlX, 'EnableHigherDerivatives', true);
Uyy = dlgradient(sum(Uy, 'all'), dlY, 'EnableHigherDerivatives', true);

% Linear diffusion PDE residual, consistent with uploaded modelGradients.m
f1 = Ut - c .* (Uxx + Uyy);

lossUnweightedF = mean(f1.^2);

% RBA update, consistent with the provided logic.
if epoch == 1
    eta = eta_init;
else
    eta = eta_main;
end

r_norm = eta .* abs(f1) ./ (max(abs(f1), [], 'all') + epsDenom);
rsum_raw = gamma .* rsum_old + r_norm;
rsum_new = dlarray(extractdata(rsum_raw), 'CB');
rsum_normalized = rsum_new ./ max(rbaScale, epsDenom);

lossWeightedF = mean((rsum_normalized .* f1).^2);

% Soft IC loss
UICPred = modelULocal(parameters, ICBC.XIC, ICBC.YIC, ICBC.TIC, activation, numLayers);
lossIC = l2loss(UICPred, ICBC.UIC);

% Soft BC loss
UBCPred = modelULocal(parameters, ICBC.XBC, ICBC.YBC, ICBC.TBC, activation, numLayers);
lossBC = l2loss(UBCPred, ICBC.UBC);

lossU = lossIC + lossBC;

lossTotal = lossWeightedF + lossU;

gradients = dlgradient(lossTotal, parameters);

end

%% ========================================================================
function U = modelULocal(parameters, dlX, dlY, dlT, activation, numLayers)

A = [dlX; dlY; dlT];

for layerNumber = 1:numLayers-1
    name = "fc" + layerNumber;

    A = fullyconnect(A, parameters.(name).Weights, parameters.(name).Bias);
    A = applyActivationLocal(A, activation);
end

name = "fc" + numLayers;
U = fullyconnect(A, parameters.(name).Weights, parameters.(name).Bias);

end

%% ========================================================================
function A = applyActivationLocal(A, activation)

switch lower(string(activation))
    case "tanh"
        A = tanh(A);
    case "sigmoid"
        A = 1 ./ (1 + exp(-A));
    case "relu"
        A = max(A, 0);
    case "sin"
        A = sin(A);
    otherwise
        error('Unknown activation: %s', string(activation));
end

end

%% ========================================================================
function parameters = initializeNetworkLocal(numLayers, numNeurons, numInputs)

parameters = struct();

parameters.fc1.Weights = initializeHeLocal([numNeurons numInputs], numInputs);
parameters.fc1.Bias = initializeZerosLocal([numNeurons 1]);

for layerNumber = 2:numLayers-1
    name = "fc" + layerNumber;
    parameters.(name).Weights = initializeHeLocal([numNeurons numNeurons], numNeurons);
    parameters.(name).Bias = initializeZerosLocal([numNeurons 1]);
end

parameters.("fc" + numLayers).Weights = initializeHeLocal([1 numNeurons], numNeurons);
parameters.("fc" + numLayers).Bias = initializeZerosLocal([1 1]);

end

function W = initializeHeLocal(sz, numIn)
W = dlarray(sqrt(2/numIn) * randn(sz, 'single'));
end

function Z = initializeZerosLocal(sz)
Z = dlarray(zeros(sz, 'single'));
end

%% ========================================================================
function FEM_data = loadFEMDataLocal(femDataFile)

S = load(femDataFile);

if isfield(S, 'FEM_data')
    FEM_data = S.FEM_data;
elseif isfield(S, 'FEM_RBA')
    FEM_data = S.FEM_RBA;
else
    names = fieldnames(S);
    error('FEM file must contain variable FEM_data or FEM_RBA. Found: %s', strjoin(names, ', '));
end

end

%% ========================================================================
function evalData = prepareEvaluationData(FEM_data, opts)

XTest_points = opts.length_x * linspace(0,1,opts.numPredictions);
YTest_points = opts.length_y * linspace(0,1,opts.numPredictions);
[Xmesh, Ymesh] = meshgrid(XTest_points, YTest_points);

X_FEM_points = opts.length_x * linspace(0,1,opts.FEM_interval);
Y_FEM_points = opts.length_y * linspace(0,1,opts.FEM_interval);
[X_FEM_mesh, Y_FEM_mesh] = meshgrid(X_FEM_points, Y_FEM_points);

FEM_all = cell(numel(opts.tTest),1);

for aa = 1:numel(opts.tTest)
    FEM_Results = zeros(opts.FEM_interval, opts.FEM_interval);

    for i = 1:size(X_FEM_mesh,1)
        for j = 1:size(Y_FEM_mesh,1)
            X_Temp = X_FEM_mesh(1,i);
            Y_Temp = Y_FEM_mesh(j,1);

            idx = abs(FEM_data(:,1) - X_Temp) < 1e-12 & ...
                  abs(FEM_data(:,2) - Y_Temp) < 1e-12;

            if any(idx)
                FEM_Results(j,i) = FEM_data(find(idx,1,'first'), 2 + aa);
            else
                error('FEM point not found: x=%g, y=%g', X_Temp, Y_Temp);
            end
        end
    end

    FEM_all{aa} = FEM_Results;
end

evalData = struct();
evalData.Xmesh = Xmesh;
evalData.Ymesh = Ymesh;
evalData.FEM_all = FEM_all;

end

%% ========================================================================
function evalResult = evaluateModel(parameters, evalData, opts, useGPU)

nT = numel(opts.tTest);

predictions = cell(nT,1);
errors = cell(nT,1);
FEM = evalData.FEM_all;

Relative_L2 = zeros(nT,1);
Legacy_L2 = zeros(nT,1);
RMSE = zeros(nT,1);
R2 = zeros(nT,1);
MAE = zeros(nT,1);
MaxAE = zeros(nT,1);

allPred = [];
allRef = [];

for aa = 1:nT
    t = opts.tTest(aa);
    TTest = t * ones(1, opts.numPredictions);

    UPred = zeros(size(evalData.Xmesh));

    for j = 1:size(evalData.Xmesh,1)
        XTest = evalData.Xmesh(j,:);
        YTest = evalData.Ymesh(j,:);

        if useGPU
            dlXTest = dlarray(gpuArray(XTest), 'CB');
            dlYTest = dlarray(gpuArray(YTest), 'CB');
            dlTTest = dlarray(gpuArray(TTest), 'CB');
        else
            dlXTest = dlarray(XTest, 'CB');
            dlYTest = dlarray(YTest, 'CB');
            dlTTest = dlarray(TTest, 'CB');
        end

        UPredTemp = modelULocal(parameters, dlXTest, dlYTest, dlTTest, opts.activation, opts.numLayers);
        UPred(j,:) = double(gather(extractdata(UPredTemp)));
    end

    URef = FEM{aa};
    errU = UPred - URef;

    Relative_L2(aa) = norm(errU(:),2) / norm(URef(:),2);
    Legacy_L2(aa) = sum(errU(:).^2) / sum(UPred(:).^2);
    RMSE(aa) = sqrt(mean(errU(:).^2));

    SS_res = sum(errU(:).^2);
    SS_tot = sum((URef(:) - mean(URef(:))).^2);
    R2(aa) = 1 - SS_res / SS_tot;

    MAE(aa) = mean(abs(errU(:)));
    MaxAE(aa) = max(abs(errU(:)));

    predictions{aa} = UPred;
    errors{aa} = errU;

    allPred = [allPred; UPred(:)]; %#ok<AGROW>
    allRef = [allRef; URef(:)]; %#ok<AGROW>
end

allErr = allPred - allRef;

Relative_L2_total = norm(allErr(:),2) / norm(allRef(:),2);
Legacy_L2_total = sum(allErr(:).^2) / sum(allPred(:).^2);
RMSE_total = sqrt(mean(allErr(:).^2));
SS_res_total = sum(allErr(:).^2);
SS_tot_total = sum((allRef(:) - mean(allRef(:))).^2);
R2_total = 1 - SS_res_total / SS_tot_total;
MAE_total = mean(abs(allErr(:)));
MaxAE_total = max(abs(allErr(:)));

MetricsByTime = table(opts.tTest(:), Relative_L2, Legacy_L2, RMSE, R2, MAE, MaxAE, ...
    'VariableNames', {'Time','Relative_L2','Legacy_L2','RMSE','R2','MAE','MaxAE'});

MetricsTotal = table(Relative_L2_total, Legacy_L2_total, RMSE_total, R2_total, MAE_total, MaxAE_total, ...
    'VariableNames', {'Relative_L2','Legacy_L2','RMSE','R2','MAE','MaxAE'});

evalResult = struct();
evalResult.predictions = predictions;
evalResult.FEM = FEM;
evalResult.errors = errors;
evalResult.MetricsByTime = MetricsByTime;
evalResult.MetricsTotal = MetricsTotal;

end

%% ========================================================================
function exportLossFigure(LossTable, fileName)

fig = figure('Visible','off','Color','w','Position',[100 100 760 500]);
semilogy(LossTable.Epoch, LossTable.TotalLoss, 'LineWidth', 1.5);
hold on
semilogy(LossTable.Epoch, LossTable.WeightedResidualLoss, '--', 'LineWidth', 1.3);
semilogy(LossTable.Epoch, LossTable.UnweightedResidualLoss, ':', 'LineWidth', 1.3);
semilogy(LossTable.Epoch, LossTable.SoftICBCLoss, '-.', 'LineWidth', 1.3);
grid on
xlabel('Epoch');
ylabel('Loss');
legend('Total loss','RBA weighted residual','Unweighted residual','Soft IC/BC loss','Location','best');
title('Vanilla-RBA loss history');
exportgraphics(fig, fileName, 'Resolution', 300);
close(fig);

end

%% ========================================================================
function exportPredictionAndErrorFigures(evalData, evalResult, opts, figDir)

predDir = fullfile(figDir, 'prediction');
errDir = fullfile(figDir, 'error');

if ~exist(predDir, 'dir'); mkdir(predDir); end
if ~exist(errDir, 'dir'); mkdir(errDir); end

maxAbsError = 0;
for aa = 1:numel(opts.tTest)
    maxAbsError = max(maxAbsError, max(abs(evalResult.errors{aa}(:))));
end
errCLim = [-maxAbsError maxAbsError];

for aa = 1:numel(opts.tTest)
    t = opts.tTest(aa);

    fig = figure('Visible','off','Color','w','Position',[100 100 520 430]);
    contourf(evalData.Xmesh, evalData.Ymesh, evalResult.predictions{aa}, 10, 'LineColor', 'none');
    caxis(opts.predictionCLim);
    colormap jet;
    colorbar;
    xlabel('x');
    ylabel('y');
    title(sprintf('Vanilla-RBA prediction, t = %.3f', t));
    exportgraphics(fig, fullfile(predDir, sprintf('prediction_t_%s.png', timeTag(t))), 'Resolution', 300);
    close(fig);

    fig = figure('Visible','off','Color','w','Position',[100 100 520 430]);
    contourf(evalData.Xmesh, evalData.Ymesh, evalResult.errors{aa}, 10, 'LineColor', 'none');
    caxis(errCLim);
    colormap jet;
    colorbar;
    xlabel('x');
    ylabel('y');
    title(sprintf('Vanilla-RBA error, t = %.3f', t));
    exportgraphics(fig, fullfile(errDir, sprintf('error_t_%s.png', timeTag(t))), 'Resolution', 300);
    close(fig);
end

end

%% ========================================================================
function tag = timeTag(t)
tag = strrep(sprintf('%.3f', t), '.', 'p');
end

%% ========================================================================
function S = gatherDlStruct(S)

if isempty(S)
    return
end

try
    S = dlupdate(@gather, S);
catch
end

end

%% ========================================================================
function C = gatherRBACell(C)

for i = 1:numel(C)
    if ~isempty(C{i})
        try
            C{i} = dlarray(gather(extractdata(C{i})), 'CB');
        catch
            try
                C{i} = gather(C{i});
            catch
            end
        end
    end
end

end

%% ========================================================================
function scale = computeRbaMemoryRmsLocal(memory)

sumSquares = 0;
count = 0;
for i = 1:numel(memory)
    if ~isempty(memory{i})
        values = memory{i};
        if isa(values, 'dlarray')
            values = extractdata(values);
        end
        values = gather(values);
        sumSquares = sumSquares + sum(double(values(:)).^2);
        count = count + numel(values);
    end
end
if count == 0
    scale = 1;
else
    scale = sqrt(sumSquares / count + 1e-12);
end

end

%% ========================================================================
function opts = applyDefaultsLocal(opts, D)

fn = fieldnames(D);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}) || isempty(opts.(fn{i}))
        opts.(fn{i}) = D.(fn{i});
    end
end

end

%% ========================================================================
function cleanupObj = activateLocalRuntime()

pathBefore = path;
fileDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(fileDir);
utilsDir = fullfile(rootDir, 'utils');
if exist(utilsDir, 'dir')
    addpath(utilsDir, '-begin');
end
addpath(fileDir, '-begin');
rehash;
cleanupObj = onCleanup(@() restoreRuntime(pathBefore));

end

%% ========================================================================
function restoreRuntime(pathBefore)

path(pathBefore);
rehash;

end
