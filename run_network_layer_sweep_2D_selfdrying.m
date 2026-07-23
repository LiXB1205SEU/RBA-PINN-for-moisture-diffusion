function results = run_network_layer_sweep_2D_selfdrying(opts)
%RUN_NETWORK_LAYER_SWEEP_2D_SELDRYING Network-size sweep for 2D self-drying BC PINN.
%
% Corrected scope:
%   PDE:        nonlinear diffusion with self-drying
%   IC/BC:      hard-constrained by the trial solution
%   Reference:  matching 2D self-drying FEM data
%   Statistics: 5 folds by default, reported as mean +/- variance

if nargin < 1 || isempty(opts)
    opts = struct();
end

rootDir = fileparts(mfilename('fullpath'));
finalCodeRoot = fileparts(rootDir);

D.caseName = "2D_selfdrying_BC";
D.hiddenLayersList = [2 4 8];
D.neuronsList = [20 40 80];
D.numFolds = 5;
D.baseSeed = 42000;

D.numInternalCollocationPoints = 10000;
D.numEpochs = 500;
D.miniBatchSize = 1000;
D.initialLearnRate = 0.01;
D.decayRate = 0.005;
D.executionEnvironment = "gpu";
D.progressEvery = [];

D.lengthX = 2.0;
D.lengthY = 2.0;
D.Tlimit = 0.1;
D.diffusionC = 1.0;
D.Uc = 0.8;
D.tTest = [0.0 0.025 0.075 0.1];
D.numPredictions = 26;
D.FEMInterval = 26;
D.activation = "sin";
D.femDataFile = fullfile(finalCodeRoot, '2D_selfdrying_FEM_data.mat');

D.outputRoot = fullfile(rootDir, 'results', '2D_selfdrying_BC');
D.saveTrainedModels = true;

opts = applyDefaults(opts, D);
ensureDir(opts.outputRoot);

trainedRoot = fullfile(opts.outputRoot, 'trained_models');
if opts.saveTrainedModels
    ensureDir(trainedRoot);
end

FEM_data = loadFEMData(opts.femDataFile);
[useGPU, mbqEnvironment] = resolveExecutionEnvironment(opts.executionEnvironment);

allRunsTotal = table();
allRunsByTime = table();
allRunResults = {};

totalJobs = numel(opts.hiddenLayersList) * numel(opts.neuronsList) * opts.numFolds;
job = 0;

for iL = 1:numel(opts.hiddenLayersList)
    hiddenLayers = opts.hiddenLayersList(iL);
    totalLayers = hiddenLayers + 1;

    for iN = 1:numel(opts.neuronsList)
        neurons = opts.neuronsList(iN);

        for fold = 1:opts.numFolds
            job = job + 1;
            seed = opts.baseSeed + 10000 * hiddenLayers + 100 * neurons + fold;

            fprintf('\n[2D] job %d/%d | hiddenLayers=%d | neurons=%d | fold=%d/%d | seed=%d\n', ...
                job, totalJobs, hiddenLayers, neurons, fold, opts.numFolds, seed);

            runResult = trainOneFold2D(hiddenLayers, totalLayers, neurons, fold, seed, FEM_data, opts, useGPU, mbqEnvironment);
            allRunResults{end+1,1} = runResult; %#ok<AGROW>

            if opts.saveTrainedModels
                saveFile = fullfile(trainedRoot, ...
                    sprintf('2D_selfdrying_BC_H%d_N%d_fold%02d_seed%d.mat', hiddenLayers, neurons, fold, seed));
                runResult.saveFile = string(saveFile);
                save(saveFile, 'runResult', '-v7.3');
            else
                runResult.saveFile = "";
            end

            totalRow = makeTotalRow(opts.caseName, hiddenLayers, totalLayers, neurons, fold, seed, runResult);
            byTimeRows = makeByTimeRows(opts.caseName, hiddenLayers, totalLayers, neurons, fold, seed, runResult);

            allRunsTotal = appendTable(allRunsTotal, totalRow);
            allRunsByTime = appendTable(allRunsByTime, byTimeRows);
        end
    end
end

summaryTotal = makeSummaryTotal(allRunsTotal, opts.hiddenLayersList, opts.neuronsList);
summaryByTime = makeSummaryByTime(allRunsByTime, opts.hiddenLayersList, opts.neuronsList, opts.tTest);

writetable(allRunsTotal, fullfile(opts.outputRoot, 'all_folds_total.xlsx'));
writetable(allRunsByTime, fullfile(opts.outputRoot, 'all_folds_by_time.xlsx'));
writetable(summaryTotal, fullfile(opts.outputRoot, 'summary_total_mean_variance.xlsx'));
writetable(summaryByTime, fullfile(opts.outputRoot, 'summary_by_time_mean_variance.xlsx'));

writetable(allRunsTotal, fullfile(opts.outputRoot, 'all_folds_total.csv'));
writetable(allRunsByTime, fullfile(opts.outputRoot, 'all_folds_by_time.csv'));
writetable(summaryTotal, fullfile(opts.outputRoot, 'summary_total_mean_variance.csv'));
writetable(summaryByTime, fullfile(opts.outputRoot, 'summary_by_time_mean_variance.csv'));

results = struct();
results.opts = opts;
results.allRunsTotal = allRunsTotal;
results.allRunsByTime = allRunsByTime;
results.summaryTotal = summaryTotal;
results.summaryByTime = summaryByTime;
results.allRunResults = allRunResults;

save(fullfile(opts.outputRoot, 'network_layer_sweep_2D_selfdrying_BC_results.mat'), 'results', '-v7.3');
end

function runResult = trainOneFold2D(hiddenLayers, totalLayers, neurons, fold, seed, FEM_data, opts, useGPU, mbqEnvironment)
rng(seed);

points = lhsdesign(opts.numInternalCollocationPoints, 3);
dataX = opts.lengthX * points(:,1);
dataY = opts.lengthY * points(:,2);
dataT = opts.Tlimit * points(:,3);
ds = arrayDatastore([dataX dataY dataT]);

parameters = initializeNetwork(hiddenLayers, neurons, 3);
if useGPU
    parameters = moveLearnablesToGPU(parameters);
end

mbq = minibatchqueue(ds, ...
    'MiniBatchSize', opts.miniBatchSize, ...
    'MiniBatchFormat', 'BC', ...
    'OutputEnvironment', mbqEnvironment);

averageGrad = [];
averageSqGrad = [];
iteration = 0;
lossHist = zeros(opts.numEpochs, 1);
lossFHist = zeros(opts.numEpochs, 1);
lossUHist = zeros(opts.numEpochs, 1);
startTime = tic;

progressEvery = opts.progressEvery;
if isempty(progressEvery)
    progressEvery = max(1, floor(opts.numEpochs / 10));
end

for epoch = 1:opts.numEpochs
    reset(mbq);

    while hasdata(mbq)
        iteration = iteration + 1;
        dlXYT = next(mbq);
        dlX = dlXYT(1,:);
        dlY = dlXYT(2,:);
        dlT = dlXYT(3,:);

        [gradients, loss, lossF, lossU] = dlfeval(@modelGradients2D, parameters, dlX, dlY, dlT, opts);

        learningRate = opts.initialLearnRate / (1 + opts.decayRate * iteration);
        [parameters, averageGrad, averageSqGrad] = adamupdate(parameters, gradients, ...
            averageGrad, averageSqGrad, iteration, learningRate);
    end

    lossHist(epoch) = scalarData(loss);
    lossFHist(epoch) = scalarData(lossF);
    lossUHist(epoch) = scalarData(lossU);

    if epoch == 1 || epoch == opts.numEpochs || mod(epoch, progressEvery) == 0
        fprintf('[2D] H=%d N=%d fold=%d epoch=%d/%d loss=%.6e elapsed=%.1fs\n', ...
            hiddenLayers, neurons, fold, epoch, opts.numEpochs, lossHist(epoch), toc(startTime));
    end
end

[predictions, references, errors, metricsByTime, metricsTotal] = evaluateModel2D(parameters, FEM_data, opts, useGPU);

runResult = struct();
runResult.caseName = opts.caseName;
runResult.hiddenLayers = hiddenLayers;
runResult.totalLayers = totalLayers;
runResult.neurons = neurons;
runResult.fold = fold;
runResult.seed = seed;
runResult.parameters = gatherLearnables(parameters);
runResult.averageGrad = gatherLearnables(averageGrad);
runResult.averageSqGrad = gatherLearnables(averageSqGrad);
runResult.iteration = iteration;
runResult.loss_hist = lossHist;
runResult.lossF_hist = lossFHist;
runResult.lossU_hist = lossUHist;
runResult.finalLoss = lossHist(end);
runResult.finalLossF = lossFHist(end);
runResult.finalLossU = lossUHist(end);
runResult.tTest = opts.tTest;
runResult.predictions = predictions;
runResult.references = references;
runResult.errors = errors;
runResult.metricsByTime = metricsByTime;
runResult.metricsTotal = metricsTotal;
runResult.opts = opts;
end

function [gradients, loss, lossF, lossU] = modelGradients2D(parameters, dlX, dlY, dlT, opts)
U = trialSolution2D(parameters, dlX, dlY, dlT, opts);

gradientsU = dlgradient(sum(U, 'all'), {dlX, dlY, dlT}, 'EnableHigherDerivatives', true);
Ux = gradientsU{1};
Uy = gradientsU{2};
Ut = gradientsU{3};

Uxx = dlgradient(sum(Ux, 'all'), dlX, 'EnableHigherDerivatives', true);
Uyy = dlgradient(sum(Uy, 'all'), dlY, 'EnableHigherDerivatives', true);

U_clip = min(max(U, 0.0), 1.0);
s = (1 - U_clip) ./ (1 - opts.Uc);
c_local = 1 ./ (1 + s.^2);
c_local_dh = 2 .* (1 - U_clip) ./ ((1 - opts.Uc)^2 .* (1 + s.^2).^2);

dhs_dt = -0.2 .* dlT;

f1 = Ut - opts.diffusionC .* c_local .* (Uxx + Uyy) ...
    - opts.diffusionC .* c_local_dh .* (Ux.^2 + Uy.^2) ...
    - dhs_dt;

lossF = mean(f1.^2);
lossU = lossF .* 0;
loss = lossF + lossU;

gradients = dlgradient(loss, parameters);
end

function U = trialSolution2D(parameters, dlX, dlY, dlT, opts)
U_nn = modelU(parameters, [dlX; dlY; dlT], opts.activation);
U0 = 0.6 + 0.4 .* sin(pi() .* dlX ./ 2.0) .* sin(pi() .* dlY ./ 2.0);
U = dlT .* U_nn .* dlX .* (dlX - opts.lengthX) .* dlY .* (dlY - opts.lengthY) + U0;
end

function [predictions, references, errors, metricsByTime, metricsTotal] = evaluateModel2D(parameters, FEM_data, opts, useGPU)
xPoints = opts.lengthX * linspace(0, 1, opts.numPredictions);
yPoints = opts.lengthY * linspace(0, 1, opts.numPredictions);
[Xmesh, Ymesh] = meshgrid(xPoints, yPoints);

xFEM = opts.lengthX * linspace(0, 1, opts.FEMInterval);
yFEM = opts.lengthY * linspace(0, 1, opts.FEMInterval);
[XFEMMesh, YFEMMesh] = meshgrid(xFEM, yFEM);

nT = numel(opts.tTest);
predictions = cell(nT, 1);
references = cell(nT, 1);
errors = cell(nT, 1);

for it = 1:nT
    t = opts.tTest(it);
    TRow = t * ones(1, opts.numPredictions);
    UPred = zeros(size(Xmesh));
    URef = zeros(opts.FEMInterval, opts.FEMInterval);

    for row = 1:size(Xmesh, 1)
        XTest = Xmesh(row,:);
        YTest = Ymesh(row,:);
        TTest = TRow;

        if useGPU
            XTest = gpuArray(XTest);
            YTest = gpuArray(YTest);
            TTest = gpuArray(TTest);
        end

        dlX = dlarray(XTest, 'CB');
        dlY = dlarray(YTest, 'CB');
        dlT = dlarray(TTest, 'CB');
        U = trialSolution2D(parameters, dlX, dlY, dlT, opts);

        UPred(row,:) = double(gather(extractdata(U)));
    end

    femColumn = 2 + it;
    for ii = 1:size(XFEMMesh, 1)
        for jj = 1:size(YFEMMesh, 2)
            xVal = XFEMMesh(ii,jj);
            yVal = YFEMMesh(ii,jj);
            idx = abs(FEM_data(:,1) - xVal) < 1e-12 & abs(FEM_data(:,2) - yVal) < 1e-12;

            if ~any(idx)
                error('FEM point not found: x=%g, y=%g', xVal, yVal);
            end

            URef(ii,jj) = FEM_data(find(idx, 1, 'first'), femColumn);
        end
    end

    predictions{it} = UPred;
    references{it} = URef;
    errors{it} = UPred - URef;
end

[metricsByTime, metricsTotal] = computeMetrics(predictions, references, opts.tTest);
end

function parameters = initializeNetwork(hiddenLayers, neurons, numInputs)
parameters = struct();

parameters.fc1.Weights = dlarray(randn([neurons numInputs]) .* sqrt(2 / numInputs));
parameters.fc1.Bias = dlarray(zeros([neurons 1]));

for layerNumber = 2:hiddenLayers
    name = "fc" + layerNumber;
    parameters.(name).Weights = dlarray(randn([neurons neurons]) .* sqrt(2 / neurons));
    parameters.(name).Bias = dlarray(zeros([neurons 1]));
end

parameters.output.Weights = dlarray(randn([1 neurons]) .* sqrt(2 / neurons));
parameters.output.Bias = dlarray(zeros([1 1]));
end

function U = modelU(parameters, A, activation)
hiddenLayers = numel(fieldnames(parameters)) - 1;

for layerNumber = 1:hiddenLayers
    name = "fc" + layerNumber;
    A = fullyconnect(A, parameters.(name).Weights, parameters.(name).Bias);
    A = applyActivation(A, activation);
end

U = fullyconnect(A, parameters.output.Weights, parameters.output.Bias);
end

function A = applyActivation(A, activation)
switch lower(string(activation))
    case "sin"
        A = sin(A);
    case "tanh"
        A = tanh(A);
    case "relu"
        A = max(A, 0);
    case "sigmoid"
        A = 1 ./ (1 + exp(-A));
    otherwise
        error('Unknown activation: %s', string(activation));
end
end

function totalRow = makeTotalRow(caseName, hiddenLayers, totalLayers, neurons, fold, seed, runResult)
M = runResult.metricsTotal;
totalRow = table(string(caseName), hiddenLayers, totalLayers, neurons, fold, seed, ...
    M.Relative_L2(1), M.RMSE(1), M.R2(1), M.MAE(1), M.MaxAE(1), ...
    runResult.finalLoss, runResult.finalLossF, runResult.finalLossU, string(runResult.saveFile), ...
    'VariableNames', {'Case','HiddenLayers','TotalFCLayers','Neurons','Fold','Seed', ...
    'Relative_L2','RMSE','R2','MAE','MaxAE','FinalLoss','FinalLossF','FinalLossU','SaveFile'});
end

function rows = makeByTimeRows(caseName, hiddenLayers, totalLayers, neurons, fold, seed, runResult)
M = runResult.metricsByTime;
nRows = height(M);
rows = table(repmat(string(caseName), nRows, 1), repmat(hiddenLayers, nRows, 1), ...
    repmat(totalLayers, nRows, 1), repmat(neurons, nRows, 1), ...
    repmat(fold, nRows, 1), repmat(seed, nRows, 1), ...
    M.Time, M.Relative_L2, M.RMSE, M.R2, M.MAE, M.MaxAE, ...
    'VariableNames', {'Case','HiddenLayers','TotalFCLayers','Neurons','Fold','Seed', ...
    'Time','Relative_L2','RMSE','R2','MAE','MaxAE'});
end

function summary = makeSummaryTotal(allRunsTotal, hiddenLayersList, neuronsList)
metricList = ["Relative_L2","RMSE","R2","MAE","MaxAE","FinalLoss","FinalLossF","FinalLossU"];
summary = table();

for iL = 1:numel(hiddenLayersList)
    hiddenLayers = hiddenLayersList(iL);
    totalLayers = hiddenLayers + 1;

    for iN = 1:numel(neuronsList)
        neurons = neuronsList(iN);
        idx = allRunsTotal.HiddenLayers == hiddenLayers & allRunsTotal.Neurons == neurons;

        row = table("2D_selfdrying_BC", hiddenLayers, totalLayers, neurons, sum(idx), ...
            'VariableNames', {'Case','HiddenLayers','TotalFCLayers','Neurons','NumFolds'});

        for im = 1:numel(metricList)
            metric = metricList(im);
            vals = allRunsTotal.(char(metric))(idx);
            meanVal = meanNoNan(vals);
            varVal = varNoNan(vals);
            row.(char("Mean_" + metric)) = meanVal;
            row.(char("Variance_" + metric)) = varVal;
            row.(char(metric + "_Mean_pm_Variance")) = string(formatMeanVariance(meanVal, varVal));
        end

        % Explicit fields kept for static checking and spreadsheet clarity:
        % Variance_R2 is the fold-to-fold variance of R2.
        summary = appendTable(summary, row);
    end
end
end

function summary = makeSummaryByTime(allRunsByTime, hiddenLayersList, neuronsList, tTest)
metricList = ["Relative_L2","RMSE","R2","MAE","MaxAE"];
summary = table();

for iL = 1:numel(hiddenLayersList)
    hiddenLayers = hiddenLayersList(iL);
    totalLayers = hiddenLayers + 1;

    for iN = 1:numel(neuronsList)
        neurons = neuronsList(iN);

        for it = 1:numel(tTest)
            t = tTest(it);
            idx = allRunsByTime.HiddenLayers == hiddenLayers & ...
                allRunsByTime.Neurons == neurons & abs(allRunsByTime.Time - t) < 1e-12;

            row = table("2D_selfdrying_BC", hiddenLayers, totalLayers, neurons, t, sum(idx), ...
                'VariableNames', {'Case','HiddenLayers','TotalFCLayers','Neurons','Time','NumFolds'});

            for im = 1:numel(metricList)
                metric = metricList(im);
                vals = allRunsByTime.(char(metric))(idx);
                meanVal = meanNoNan(vals);
                varVal = varNoNan(vals);
                row.(char("Mean_" + metric)) = meanVal;
                row.(char("Variance_" + metric)) = varVal;
                row.(char(metric + "_Mean_pm_Variance")) = string(formatMeanVariance(meanVal, varVal));
            end

            summary = appendTable(summary, row);
        end
    end
end
end

function [metricsByTime, metricsTotal] = computeMetrics(predictions, references, times)
times = double(times(:));
nT = numel(times);
relativeL2 = zeros(nT, 1);
rmse = zeros(nT, 1);
r2 = zeros(nT, 1);
mae = zeros(nT, 1);
maxAe = zeros(nT, 1);
allPredictions = cell(nT, 1);
allReferences = cell(nT, 1);

for it = 1:nT
    pred = double(predictions{it}(:));
    ref = double(references{it}(:));
    [relativeL2(it), rmse(it), r2(it), mae(it), maxAe(it)] = calculateMetrics(pred, ref);
    allPredictions{it} = pred;
    allReferences{it} = ref;
end

[totalL2, totalRMSE, totalR2, totalMAE, totalMaxAE] = ...
    calculateMetrics(vertcat(allPredictions{:}), vertcat(allReferences{:}));

metricsByTime = table(times, relativeL2, rmse, r2, mae, maxAe, ...
    'VariableNames', {'Time','Relative_L2','RMSE','R2','MAE','MaxAE'});
metricsTotal = table(totalL2, totalRMSE, totalR2, totalMAE, totalMaxAE, ...
    'VariableNames', {'Relative_L2','RMSE','R2','MAE','MaxAE'});
end

function [relativeL2, rmse, r2, mae, maxAe] = calculateMetrics(predictionValues, referenceValues)
predictionValues = double(predictionValues(:));
referenceValues = double(referenceValues(:));
errors = predictionValues - referenceValues;

den = norm(referenceValues, 2);
if den == 0
    relativeL2 = NaN;
else
    relativeL2 = norm(errors, 2) / den;
end

rmse = sqrt(mean(errors.^2));
mae = mean(abs(errors));
maxAe = max(abs(errors));

r2Den = sum((referenceValues - mean(referenceValues)).^2);
if r2Den == 0
    r2 = NaN;
else
    r2 = 1 - sum(errors.^2) / r2Den;
end
end

function FEM_data = loadFEMData(femDataFile)
if ~isfile(femDataFile)
    error('FEM data file not found: %s', femDataFile);
end

S = load(femDataFile);
if isfield(S, 'FEM_data')
    FEM_data = S.FEM_data;
else
    names = fieldnames(S);
    FEM_data = S.(names{1});
end

if size(FEM_data, 2) < 6
    error('FEM data must contain x, y, and four time-result columns.');
end
end

function [useGPU, mbqEnvironment] = resolveExecutionEnvironment(executionEnvironment)
executionEnvironment = string(executionEnvironment);

if executionEnvironment == "gpu"
    gpuDevice(1);
    useGPU = true;
    mbqEnvironment = "gpu";
elseif executionEnvironment == "cpu"
    useGPU = false;
    mbqEnvironment = "cpu";
elseif executionEnvironment == "auto"
    try
        gpuDevice(1);
        useGPU = true;
        mbqEnvironment = "gpu";
    catch
        useGPU = false;
        mbqEnvironment = "cpu";
    end
else
    error('Unknown executionEnvironment: %s', executionEnvironment);
end
end

function parameters = moveLearnablesToGPU(parameters)
parameters = dlupdate(@(p) dlarray(gpuArray(extractdata(p))), parameters);
end

function value = gatherLearnables(value)
if isempty(value)
    return;
end
value = dlupdate(@(p) dlarray(gather(extractdata(p))), value);
end

function x = scalarData(x)
x = double(gather(extractdata(x)));
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

function ensureDir(folderPath)
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end
end

function out = appendTable(out, rows)
if isempty(out)
    out = rows;
else
    out = [out; rows];
end
end

function value = meanNoNan(values)
values = values(~isnan(values));
if isempty(values)
    value = NaN;
else
    value = mean(values);
end
end

function value = varNoNan(values)
values = values(~isnan(values));
if numel(values) < 2
    value = 0;
else
    value = var(values, 0);
end
end

function text = formatMeanVariance(meanVal, varVal)
text = sprintf('%.6g +/- %.3g', meanVal, varVal);
end
