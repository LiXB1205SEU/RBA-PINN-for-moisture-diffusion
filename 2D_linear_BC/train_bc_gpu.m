function result = train_bc_gpu(saveFile, femDataFile, doPlot, opts)
%TRAIN_BC Train 2D BC-PINN with hard IC/BC constraint for linear diffusion.

if nargin < 1 || isempty(saveFile)
    saveFile = fullfile(pwd, 'bc_trained.mat');
end
if nargin < 2 || isempty(femDataFile)
    femDataFile = fullfile(pwd, 'FEM_data.mat');
end
if nargin < 3 || isempty(doPlot)
    doPlot = true;
end
if nargin < 4
    opts = struct();
end

runtimeCleanup = activateLocalRuntime(); %#ok<NASGU>

D.length_x = 2.0;
D.length_y = 2.0;
D.Tlimit = 1.0;
D.numInternalCollocationPoints = 10000;
D.numLayers = 8;
D.numNeurons = 20;
D.numEpochs = 500;
D.miniBatchSize = 1000;
D.initialLearnRate = 0.01;
D.decayRate = 0.005;
D.executionEnvironment = "gpu";
D.tTest = [0.0 0.25 0.75 1.0];
D.numPredictions = 26;
D.c = 1;
D.rngSeed = [];
D.collocationSeed = 20260710;
D.networkSeed = 20260711;
D.icbcSeed = 20260712;
D.progressRoot = "";
D.caseName = "";
D.modelName = "";
D.progressEvery = [];
opts = applyDefaults(opts, D);

length_x = opts.length_x;
length_y = opts.length_y;
Tlimit = opts.Tlimit;
numInternalCollocationPoints = opts.numInternalCollocationPoints;
numLayers = opts.numLayers;
numNeurons = opts.numNeurons;
numEpochs = opts.numEpochs;
miniBatchSize = opts.miniBatchSize;
initialLearnRate = opts.initialLearnRate;
decayRate = opts.decayRate;
executionEnvironment = opts.executionEnvironment;
tTest = opts.tTest;
numPredictions = opts.numPredictions;
c = opts.c;
collocationSeed = opts.collocationSeed;
networkSeed = opts.networkSeed;
icbcSeed = opts.icbcSeed;

useGPU = executionEnvironment == "gpu";
toDevice = @(x) x;
if useGPU
    toDevice = @(x) gpuArray(x);
end

rng(collocationSeed, 'twister');
points = lhsdesign(numInternalCollocationPoints, 3);
dataX = length_x * points(:,1);
dataY = length_y * points(:,2);
dataT = Tlimit * points(:,3);
ds = arrayDatastore([dataX dataY dataT]);

rng(networkSeed, 'twister');
parameters = struct();
parameters.fc1.Weights = initializeHe([numNeurons 3], 3);
parameters.fc1.Bias = initializeZeros([numNeurons 1]);
for layerNumber = 2:numLayers-1
    name = "fc" + layerNumber;
    parameters.(name).Weights = initializeHe([numNeurons numNeurons], numNeurons);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end
parameters.("fc" + numLayers).Weights = initializeHe([1 numNeurons], numNeurons);
parameters.("fc" + numLayers).Bias = initializeZeros([1 1]);
initialParameters = parameters;

if useGPU
    parameters = dlupdate(@gpuArray, parameters);
end

mbq = minibatchqueue(ds, ...
    'MiniBatchSize', miniBatchSize, ...
    'MiniBatchFormat', 'BC', ...
    'OutputEnvironment', executionEnvironment);

averageGrad = [];
averageSqGrad = [];
iteration = 0;
loss_hist = zeros(numEpochs,1);

if doPlot
    figure('Name', 'BC Training Loss');
    C = colororder;
    lineLoss = animatedline('Color', C(2,:));
    ylim([0 inf])
    xlabel('Iteration')
    ylabel('Loss')
    grid on
end

start = tic;

for epoch = 1:numEpochs
    reset(mbq);
    epochLoss = 0;
    epochSampleCount = 0;
    while hasdata(mbq)
        iteration = iteration + 1;
        dlXYT = next(mbq);
        dlX = dlXYT(1,:);
        dlY = dlXYT(2,:);
        dlT = dlXYT(3,:);

        [gradients, loss] = dlfeval(@modelGradients, parameters, dlX, dlY, dlT, c);
        learningRate = initialLearnRate / (1 + decayRate * iteration);
        [parameters, averageGrad, averageSqGrad] = adamupdate( ...
            parameters, gradients, averageGrad, averageSqGrad, iteration, learningRate);
        batchSize = numel(dlX);
        epochLoss = epochLoss + double(gather(extractdata(loss))) * batchSize;
        epochSampleCount = epochSampleCount + batchSize;
    end

    loss = epochLoss / epochSampleCount;
    if doPlot
        addpoints(lineLoss, iteration, loss);
        Dtime = duration(0,0,toc(start), 'Format', 'hh:mm:ss');
        title("Epoch: " + epoch + ", Elapsed: " + string(Dtime) + ", Loss: " + loss)
        drawnow
    end
    loss_hist(epoch) = loss;

    if strlength(string(opts.progressRoot)) > 0 && exist('paperProgressUpdate', 'file') == 2
        progressEvery = opts.progressEvery;
        if isempty(progressEvery)
            progressEvery = max(1, floor(numEpochs / 20));
        end
        if epoch == 1 || epoch == numEpochs || mod(epoch, progressEvery) == 0
            paperProgressUpdate(opts.progressRoot, opts.caseName, opts.modelName, ...
                "RUNNING", epoch, numEpochs, loss_hist(epoch), toc(start), "");
        end
    end
end

%% Postprocess
load(femDataFile, 'FEM_data');
FEM_interval = 26;
XTest_points = length_x * linspace(0,1,numPredictions);
YTest_points = length_y * linspace(0,1,numPredictions);
[Xmesh, Ymesh] = meshgrid(XTest_points, YTest_points);
X_FEM_points = length_x * linspace(0,1,FEM_interval);
Y_FEM_points = length_y * linspace(0,1,FEM_interval);
[X_FEM_mesh, Y_FEM_mesh] = meshgrid(X_FEM_points, Y_FEM_points);

Legacy_L2_all = zeros(length(tTest),1);
predictions = cell(length(tTest),1);
FEM_all = cell(length(tTest),1);
errors = cell(length(tTest),1);

for aa = 1:length(tTest)
    t = tTest(aa);
    TTest = t * ones(1, numPredictions);
    dlUPred = zeros(size(Xmesh));
    FEM_Results = zeros(FEM_interval, FEM_interval);

    for j = 1:size(Xmesh,1)
        XTest = Xmesh(j,:);
        YTest = Ymesh(j,:);
        dlXTest = dlarray(toDevice(XTest), 'CB');
        dlYTest = dlarray(toDevice(YTest), 'CB');
        dlTTest = dlarray(toDevice(TTest), 'CB');
        UPred_temp1 = modelU(parameters, dlXTest, dlYTest, dlTTest);
        UPred_temp2 = dlTTest .* UPred_temp1 .* dlXTest .* (dlXTest-2) .* dlYTest .* (dlYTest-2) ...
                    + 0.6 + 0.4 * sin(pi()*dlXTest/2) .* sin(pi()*dlYTest/2);
        dlUPred(j,:) = double(gather(extractdata(UPred_temp2)));
    end

    for i = 1:size(X_FEM_mesh,1)
        for j = 1:size(Y_FEM_mesh,1)
            X_Temp = X_FEM_mesh(1,i);
            Y_Temp = Y_FEM_mesh(j,1);
            for k = 1:(FEM_interval*FEM_interval)
                if (X_Temp == FEM_data(k,1)) && (Y_Temp == FEM_data(k,2))
                    FEM_Results(j,i) = FEM_data(k, 2 + aa);
                end
            end
        end
    end

    UPred = dlUPred;
    errU = UPred - FEM_Results;
    L2_error_term1 = 0.0;
    L2_error_term2 = 0.0;
    for i = 1:size(X_FEM_mesh,1)
        for j = 1:size(X_FEM_mesh,2)
            L2_error_term1 = L2_error_term1 + (FEM_Results(i,j) - UPred(i,j))^2;
            L2_error_term2 = L2_error_term2 + UPred(i,j)^2;
        end
    end
    Legacy_L2_all(aa) = L2_error_term1 / L2_error_term2;
    predictions{aa} = UPred;
    FEM_all{aa} = FEM_Results;
    errors{aa} = errU;

    if doPlot
        figure(aa)
        contourf(Xmesh, Ymesh, dlUPred, 10, 'LineColor', 'none');
        caxis([0.6,1]); colormap jet; title(sprintf('Prediction t = %.3f', t)); colorbar
        figure(10+aa)
        contourf(Xmesh, Ymesh, FEM_Results, 10, 'LineColor', 'none');
        caxis([0.6,1]); colormap jet; title(sprintf('FEM t = %.3f', t)); colorbar
        figure(20+aa)
        contourf(Xmesh, Ymesh, errU, 10, 'LineColor', 'none');
        colormap jet; title(sprintf('Error t = %.3f', t)); colorbar
    end
end

if exist('computePaperMetrics', 'file') ~= 2
    rootUtils = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'utils');
    if exist(rootUtils, 'dir')
        addpath(rootUtils);
    end
end
assert(exist('computePaperMetrics', 'file') == 2, 'computePaperMetrics.m is required for paper metrics.');
[MetricsByTime, MetricsTotal] = computePaperMetrics(predictions, FEM_all, tTest);
Result_Table = MetricsByTime;
assert(all(isfinite(MetricsByTime.Relative_L2)) && all(isfinite(MetricsByTime.RMSE)) && ...
       all(isfinite(MetricsByTime.MAE)) && all(isfinite(MetricsByTime.MaxAE)), ...
       'Paper metrics contain nonfinite values.');

if useGPU
    parameters_to_save = dlupdate(@gather, parameters);
    if isempty(averageGrad)
        averageGrad_to_save = averageGrad;
    else
        averageGrad_to_save = dlupdate(@gather, averageGrad);
    end
    if isempty(averageSqGrad)
        averageSqGrad_to_save = averageSqGrad;
    else
        averageSqGrad_to_save = dlupdate(@gather, averageSqGrad);
    end
else
    parameters_to_save = parameters;
    averageGrad_to_save = averageGrad;
    averageSqGrad_to_save = averageSqGrad;
end

result = struct();
result.name = 'BC';
result.parameters = parameters_to_save;
result.initialParameters = initialParameters;
result.loss_hist = loss_hist;
result.Legacy_L2_all = Legacy_L2_all;
result.Result_Table = Result_Table;
result.MetricsByTime = MetricsByTime;
result.MetricsTotal = MetricsTotal;
result.tTest = tTest;
result.length_x = length_x;
result.length_y = length_y;
result.Tlimit = Tlimit;
result.c = c;
result.numLayers = numLayers;
result.numNeurons = numNeurons;
result.numEpochs = numEpochs;
result.miniBatchSize = miniBatchSize;
result.initialLearnRate = initialLearnRate;
result.decayRate = decayRate;
result.executionEnvironment = executionEnvironment;
result.numInternalCollocationPoints = numInternalCollocationPoints;
result.numPredictions = numPredictions;
result.averageGrad = averageGrad_to_save;
result.averageSqGrad = averageSqGrad_to_save;
result.iteration = iteration;
result.prediction_mode = 'hard_constraint';
result.predictions = predictions;
result.FEM = FEM_all;
result.errors = errors;
result.femDataFile = femDataFile;
result.saveFile = saveFile;
result.opts = opts;
result.collocationSeed = collocationSeed;
result.networkSeed = networkSeed;
result.icbcSeed = icbcSeed;
result.dataX = dataX;
result.dataY = dataY;
result.dataT = dataT;

save(saveFile, 'result', '-v7.3');

end

function opts = applyDefaults(opts, defaultOpts)
fn = fieldnames(defaultOpts);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}) || isempty(opts.(fn{i}))
        opts.(fn{i}) = defaultOpts.(fn{i});
    end
end
end

function cleanupObj = activateLocalRuntime()
pathBefore = path;
fileDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(fileDir);
utilsDir = fullfile(rootDir, 'utils');
if exist(utilsDir, 'dir')
    addpath(utilsDir, '-begin');
end
addpath(fileDir, '-begin');
clear modelGradients modelU initializeHe initializeZeros
rehash;
cleanupObj = onCleanup(@() restoreRuntime(pathBefore));
end

function restoreRuntime(pathBefore)
path(pathBefore);
rehash;
end
