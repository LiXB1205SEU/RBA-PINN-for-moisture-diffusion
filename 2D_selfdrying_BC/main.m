%
% PINN for solving the PDE(mass diffusion): dh/dt  - C*dh2/d2x = 0,   C = 1 
% IC:  h(x,0) = 0.6+0.4*sin(pi()*x/(2*L))    L=1
% BC:  dh/dx (1,t) = 0     t>0
%      h(0,t) = 0.6        t>0
%
%
if exist('paperRunConfig', 'var')
    clearvars -except paperRunConfig
else
    clear
end
clc
close all
%% collocation points
%
numInternalCollocationPoints  =  10000;

Tlimit  = 0.1;
length_x = 2.0;
length_y = 2.0;
collocationSeed = 20260710;
networkSeed = 20260711;
icbcSeed = 20260712;
if exist('paperRunConfig', 'var')
    if isfield(paperRunConfig, 'collocationSeed'); collocationSeed = paperRunConfig.collocationSeed; end
    if isfield(paperRunConfig, 'networkSeed'); networkSeed = paperRunConfig.networkSeed; end
    if isfield(paperRunConfig, 'icbcSeed'); icbcSeed = paperRunConfig.icbcSeed; end
end
%
rng(collocationSeed, 'twister');
points =  lhsdesign (numInternalCollocationPoints, 3); 
%
dataT  = Tlimit * points (:,3);
dataX  = length_x * points (:,1); 
dataY  = length_y * points (:,2);


ds     = arrayDatastore ([dataX dataY dataT]);

%%
numLayers              =   8;
numNeurons             =   20; 
rng(networkSeed, 'twister');
parameters             =   struct; 
%
sz                     =   [numNeurons 3];
%
parameters.fc1.Weights = initializeHe (sz, 3);                       % 3 inputs
parameters.fc1.Bias    = initializeZeros ([numNeurons 1]);
%
for layerNumber = 2: numLayers -1 
    name  = "fc" + layerNumber; 
    %
    sz    = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe (sz, numIn); 
    parameters.(name).Bias    = initializeZeros ([numNeurons 1]);
end
%

sz      =  [1 numNeurons ];
numIn   =  numNeurons ;
parameters.("fc" + numLayers).Weights = initializeHe (sz , numIn);
parameters.("fc" + numLayers).Bias    = initializeZeros ([1 1]) ;    % 1 outputs      
initialParameters = parameters;
%
%%
% Specify Training Options
% Train the model for 1000 epochs with a mini - batch size of 1000.
%
numEpochs            =  500;
miniBatchSize        = 1000;
initialLearnRate     = 0.01;
decayRate            = 0.005;
executionEnvironment = "gpu";
if exist('paperRunConfig', 'var')
    if isfield(paperRunConfig, 'numEpochs') && ~isempty(paperRunConfig.numEpochs)
        numEpochs = paperRunConfig.numEpochs;
    end
    if isfield(paperRunConfig, 'executionEnvironment') && ~isempty(paperRunConfig.executionEnvironment)
        executionEnvironment = string(paperRunConfig.executionEnvironment);
    end
end

useGPU = executionEnvironment == "gpu";
toDevice = @(x) x;
if useGPU
    gpuDevice(1);
    parameters = dlupdate(@gpuArray, parameters);
    toDevice = @(x) gpuArray(x);
else
    error('Paper main-case training requires executionEnvironment = "gpu".');
end

% Train network
mbq = minibatchqueue (ds , ...
             'MiniBatchSize', miniBatchSize , ...
             'MiniBatchFormat', 'BC', ...
             'OutputEnvironment',  executionEnvironment);



averageGrad   = [];
averageSqGrad = [];


figure (1)
C        = colororder ;
lineLoss = animatedline ('Color',C(2 ,:));
ylim ([0 inf ])
xlabel ("Iteration")
ylabel ("Loss")
grid on


%%
c = 1;

start     = tic;

iteration = 0;
loss_hist = zeros(numEpochs,1);


for epoch = 1: numEpochs

    reset (mbq);
    epochLoss = 0;
    epochSampleCount = 0;

    while hasdata (mbq)
        %
        iteration = iteration + 1;

        dlXYT = next  (mbq);
    	dlX   = dlXYT (1, :);
	    dlY   = dlXYT (2, :);
	    dlT   = dlXYT (3, :);


        % Evaluate the model gradients and loss using dlfeval and the 
        % modelGradients function .


        [gradients, loss]  = dlfeval(@modelGradients, parameters, dlX, dlY, dlT, c);

        % Update learning rate .

        learningRate = initialLearnRate / (1+ decayRate * iteration );

        % Update the network parameters using the adamupdate function .

        [parameters, averageGrad, averageSqGrad] = adamupdate(parameters, gradients, averageGrad, ...
            averageSqGrad, iteration, learningRate);
        batchSize = numel(dlX);
        epochLoss = epochLoss + double(gather(extractdata(loss))) * batchSize;
        epochSampleCount = epochSampleCount + batchSize;
    end
    %                
    % Plot training progress .
    %
    loss = epochLoss / epochSampleCount;

    addpoints ( lineLoss , iteration , loss );

    D = duration (0,0, toc( start ), 'Format', 'hh:mm:ss');
    title (" Epoch : " + epoch + ", Elapsed : " + string (D) + ", Loss : " + loss )
    drawnow
    
    loss_hist  ( epoch, 1 )  = loss ;
    if exist('paperRunConfig', 'var') && isfield(paperRunConfig, 'progressRoot') && ...
            isfield(paperRunConfig, 'caseName') && isfield(paperRunConfig, 'modelName') && ...
            ~isempty(paperRunConfig.progressRoot) && exist('paperProgressUpdate', 'file') == 2
        if ~isfield(paperRunConfig, 'progressEvery') || isempty(paperRunConfig.progressEvery)
            paperRunConfig.progressEvery = max(1, floor(numEpochs / 20));
        end
        if epoch == 1 || epoch == numEpochs || mod(epoch, paperRunConfig.progressEvery) == 0
            paperProgressUpdate(paperRunConfig.progressRoot, paperRunConfig.caseName, ...
                paperRunConfig.modelName, "RUNNING", epoch, numEpochs, loss, toc(start), "");
        end
    end
   
end


%% Evaluate model accuracy
%
% 

tTest                     = [0.0 0.025 0.075 0.1];
Legacy_L2_all = zeros(length(tTest),1);

numPredictions            = 26;
XTest_points              = length_x * linspace (0,1, numPredictions );
YTest_points              = length_y * linspace (0,1, numPredictions );
[Xmesh, Ymesh]            = meshgrid (XTest_points, YTest_points);


%  离散有限元的节点坐标。

scriptDir = fileparts(mfilename('fullpath'));
if exist('paperRunConfig', 'var') && isfield(paperRunConfig, 'femDataFile') && ~isempty(paperRunConfig.femDataFile)
    femDataFile = paperRunConfig.femDataFile;
elseif exist(fullfile(scriptDir, 'FEM_data.mat'), 'file') == 2
    femDataFile = fullfile(scriptDir, 'FEM_data.mat');
else
    femDataFile = fullfile(fileparts(scriptDir), 'FEM_data.mat');
end
load(femDataFile, 'FEM_data');
utilsDir = fullfile(fileparts(fileparts(scriptDir)), 'utils');
if exist('computePaperMetrics', 'file') ~= 2
    addpath(utilsDir);
end
assert(exist('computePaperMetrics', 'file') == 2, ...
    'computePaperMetrics.m not found. Expected utils directory: %s', utilsDir);
%
FEM_interval              = 26;
X_FEM_points              = length_x * linspace (0,1, FEM_interval );
Y_FEM_points              = length_y * linspace (0,1, FEM_interval);
[X_FEM_mesh, Y_FEM_mesh]  = meshgrid (X_FEM_points, Y_FEM_points);

predictions = cell(length(tTest), 1);
FEM_all = cell(length(tTest), 1);
errors = cell(length(tTest), 1);

%%
for aa = 1:length(tTest)

    t     = tTest(aa);
    TTest = t * ones(1, numPredictions);

    dlUPred = zeros(size(Xmesh));
    FEM_Results = NaN(FEM_interval, FEM_interval);

    % PINN prediction at current time
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

    % FEM result at current time
    for i = 1:size(X_FEM_mesh,1)
        for j = 1:size(Y_FEM_mesh,1)
            X_Temp = X_FEM_mesh(1,i);
            Y_Temp = Y_FEM_mesh(j,1);

            idx = abs(FEM_data(:,1) - X_Temp) < 1e-12 & abs(FEM_data(:,2) - Y_Temp) < 1e-12;
            assert(any(idx), 'FEM point not found: x=%g, y=%g', X_Temp, Y_Temp);
            FEM_Results(j,i) = FEM_data(find(idx, 1, 'first'), 2 + aa);
        end
    end
    assert(all(isfinite(FEM_Results(:))), 'FEM reference contains nonfinite values.');

    % L2 error
    UPred = dlUPred;
    errU  = UPred - FEM_Results;
    predictions{aa} = UPred;
    FEM_all{aa} = FEM_Results;
    errors{aa} = errU;

    L2_error_term1 = 0.0;
    L2_error_term2 = 0.0;

    for i = 1:size(X_FEM_mesh,1)
        for j = 1:size(Y_FEM_mesh,1)
            L2_error_term1 = L2_error_term1 + (FEM_Results(i,j) - UPred(i,j))^2;
            L2_error_term2 = L2_error_term2 + UPred(i,j)^2;
        end
    end

    Legacy_L2_all(aa) = L2_error_term1 / L2_error_term2;
    fprintf('t = %.2f, legacy L2 = %.8e\n', t, Legacy_L2_all(aa));
    
    % % Plot predictions .

    figure(aa)
    contourf(Xmesh, Ymesh, dlUPred, 10, 'LineColor', 'none');
    clim([0.6,1])
    colormap jet
    title(sprintf('预测点云图 t = %.3f', t))
    colorbar
    
    % % Plot FEM .
    figure (10+aa)
    contourf(Xmesh ,Ymesh , FEM_Results, 10, 'LineColor', 'none'); % 20表示等值线份数
    clim([0.6,1])
    colormap jet; % 设置颜色主题
    title(sprintf('有限元点云图 t = %.3f', t))
    colorbar

    % Plot error values .
    figure (20+aa)
    contourf(Xmesh ,Ymesh , errU, 10, 'LineColor', 'none'); % 20表示等值线份数
    colorbar; % 显示颜色条
    colormap jet; % 设置颜色主题
    title(sprintf('误差点云图 t = %.3f', t))
    colorbar

end

[MetricsByTime, MetricsTotal] = computePaperMetrics(predictions, FEM_all, tTest);
Result_Table = MetricsByTime;
assert(all(isfinite(MetricsByTime.Relative_L2)) && all(isfinite(MetricsByTime.RMSE)) && ...
       all(isfinite(MetricsByTime.MAE)) && all(isfinite(MetricsByTime.MaxAE)), ...
       'Paper metrics contain nonfinite values.');
disp(Result_Table);

parameters_to_save = dlupdate(@gather, parameters);
averageGrad_to_save = dlupdate(@gather, averageGrad);
averageSqGrad_to_save = dlupdate(@gather, averageSqGrad);

result = struct();
if exist('scriptDir', 'var')
    [~, resultName] = fileparts(scriptDir);
else
    [~, resultName] = fileparts(pwd);
end
result.name = string(resultName);
result.parameters = parameters_to_save;
result.initialParameters = initialParameters;
result.averageGrad = averageGrad_to_save;
result.averageSqGrad = averageSqGrad_to_save;
result.predictions = predictions;
result.FEM = FEM_all;
result.errors = errors;
result.tTest = tTest;
result.loss_hist = loss_hist;
if exist('lossF_hist', 'var'); result.lossF_hist = lossF_hist; end
if exist('lossU_hist', 'var'); result.lossU_hist = lossU_hist; end
if exist('MetricsByTime', 'var'); result.MetricsByTime = MetricsByTime; end
if exist('MetricsTotal', 'var'); result.MetricsTotal = MetricsTotal; end
result.Result_Table = Result_Table;
result.Legacy_L2_all = Legacy_L2_all;
result.numEpochs = numEpochs;
result.numLayers = numLayers;
result.numNeurons = numNeurons;
result.miniBatchSize = miniBatchSize;
result.initialLearnRate = initialLearnRate;
result.decayRate = decayRate;
result.executionEnvironment = executionEnvironment;
result.numInternalCollocationPoints = numInternalCollocationPoints;
result.numPredictions = numPredictions;
result.length_x = length_x;
result.length_y = length_y;
result.Tlimit = Tlimit;
if exist('c', 'var'); result.c = c; end
if exist('femDataFile', 'var'); result.femDataFile = femDataFile; end
result.collocationSeed = collocationSeed;
result.networkSeed = networkSeed;
result.icbcSeed = icbcSeed;
result.dataX = dataX;
result.dataY = dataY;
result.dataT = dataT;

if exist('paperRunConfig', 'var') && isfield(paperRunConfig, 'resultFile') && ~isempty(paperRunConfig.resultFile)
    resultDir = fileparts(paperRunConfig.resultFile);
    if ~exist(resultDir, 'dir')
        mkdir(resultDir);
    end
    save(paperRunConfig.resultFile, 'result', '-v7.3');
end


% % Plot predictions .
%     figure (1)
%     contourf(Xmesh ,Ymesh , dlUPred , 10, 'LineColor', 'none'); % 20表示等值线份数
%     caxis([0.6,1])
%     colormap jet; % 设置颜色主题
%     title("预测点云图 t = " + t);
%     colorbar
% 
% 
% 
% % Plot true values .
%     figure (2)
%     contourf(Xmesh ,Ymesh , FEM_Results, 10, 'LineColor', 'none'); % 20表示等值线份数
%     caxis([0.6,1])
%     colormap jet; % 设置颜色主题
%     title("有限元点云图 t = " + t);
%     colorbar
% 
% 
% % Plot error values .
%     figure (3)
%     contourf(Xmesh ,Ymesh , errU, 10, 'LineColor', 'none'); % 20表示等值线份数
%     colorbar; % 显示颜色条
%     colormap jet; % 设置颜色主题
%     title("误差点云图 t = " + t);
%     colorbar








     
