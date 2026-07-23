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
%%
% 
% 选择 25 个等间距的时间点来强制执行每个边界条件 

length_x = 2.0; 
length_y = 2.0; 
Tlimit   = 0.1;
collocationSeed = 20260710;
networkSeed = 20260711;
icbcSeed = 20260712;
if exist('paperRunConfig', 'var')
    if isfield(paperRunConfig, 'collocationSeed'); collocationSeed = paperRunConfig.collocationSeed; end
    if isfield(paperRunConfig, 'networkSeed'); networkSeed = paperRunConfig.networkSeed; end
    if isfield(paperRunConfig, 'icbcSeed'); icbcSeed = paperRunConfig.icbcSeed; end
end

% ------------------------------- %
rng(icbcSeed, 'twister');
numBCSpace = 26;
numBCTime = 26;
[S, TT] = meshgrid(linspace(0,1,numBCSpace), linspace(0,Tlimit,numBCTime));
x0BC1 = length_x * S(:)'; y0BC1 = zeros(1,numel(S));
x0BC2 = length_x * S(:)'; y0BC2 = length_y * ones(1,numel(S));
x0BC3 = length_x * ones(1,numel(S)); y0BC3 = length_y * S(:)';
x0BC4 = zeros(1,numel(S)); y0BC4 = length_y * S(:)';
t0BC1 = TT(:)'; t0BC2 = TT(:)'; t0BC3 = TT(:)'; t0BC4 = TT(:)';
u0BC1 = 0.6 * ones(size(t0BC1)); u0BC2 = 0.6 * ones(size(t0BC2));
u0BC3 = 0.6 * ones(size(t0BC3)); u0BC4 = 0.6 * ones(size(t0BC4));

%% IC points

numICPerDim = 26;
[XIC, YIC] = meshgrid(length_x * linspace(0,1,numICPerDim), ...
    length_y * linspace(0,1,numICPerDim));
x0IC = XIC(:)';
y0IC = YIC(:)';
t0IC = zeros(size(x0IC));
u0IC = 0.6 + 0.4*sin(pi()*x0IC/2).*sin(pi()*y0IC/2);    % Initial condition

%% collocation points
%
numInternalCollocationPoints  =  10000;
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

dlX0IC   = dlarray (toDevice(x0IC), 'CB');
dlY0IC   = dlarray (toDevice(y0IC), 'CB');
dlT0IC   = dlarray (toDevice(t0IC), 'CB');
dlU0IC   = dlarray (toDevice(u0IC), 'CB');

dlX0BC1  = dlarray (toDevice(x0BC1), 'CB');
dlX0BC2  = dlarray (toDevice(x0BC2), 'CB');
dlX0BC3  = dlarray (toDevice(x0BC3), 'CB');
dlX0BC4  = dlarray (toDevice(x0BC4), 'CB');

dlY0BC1  = dlarray (toDevice(y0BC1), 'CB');
dlY0BC2  = dlarray (toDevice(y0BC2), 'CB');
dlY0BC3  = dlarray (toDevice(y0BC3), 'CB');
dlY0BC4  = dlarray (toDevice(y0BC4), 'CB');

dlT0BC1	 = dlarray (toDevice(t0BC1), 'CB');
dlT0BC2	 = dlarray (toDevice(t0BC2), 'CB');
dlT0BC3	 = dlarray (toDevice(t0BC3), 'CB');
dlT0BC4	 = dlarray (toDevice(t0BC4), 'CB');
		
dlU0BC1	 = dlarray (toDevice(u0BC1), 'CB');
dlU0BC2	 = dlarray (toDevice(u0BC2), 'CB');
dlU0BC3	 = dlarray (toDevice(u0BC3), 'CB');
dlU0BC4	 = dlarray (toDevice(u0BC4), 'CB');



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
lossF_hist = zeros(numEpochs,1);
lossU_hist = zeros(numEpochs,1);


for epoch = 1: numEpochs

    reset (mbq);
    epochLoss = 0;
    epochLossF = 0;
    epochLossU = 0;
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


        [gradients, loss, lossF, lossU]  = dlfeval(@modelGradients, parameters, dlX, dlY, dlT,...
	                                        dlX0IC,  dlY0IC,  dlT0IC,  dlU0IC, ...
                                            dlX0BC1, dlY0BC1, dlT0BC1, dlU0BC1,...
                                            dlX0BC2, dlY0BC2, dlT0BC2, dlU0BC2,...
                                            dlX0BC3, dlY0BC3, dlT0BC3, dlU0BC3,... 
                                            dlX0BC4, dlY0BC4, dlT0BC4, dlU0BC4,...
                                            c);

        % Update learning rate .

        learningRate = initialLearnRate / (1+ decayRate * iteration );

        % Update the network parameters using the adamupdate function .

        [parameters, averageGrad, averageSqGrad] = adamupdate(parameters, gradients, averageGrad, ...
            averageSqGrad, iteration, learningRate);
        batchSize = numel(dlX);
        epochLoss = epochLoss + double(gather(extractdata(loss))) * batchSize;
        epochLossF = epochLossF + double(gather(extractdata(lossF))) * batchSize;
        epochLossU = epochLossU + double(gather(extractdata(lossU))) * batchSize;
        epochSampleCount = epochSampleCount + batchSize;
    end
    %                
    % Plot training progress .
    %
    loss = epochLoss / epochSampleCount;
    lossF = epochLossF / epochSampleCount;
    lossU = epochLossU / epochSampleCount;

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
    lossF_hist ( epoch, 1 )  = lossF ;
    lossU_hist ( epoch, 1 )  = lossU ;
   
end


%% Evaluate model accuracy
%
% 

tTest                     = [0.0 0.025 0.075 0.1];
Legacy_L2_all             = zeros(length(tTest),1);

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

        UPred_temp = modelU(parameters, dlXTest, dlYTest, dlTTest);
        dlUPred(j,:) = double(gather(extractdata(UPred_temp)));
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
    caxis([0.6,1])
    colormap jet
    title(sprintf('预测点云图 t = %.3f', t))
    colorbar
    
    % % Plot FEM .
    figure (10+aa)
    contourf(Xmesh ,Ymesh , FEM_Results, 10, 'LineColor', 'none'); % 20表示等值线份数
    caxis([0.6,1])
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
result.ICBC = struct('xIC', x0IC, 'yIC', y0IC, 'tIC', t0IC, 'uIC', u0IC, ...
    'xBC', {{x0BC1,x0BC2,x0BC3,x0BC4}}, ...
    'yBC', {{y0BC1,y0BC2,y0BC3,y0BC4}}, ...
    'tBC', {{t0BC1,t0BC2,t0BC3,t0BC4}}, ...
    'uBC', {{u0BC1,u0BC2,u0BC3,u0BC4}});

if exist('paperRunConfig', 'var') && isfield(paperRunConfig, 'resultFile') && ~isempty(paperRunConfig.resultFile)
    resultDir = fileparts(paperRunConfig.resultFile);
    if ~exist(resultDir, 'dir')
        mkdir(resultDir);
    end
    save(paperRunConfig.resultFile, 'result', '-v7.3');
end
