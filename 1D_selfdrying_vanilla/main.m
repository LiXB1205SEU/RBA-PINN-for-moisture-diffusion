%
% PINN for solving the PDE(mass diffusion): dh/dt  - C*dh2/d2x = 0,   C = 1 
% IC:  h(x,0) = 0.6+0.4*sin(pi()*x/(2*L))    L=1
% BC:  dh/dx (1,t) = 0     t>0
%      h(0,t) = 0.6        t>0
%
if exist('paperRunConfig', 'var')
    clearvars -except paperRunConfig
else
    clear
end
%%
% 
% 选择 25 个等间距的时间点来强制执行每个边界条件 
numBoundaryConditionPoints = [25 25];

x0BC1 = -0 * ones(1,numBoundaryConditionPoints(1));                   % -0, -0, -0, ..., -0     25 points
x0BC2 =  2 * ones(1,numBoundaryConditionPoints(2));                   %  1,  1,  1, ...,  1     25 points

t0BC1 =  2 * linspace(0,1,numBoundaryConditionPoints(1));             %  0,  0.0417*2, ...,  2    25 points
t0BC2 =  2 * linspace(0,1,numBoundaryConditionPoints(2));             %  0,  0.0417*2, ...,  2    25 points

u0BC1 =  0.6 * ones(1,numBoundaryConditionPoints(1));                 %  0,  0,  0, ...,  0     25 points
u0BC2 =  0.6 * ones(1,numBoundaryConditionPoints(1));                 %  0,  0,  0, ...,  0     25 points

%% Select 50 equally spaced spatial points to enforce the initial condition .
%
numInitialConditionPoints  = 50;

x0IC = linspace(0,2,numInitialConditionPoints);                       %  -1,  -0.959, ..., 0.959, 1    50 points
t0IC = zeros(1,numInitialConditionPoints);                            %   0,  0,  0, ...,  0           50 points
%
u0IC = 0.6 + 0.4*sin(pi()*x0IC/2);                                    %   IC: u(x,0) = -sin(pi*x)
%u0IC  = ones(1,numInitialConditionPoints);
plot(x0IC, u0IC)

%% Uniformly sample 10,000 points  to enforce the output of the network to fulfill the Burger's equation.

numInternalCollocationPoints = 100;

points = rand(numInternalCollocationPoints,2);

dataX = 2*points(:,1);                                       % (0,2) spatial coordinates
dataT = points(:,2);                                         % (0,1)  time range


ds     = arrayDatastore ([dataX dataT]);

%%
numLayers    =   8;
numNeurons   =   20; 
parameters   =   struct; 
%
sz           =   [numNeurons 2];
%
parameters.fc1.Weights = initializeHe (sz, 2);                       % 2 inputs, dlx, dlt
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
%
%%
% Specify Training Options
% Train the model for 1000 epochs with a mini - batch size of 1000.
%
numEpochs            = 500;
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

dlX0IC   = dlarray (toDevice(x0IC),  'CB');                 % spatial coordinates (0,2)|50  
dlX0BC1  = dlarray (toDevice(x0BC1), 'CB');                 % spatial coordinates for BC1 (0)|25      
dlX0BC2  = dlarray (toDevice(x0BC2), 'CB');                 % spatial coordinates for BC2 (2)|25      


dlT0IC   = dlarray (toDevice(t0IC),  'CB');                 % spatial coordinates 0|50  
dlT0BC1	 = dlarray (toDevice(t0BC1), 'CB');                 % time series for BC1 (0,2)|25 
dlT0BC2	 = dlarray (toDevice(t0BC2), 'CB');                 % time series for BC2 (0,2)|25  

dlU0IC   = dlarray (toDevice(u0IC),  'CB');                 % Intial condition IC 0|50  
dlU0BC1	 = dlarray (toDevice(u0BC1), 'CB');                 % Boundary condtion BC1 (0,2)|25 
dlU0BC2	 = dlarray (toDevice(u0BC2), 'CB');                 % Boundary condtion BC2 (0,2)|25 


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


for epoch = 1: numEpochs

    reset (mbq);

    while hasdata (mbq)
        %
        iteration = iteration + 1;

        dlXT = next  (mbq);
    	dlX   = dlXT (1, :);
	    dlT   = dlXT (2, :);


        % Evaluate the model gradients and loss using dlfeval and the 
        % modelGradients function .


        [gradients, loss, lossF, lossU]  = dlfeval(@modelGradients, parameters, dlX, dlT,...
	                                        dlX0IC, dlX0BC1,  dlX0BC2,...
                                            dlT0IC, dlT0BC1,  dlT0BC2,...
                                            dlU0IC, dlU0BC1,  dlU0BC2,...
                                            c);

        % Update learning rate .

        learningRate = initialLearnRate / (1+ decayRate * iteration );

        % Update the network parameters using the adamupdate function .

        [parameters, averageGrad, averageSqGrad] = adamupdate(parameters, gradients, averageGrad, ...
            averageSqGrad, iteration, learningRate);
    end
    %                
    % Plot training progress .
    %
    loss  = double (gather ( extractdata ( loss )));
    lossF = double (gather ( extractdata ( lossF )));
    lossU = double (gather ( extractdata ( lossU )));

    addpoints ( lineLoss , iteration , loss );

    D = duration (0,0, toc( start ), 'Format', 'hh:mm:ss');
    title (" Epoch : " + epoch + ", Elapsed : " + string (D) + ", Loss : " + loss )
    drawnow
    
    loss_hist  ( epoch, 1 )  = loss ;
    if exist('paperRunConfig', 'var') && ...
            isfield(paperRunConfig, 'progressRoot') && ~isempty(paperRunConfig.progressRoot) && ...
            isfield(paperRunConfig, 'caseName') && ~isempty(paperRunConfig.caseName) && ...
            isfield(paperRunConfig, 'modelName') && ~isempty(paperRunConfig.modelName) && ...
            exist('paperProgressUpdate', 'file') == 2
        if ~isfield(paperRunConfig, 'progressEvery') || isempty(paperRunConfig.progressEvery)
            paperRunConfig.progressEvery = max(1, floor(numEpochs / 20));
        end
        if epoch == 1 || epoch == numEpochs || mod(epoch, paperRunConfig.progressEvery) == 0
            paperProgressUpdate(paperRunConfig.progressRoot, paperRunConfig.caseName, ...
                paperRunConfig.modelName, "RUNNING", epoch, numEpochs, loss, toc(start), "");
        end
    end
    lossF_hist ( epoch, 1 ) = lossF ;
    lossU_hist ( epoch, 1 ) = lossU ;
   
end




%% Evaluate model accuracy
%
% Evaluate model accuracy

tTest               = [0.0 0.25 0.75 1];
numObservationsTest = numel(tTest);

referenceFile = fullfile(fileparts(pwd), 'Vary_c_self_drying.xlsx');
referenceSheet = 'Vanilla';
referenceData = readmatrix(referenceFile, 'Sheet', referenceSheet);
referenceRows = 1:26;
referenceValueColumns = [24 26 28 30];
XTest = referenceData(referenceRows, 22)';
UTest = referenceData(referenceRows, referenceValueColumns)';
assert(~any(isnan(XTest(:))) && ~any(isnan(UTest(:))), ...
    'Reference data contains NaNs.');
szXTest = numel(XTest);

%Test the model. For each of the test inputs, predict the PDE solutions using the PINN and compare them to the solutions given by the solveBurgers function, listed in the Solve Burger's Equation Function section of the example. To access this function, open the example as a live script. Evaluate the accuracy by computing the relative error between the predictions and targets.
UPred = zeros(numObservationsTest, szXTest);

for i = 1:numObservationsTest
    t = tTest(i);

    dlXTest = dlarray(toDevice(XTest), 'CB');
    dlTTest = dlarray(toDevice(t * ones(1, szXTest)), 'CB');

    UPred(i,:) = double(gather(extractdata(modelU(parameters, dlXTest, dlTTest))));
end

err = norm(UPred - UTest) / norm(UTest)

if exist('computePaperMetrics', 'file') == 2
    [MetricsByTime, MetricsTotal] = computePaperMetrics(UPred, UTest, tTest);
    disp(MetricsByTime);
end

result = struct();
[~, resultName] = fileparts(pwd);
result.name = string(resultName);
result.UPred = UPred;
result.UTest = UTest;
result.predictions = UPred;
result.FEM = UTest;
result.tTest = tTest;
result.loss_hist = loss_hist;
if exist('lossF_hist', 'var'); result.lossF_hist = lossF_hist; end
if exist('lossU_hist', 'var'); result.lossU_hist = lossU_hist; end
if exist('MetricsByTime', 'var'); result.MetricsByTime = MetricsByTime; end
if exist('MetricsTotal', 'var'); result.MetricsTotal = MetricsTotal; end
result.numEpochs = numEpochs;
result.numLayers = numLayers;
result.numNeurons = numNeurons;
result.miniBatchSize = miniBatchSize;
result.initialLearnRate = initialLearnRate;
result.decayRate = decayRate;
result.executionEnvironment = executionEnvironment;
result.numInternalCollocationPoints = numInternalCollocationPoints;
result.numPredictions = szXTest;
if exist('c', 'var'); result.c = c; end

if exist('paperRunConfig', 'var') && isfield(paperRunConfig, 'resultFile') && ~isempty(paperRunConfig.resultFile)
    resultFolder = fileparts(paperRunConfig.resultFile);
    if ~isempty(resultFolder) && ~exist(resultFolder, 'dir')
        mkdir(resultFolder);
    end
    save(paperRunConfig.resultFile, 'result', '-v7.3');
end

figure
plot(XTest, (UPred(1,:) - UTest(1,:) ))
figure
plot(XTest, (UPred(2,:) - UTest(2,:) ))
figure
plot(XTest, (UPred(3,:) - UTest(3,:) ))
figure
plot(XTest, (UPred(4,:) - UTest(4,:) ))

%Visualize the test predictions in a plot.

figure
tiledlayout("flow")

for i = 1:numel(tTest)
    nexttile
    
    plot(XTest,UPred(i,:),"-",LineWidth=2);

    hold on
    plot(XTest, UTest(i,:),"--",LineWidth=2)
    hold off

    ylim([-1.1, 1.1])
    xlabel("x")
    ylabel("u(x," + t + ")")
end

legend(["Prediction" "Target"])






