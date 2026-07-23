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
numInternalCollocationPoints = 100;

points = rand(numInternalCollocationPoints,2);

dataX = 2*points(:,1);                                         % (0,2) spatial coordinates
dataT =   points(:,2);                                         % (0,1)  time range

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
        
        if (epoch ==1 )
            rsum_old = zeros(1,length(dlX));
        end

        [gradients, loss, lossF, lossF_unweighted, rsum_new]  = dlfeval(@modelGradients, parameters, dlX, dlT, c, epoch, rsum_old);
        


        % Update learning rate .

        learningRate = initialLearnRate / (1+ decayRate * iteration );

        % Update the network parameters using the adamupdate function .

        [parameters, averageGrad, averageSqGrad] = adamupdate(parameters, gradients, averageGrad, ...
            averageSqGrad, iteration, learningRate);
    end

    rsum_old = rsum_new;
    %                
    % Plot training progress .
    %
    loss  = double (gather ( extractdata ( loss )));
    lossF = double (gather ( extractdata ( lossF )));
    lossF_unweighted = double (gather ( extractdata ( lossF_unweighted )));
    lossU = 0;

    addpoints ( lineLoss , iteration , loss );

    D = duration (0,0, toc( start ), 'Format', 'hh:mm:ss');
    title (" Epoch : " + epoch + ", Elapsed : " + string (D) + ", Loss : " + loss )
    drawnow
    
    loss_hist  ( epoch, 1 )  = loss ;
    if exist('paperRunConfig', 'var') && isfield(paperRunConfig, 'progressRoot') && ...
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
    weightedLossF_hist ( epoch, 1 ) = lossF ;
    unweightedLossF_hist ( epoch, 1 ) = lossF_unweighted ;
    lossU_hist ( epoch, 1 ) = lossU ;
   
end




%% Evaluate model accuracy
%
% Evaluate model accuracy

tTest               = [0.0 0.25 0.75 1];
numObservationsTest = numel(tTest);

szXTest = 1001;
XTest = linspace(0, 2, szXTest);

%Test the model. For each of the test inputs, predict the PDE solutions using the PINN and compare them to the solutions given by the solveBurgers function, listed in the Solve Burger's Equation Function section of the example. To access this function, open the example as a live script. Evaluate the accuracy by computing the relative error between the predictions and targets.
UPred = zeros(numObservationsTest, szXTest);
UTest = zeros(numObservationsTest, szXTest);

for i = 1:numObservationsTest
    t = tTest(i);

    dlXTest = dlarray(toDevice(XTest), 'CB');
    dlTTest = dlarray(toDevice(t * ones(1, szXTest)), 'CB');

    %
    UPred(i,:) = double(gather(extractdata( ...
        dlTTest.*modelU(parameters, dlXTest, dlTTest).*dlXTest.*(dlXTest-2) ...
        + 0.6 + 0.4 * sin(pi() * dlXTest / 2.0))));
    %
    UTest(i,:) = 0.6 + 0.4*sin(pi()*XTest/2) * exp(-t/ (4/pi()/pi())  );   
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
if exist('weightedLossF_hist', 'var'); result.weightedLossF_hist = weightedLossF_hist; end
if exist('unweightedLossF_hist', 'var'); result.unweightedLossF_hist = unweightedLossF_hist; end
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

if exist('paperRunConfig', 'var') && isfield(paperRunConfig, 'resultFile')
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







