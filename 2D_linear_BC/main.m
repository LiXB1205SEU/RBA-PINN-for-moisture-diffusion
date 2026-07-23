%
% PINN for solving the PDE(mass diffusion): dh/dt  - C*dh2/d2x = 0,   C = 1 
% IC:  h(x,0) = 0.6+0.4*sin(pi()*x/(2*L))    L=1
% BC:  dh/dx (1,t) = 0     t>0
%      h(0,t) = 0.6        t>0
%
%
clear
clc
close all
%% collocation points
%
numInternalCollocationPoints  =  10000;

Tlimit  = 0.1;
length_x = 2.0;
length_y = 2.0;
%
points =  lhsdesign (numInternalCollocationPoints, 3); 
%
dataT  = Tlimit * points (:,3);
dataX  = length_x * points (:,1); 
dataY  = length_y * points (:,2);

ds     = arrayDatastore ([dataX dataY dataT]);

%%
numLayers              =   8;
numNeurons             =   20; 
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
%
%%
% Specify Training Options
% Train the model for 1000 epochs with a mini - batch size of 1000.
%
numEpochs            =  500;
miniBatchSize        = 1000;
initialLearnRate     = 0.01;
decayRate            = 0.005;
executionEnvironment = "auto";

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
    end
    %                
    % Plot training progress .
    %
    loss  = double (gather ( extractdata ( loss )));

    addpoints ( lineLoss , iteration , loss );

    D = duration (0,0, toc( start ), 'Format', 'hh:mm:ss');
    title (" Epoch : " + epoch + ", Elapsed : " + string (D) + ", Loss : " + loss )
    drawnow
    
    loss_hist  ( epoch, 1 )  = loss ;
   
end


%% Evaluate model accuracy
%
% 

tTest                     = [0.0 0.025 0.075 0.1];
L2_error_all = zeros(length(tTest),1);

numPredictions            = 26;
XTest_points              = length_x * linspace (0,1, numPredictions );
YTest_points              = length_y * linspace (0,1, numPredictions );
[Xmesh, Ymesh]            = meshgrid (XTest_points, YTest_points);


%  离散有限元的节点坐标。

load('FEM_data.mat')
%
FEM_interval              = 26;
X_FEM_points              = length_x * linspace (0,1, FEM_interval );
Y_FEM_points              = length_y * linspace (0,1, FEM_interval);
[X_FEM_mesh, Y_FEM_mesh]  = meshgrid (X_FEM_points, Y_FEM_points);


%%
for aa = 1:length(tTest)

    t     = tTest(aa);
    TTest = t * ones(1, numPredictions);

    dlUPred = zeros(size(Xmesh));
    FEM_Results = zeros(FEM_interval, FEM_interval);

    % PINN prediction at current time
    for j = 1:size(Xmesh,1)
        XTest = Xmesh(j,:);
        YTest = Ymesh(j,:);

        dlXTest = dlarray(XTest, 'CB');
        dlYTest = dlarray(YTest, 'CB');
        dlTTest = dlarray(TTest, 'CB');

        UPred_temp1 = modelU(parameters, dlXTest, dlYTest, dlTTest);

        UPred_temp2 = dlTTest .* UPred_temp1 .* dlXTest .* (dlXTest-2) .* dlYTest .* (dlYTest-2) ...
                    + 0.6 + 0.4 * sin(pi()*dlXTest/2) .* sin(pi()*dlYTest/2);

        dlUPred(j,:) = double(extractdata(UPred_temp2));
    end

    % FEM result at current time
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

    % L2 error
    UPred = dlUPred;
    errU  = UPred - FEM_Results;

    L2_error_term1 = 0.0;
    L2_error_term2 = 0.0;

    for i = 1:size(X_FEM_mesh,1)
        for j = 1:size(Y_FEM_mesh,1)
            L2_error_term1 = L2_error_term1 + (FEM_Results(i,j) - UPred(i,j))^2;
            L2_error_term2 = L2_error_term2 + UPred(i,j)^2;
        end
    end

    L2_error_all(aa) = L2_error_term1 / L2_error_term2;
    fprintf('t = %.2f, L2_error = %.8e\n', t, L2_error_all(aa));
    
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

Result_Table = table(tTest(:), L2_error_all, ...
    'VariableNames', {'Time', 'L2_error'});
disp(Result_Table)


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








     