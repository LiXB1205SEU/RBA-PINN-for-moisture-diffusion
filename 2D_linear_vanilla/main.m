%
% PINN for solving the PDE(mass diffusion): dh/dt  - C*dh2/d2x = 0,   C = 1 
% IC:  h(x,0) = 0.6+0.4*sin(pi()*x/(2*L))    L=1
% BC:  dh/dx (1,t) = 0     t>0
%      h(0,t) = 0.6        t>0
%
%
clear
clc
close all;
%%
% 
% 选择 25 个等间距的时间点来强制执行每个边界条件 

length_x = 2.0; 
length_y = 2.0; 
Tlimit   = 0.1;

% ------------------------------- %
numBoundaryConditionPoints  = [25 25 25 25];

x0BC1   =   length_x * linspace (0, 1, numBoundaryConditionPoints(1));                   % x coorinates for BC1 (x,0)
x0BC2   =   length_x * linspace (0, 1, numBoundaryConditionPoints(2));                   % x coorinates for BC2 (x,2)
x0BC3   =   length_x * ones     (1,    numBoundaryConditionPoints(3));                   % x coorinates for BC3 (2,y)
x0BC4   =   length_x * zeros    (1,    numBoundaryConditionPoints(4));                   % x coorinates for BC4 (0,y)

y0BC1   =   length_y * zeros    (1,    numBoundaryConditionPoints(1));                   % y coorinates for BC1 (x,0) 
y0BC2   =   length_y * ones     (1,    numBoundaryConditionPoints(2));                   % x coorinates for BC2 (x,2)
y0BC3   =   length_y * linspace (0, 1, numBoundaryConditionPoints(3));                   % x coorinates for BC3 (2,y) 
y0BC4   =   length_y * linspace (0, 1, numBoundaryConditionPoints(4));                   % x coorinates for BC4 (0,y)

% ------------------------------- %

t0BC1   =   linspace (0, Tlimit, numBoundaryConditionPoints(1));                  % BC1: time (0,2)  
t0BC2   =   linspace (0, Tlimit, numBoundaryConditionPoints(2));                  % BC2: time (0,2)   
t0BC3   =   linspace (0, Tlimit, numBoundaryConditionPoints(3));                  % BC3: time (0,2)   
t0BC4   =   linspace (0, Tlimit, numBoundaryConditionPoints(4));                  % BC4: time (0,2)  

u0BC1   = 0.6 * ones(1,numBoundaryConditionPoints(1));                          % BC1: 
u0BC2   = 0.6 * ones(1,numBoundaryConditionPoints(1)); 
u0BC3   = 0.6 * ones(1,numBoundaryConditionPoints(1)); 
u0BC4   = 0.6 * ones(1,numBoundaryConditionPoints(1));

%% IC points

numInitialConditionPoints  =  1000;

points = 2.0 * lhsdesign (numInitialConditionPoints , 3); 
x0IC  = points (:,1)'; 
y0IC  = points (:,2)';
%x0IC = linspace (0, 1, numInitialConditionPoints); 
%y0IC = linspace (0, 1, numInitialConditionPoints); 
t0IC = zeros    (1,    numInitialConditionPoints);
u0IC = sin(pi*x0IC).*sin(pi*y0IC);       

u0IC = 0.6 + 0.4*sin(pi()*x0IC/2).*sin(pi()*y0IC/2);    % Initial condition

plot3(x0IC,y0IC,u0IC)

%% collocation points
%
numInternalCollocationPoints  =  10000;
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

dlX0IC   = dlarray (x0IC, 'CB');
dlY0IC   = dlarray (y0IC, 'CB');
dlT0IC   = dlarray (t0IC, 'CB');
dlU0IC   = dlarray (u0IC, 'CB');

dlX0BC1  = dlarray (x0BC1, 'CB');
dlX0BC2  = dlarray (x0BC2, 'CB');
dlX0BC3  = dlarray (x0BC3, 'CB');
dlX0BC4  = dlarray (x0BC4, 'CB');

dlY0BC1  = dlarray (y0BC1, 'CB');
dlY0BC2  = dlarray (y0BC2, 'CB');
dlY0BC3  = dlarray (y0BC3, 'CB');
dlY0BC4  = dlarray (y0BC4, 'CB');

dlT0BC1	 = dlarray (t0BC1, 'CB');
dlT0BC2	 = dlarray (t0BC2, 'CB');
dlT0BC3	 = dlarray (t0BC3, 'CB');
dlT0BC4	 = dlarray (t0BC4, 'CB');
		
dlU0BC1	 = dlarray (u0BC1, 'CB');
dlU0BC2	 = dlarray (u0BC2, 'CB');
dlU0BC3	 = dlarray (u0BC3, 'CB');
dlU0BC4	 = dlarray (u0BC4, 'CB');



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
    lossF_hist ( epoch, 1 )  = lossF ;
    lossU_hist ( epoch, 1 )  = lossU ;
   
end


%% Evaluate model accuracy
%
% 

tTest                     = [0.0 0.025 0.075 0.1];
L2_error_all              = zeros(length(tTest),1);

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

        UPred_temp = modelU(parameters, dlXTest, dlYTest, dlTTest);
        dlUPred(j,:) = double(extractdata(UPred_temp));
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
