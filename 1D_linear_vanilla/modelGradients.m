% Model Gradients Function
%
function [gradients, loss, lossF, lossU] = modelGradients (parameters, dlX, dlT,...
	                                        dlX0IC, dlX0BC1,  dlX0BC2,...
                                            dlT0IC, dlT0BC1,  dlT0BC2,...
                                            dlU0IC, dlU0BC1,  dlU0BC2,...
                                            c)

    % Make predictions with the initial conditions .
    U = modelU (parameters, dlX, dlT);

    % size ( extractdata (U))
    % Calculate derivatives with respect to T.
    %
    gradientsU = dlgradient (sum(U, 'all'), {dlX, dlT}, 'EnableHigherDerivatives', true);
    %
    Ux = gradientsU {1};
    Ut = gradientsU {2};
    %
    % Calculate second - order derivatives with respect to X.
    Uxx  =  dlgradient( sum ( Ux , 'all'), dlX , 'EnableHigherDerivatives', true );
    % Calculate second - order derivatives with respect to T.
    Utt  =  dlgradient( sum ( Ut , 'all'), dlT , 'EnableHigherDerivatives', true );

    %c_local = c * ones (1, length (U)) - U/1.5;
    %
    % Calculate lossF . Enforce the PDE.
    %
    f1 = Ut - c*(Uxx) ;
    %
    %f1 = Ut - c_local.*(Uxx) ;

    lossF = mean(f1.^2);

    % Calculate lossU . Enforce initial and boundary conditions .

    dlU0ICPred    = modelU (parameters, dlX0IC, dlT0IC);
    lossU0IC      = l2loss    (dlU0ICPred, dlU0IC );
  

    dlU0BC1Pred   = modelU(parameters, dlX0BC1, dlT0BC1);
    lossU0BC1     = l2loss( dlU0BC1Pred , dlU0BC1);

    dlU0BC2Pred   = modelU(parameters, dlX0BC2, dlT0BC2);
    lossU0BC2     = l2loss( dlU0BC2Pred , dlU0BC2);

    lossU         = lossU0IC + lossU0BC1 + lossU0BC2;

    % Combine losses .

    loss = lossF +  lossU ;


    % Calculate gradients with respect to the learnable parameters .
    gradients = dlgradient (loss, parameters);


end

