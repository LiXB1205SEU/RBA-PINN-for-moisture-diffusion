% Model Gradients Function
%
function [gradients, loss, lossF_weighted, lossF_unweighted, lossU, rsum_new] = modelGradients (parameters, dlX, dlT,...
	                                        dlX0IC, dlX0BC1,  dlX0BC2,...
                                            dlT0IC, dlT0BC1,  dlT0BC2,...
                                            dlU0IC, dlU0BC1,  dlU0BC2,...
                                            c,epoch, rsum_old)

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

    f1 = Ut - c*(Uxx);
    
    gamma = 0.999;

    if epoch == 1
        eta = 1;
    else
        eta = 0.01;
    end

    

    epsDenom = 1e-12;
    r_norm = eta .* abs(f1) ./ (max(abs(f1), [], 'all') + epsDenom);


    rsum = gamma .* rsum_old + r_norm;

    rsum_new = rsum;


    %
    % Calculate lossF . Enforce the PDE.
    %
    %
    


    lossF_unweighted = mean((f1).^2);
    lossF_weighted = mean((rsum.*f1).^2);
    lossF = lossF_weighted;

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

