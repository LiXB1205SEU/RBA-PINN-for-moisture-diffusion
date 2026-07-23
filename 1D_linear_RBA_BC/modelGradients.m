% Model Gradients Function
%
function [gradients, loss, lossF_weighted, lossF_unweighted, rsum_new] = modelGradients (parameters, dlX, dlT, c,epoch, rsum_old)
    %                                        
    %
    % Make predictions with the initial conditions .
    U = dlT.*modelU (parameters, dlX, dlT).*dlX.*(dlX-2) + 0.6 + 0.4 * sin( pi() * dlX / 2.0);

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

   % Combine losses .

    loss = lossF;



    % Calculate gradients with respect to the learnable parameters .
    gradients = dlgradient (loss, parameters);


end

