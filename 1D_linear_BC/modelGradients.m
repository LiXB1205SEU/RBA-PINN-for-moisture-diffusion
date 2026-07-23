% Model Gradients Function
%
function [gradients, loss] = modelGradients (parameters, dlX, dlT, c)

    % Make predictions with the initial conditions .

    U =  dlT.*modelU (parameters, dlX, dlT).*dlX.*(dlX-2) + 0.6 + 0.4 * sin( pi() * dlX / 2.0);


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


    %
    % Calculate lossF . Enforce the PDE.
    %
    f1 = Ut - c*(Uxx) ;
    %

    lossF = mean(f1.^2);

   

    % Combine losses .

    loss = lossF;


    % Calculate gradients with respect to the learnable parameters .
    gradients = dlgradient (loss, parameters);


end

