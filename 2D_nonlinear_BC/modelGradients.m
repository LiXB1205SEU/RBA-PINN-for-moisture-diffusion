% Model Gradients Function
%
function [gradients, loss] = modelGradients (parameters, dlX, dlY, dlT, c)

    % Make predictions with the initial conditions .
    U_temp1 = modelU (parameters,  dlX, dlY, dlT);

    U_temp2 = dlT.*U_temp1.*dlX.*(dlX-2).*dlY.*(dlY-2) + 0.6 + 0.4 * sin(pi()*dlX/2).*sin(pi()*dlY/2);  

    U =  U_temp2;



    % size ( extractdata (U))
    % Calculate derivatives with respect to T.
    %
    gradientsU = dlgradient (sum(U, 'all'), {dlX ,dlY ,dlT}, 'EnableHigherDerivatives', true);
    %
    Ux = gradientsU {1};
    Uy = gradientsU {2};
    Ut = gradientsU {3};
    %
    % Calculate second - order derivatives with respect to X.
    Uxx  =  dlgradient( sum ( Ux , 'all'), dlX , 'EnableHigherDerivatives', true );
    % Calculate second - order derivatives with respect to Y.
    Uyy  =  dlgradient( sum ( Uy , 'all'), dlY , 'EnableHigherDerivatives', true );
%   % Calculate second - order derivatives with respect to T.
%   Utt  =  dlgradient( sum ( Ut , 'all'), dlT , 'EnableHigherDerivatives', true );

    
    % very coefficient
    U_c = 0.8;

    U_clip = min(max(U, 0.0), 1.0);
    s = (1 - U_clip) ./ (1 - U_c);

    c_local = 1 ./ (1 + s.^2);
    c_local_dh = 2 .* (1 - U_clip) ./ ((1 - U_c)^2 .* (1 + s.^2).^2);

    %
    % Calculate lossF . Enforce the PDE.
    %
    f1 = Ut - c_local .* (Uxx + Uyy) - c_local_dh .* (Ux.^2 + Uy.^2);

    lossF = mean(f1.^2);

    % Combine losses .

    loss = lossF ;




    % Calculate gradients with respect to the learnable parameters .
    gradients = dlgradient (loss, parameters );



  

end


