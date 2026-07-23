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


    % ---------
    % calculation of c_local
    %
    U_c           = 0.8;

    U_vector = extractdata(U);

    for i = 1 : length (U_vector)
        term = U_vector(i);
        %
        term = min (term, 1.0);
        term = max (term, 0.0);

        %
        c_local_term1   = ( (1-term)/(1-U_c) );
        c_local_term2   = c_local_term1*c_local_term1;
        c_local_term3   = 1/(1+c_local_term2);
        c_local(i)      = c_local_term3;

        c_local_dh_term1  =  -1;
        c_local_dh_term2  =  c_local_term3 * c_local_term3;
        c_local_dh_term3  =  2 *  (1-term)/(1-U_c);
        c_local_dh_term4  =  -1 /(1-U_c);
        


        c_local_dh(i) = c_local_dh_term1*c_local_dh_term2*c_local_dh_term3*c_local_dh_term4;
    end
    

    
    c_local     =  dlarray(c_local);   
    c_local_dh  =  dlarray(c_local_dh);
 
    %
    % Calculate lossF . Enforce the PDE.
    %
    %f1 = Ut - c*(Uxx) ;
    %不对c_local求偏导（不牛逼）
    %f1 = Ut - c_local.*(Uxx);
    %对c_local 求偏导（牛逼）
    f1 = Ut - c_local.*(Uxx) - c_local_dh.* (Ux.*Ux);

    lossF = mean(f1.^2);

   

    % Combine losses .

    loss = lossF;


    % Calculate gradients with respect to the learnable parameters .
    gradients = dlgradient (loss, parameters);


end

