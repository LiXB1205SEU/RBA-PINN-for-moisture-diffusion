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
    %
    f1 = Ut - c_local.*(Uxx) - c_local_dh.*(Ux.*Ux);
    %f1 = Ut - c_local.*(Uxx) - c_local_dh.* (Ux.*Ux);



    
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
    %f1 = Ut - c*(Uxx) ;
    %

    lossF_unweighted = mean((f1).^2);
    lossF_weighted = mean((rsum.*f1).^2);
    lossF = lossF_weighted;

   % Combine losses .

    loss = lossF;



    % Calculate gradients with respect to the learnable parameters .
    gradients = dlgradient (loss, parameters);


end

