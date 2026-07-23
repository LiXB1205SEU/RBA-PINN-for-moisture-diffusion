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
    
    % ---------
    % calculation of c_local
    %
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
    %f1 = Ut - c_local.*(Uxx);
    f1 = Ut - c_local.*(Uxx) - c_local_dh.* (Ux.*Ux);

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

