% Model Function
%
function [dlU, Ux] = modelU_deri (parameters, dlX, dlT)

    dlXT     = [dlX; dlT];
    



    numLayers = numel(fieldnames(parameters));

    % First fully connect operation .
    %
    weights = parameters.fc1.Weights;
    bias    = parameters.fc1.Bias;
    %
    dlU     = fullyconnect (dlXT, weights, bias);

    %
    %  tanh and fully connect operations for remaining layers .
    %
    for i = 2: numLayers
        name = "fc" + i;

        dlU = sin(dlU);

        weights = parameters.(name).Weights ;
        bias    = parameters.(name).Bias ;

        dlU      = fullyconnect (dlU , weights , bias );
    end
    
    
    Ux  =  dlgradient( sum ( dlU , 'all'), dlX , 'EnableHigherDerivatives', true );
end





