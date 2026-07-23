% Model Function
%
function dlU = modelU (parameters, dlX, dlY, dlT)

    dlXYT     = [dlX; dlY; dlT];

    numLayers = numel(fieldnames(parameters));

    % First fully connect operation .
    %
    weights = parameters.fc1.Weights;
    bias    = parameters.fc1.Bias;
    %
    dlU     = fullyconnect (dlXYT, weights, bias);

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
end





