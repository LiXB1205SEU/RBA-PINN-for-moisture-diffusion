function [gradients, loss, lossF, lossU] = modelGradients(parameters, dlX, dlY, dlT, ...
                                            dlX0IC,  dlY0IC,  dlT0IC,  dlU0IC, ...
                                            dlX0BC1, dlY0BC1, dlT0BC1, dlU0BC1, ...
                                            dlX0BC2, dlY0BC2, dlT0BC2, dlU0BC2, ...
                                            dlX0BC3, dlY0BC3, dlT0BC3, dlU0BC3, ...
                                            dlX0BC4, dlY0BC4, dlT0BC4, dlU0BC4, ...
                                            c)

    % ------------------------------------------------------------------
    % Vanilla PINN: network output
    % ------------------------------------------------------------------
    U = modelU(parameters, dlX, dlY, dlT);

    % ------------------------------------------------------------------
    % First-order derivatives
    % ------------------------------------------------------------------
    gradientsU = dlgradient(sum(U, 'all'), {dlX, dlY, dlT}, ...
        'EnableHigherDerivatives', true);

    Ux = gradientsU{1};
    Uy = gradientsU{2};
    Ut = gradientsU{3};

    % ------------------------------------------------------------------
    % Second-order derivatives
    % ------------------------------------------------------------------
    Uxx = dlgradient(sum(Ux, 'all'), dlX, 'EnableHigherDerivatives', true);
    Uyy = dlgradient(sum(Uy, 'all'), dlY, 'EnableHigherDerivatives', true);

    % ------------------------------------------------------------------
    % Nonlinear diffusion coefficient c_local(U)
    %
    % c_local(U) = 1 / ( 1 + ((1-U)/(1-U_c))^2 )
    %
    % dc_local/dU = 2(1-U) / [ (1-U_c)^2 * (1 + ((1-U)/(1-U_c))^2 )^2 ]
    %
    % U is clipped to [0,1] following the original implementation
    % ------------------------------------------------------------------
    U_c = 0.8;

    U_clip = min(max(U, 0.0), 1.0);
    s = (1 - U_clip) ./ (1 - U_c);

    c_local = 1 ./ (1 + s.^2);
    c_local_dh = 2 .* (1 - U_clip) ./ ((1 - U_c)^2 .* (1 + s.^2).^2);

    % ------------------------------------------------------------------
    % PDE residual
    %
    % 2D nonlinear diffusion:
    % Ut - div(c_local grad(U)) = 0
    %
    % Expanded form:
    % Ut - c_local*(Uxx + Uyy) - c_local_dh*(Ux^2 + Uy^2) = 0
    % ------------------------------------------------------------------
    f1 = Ut - c_local .* (Uxx + Uyy) - c_local_dh .* (Ux.^2 + Uy.^2);

    lossF = mean(f1.^2);

    % ------------------------------------------------------------------
    % Initial condition loss
    % ------------------------------------------------------------------
    dlU0ICPred = modelU(parameters, dlX0IC, dlY0IC, dlT0IC);
    lossU0IC = l2loss(dlU0ICPred, dlU0IC);

    % ------------------------------------------------------------------
    % Boundary condition losses
    % ------------------------------------------------------------------
    dlU0BC1Pred = modelU(parameters, dlX0BC1, dlY0BC1, dlT0BC1);
    lossU0BC1 = l2loss(dlU0BC1Pred, dlU0BC1);

    dlU0BC2Pred = modelU(parameters, dlX0BC2, dlY0BC2, dlT0BC2);
    lossU0BC2 = l2loss(dlU0BC2Pred, dlU0BC2);

    dlU0BC3Pred = modelU(parameters, dlX0BC3, dlY0BC3, dlT0BC3);
    lossU0BC3 = l2loss(dlU0BC3Pred, dlU0BC3);

    dlU0BC4Pred = modelU(parameters, dlX0BC4, dlY0BC4, dlT0BC4);
    lossU0BC4 = l2loss(dlU0BC4Pred, dlU0BC4);

    lossU = lossU0IC + lossU0BC1 + lossU0BC2 + lossU0BC3 + lossU0BC4;

    % ------------------------------------------------------------------
    % Total loss
    % ------------------------------------------------------------------
    loss = lossF + lossU;

    % ------------------------------------------------------------------
    % Gradients w.r.t. learnable parameters
    % ------------------------------------------------------------------
    gradients = dlgradient(loss, parameters);

end