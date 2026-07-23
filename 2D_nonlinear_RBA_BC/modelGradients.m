function [gradients, loss, lossF_weighted, lossF_unweighted, rsum_new] = modelGradients(parameters, dlX, dlY, dlT, c, epoch, rsum_old, gamma, eta_init, eta_main, rbaScale)

    if nargin < 8 || isempty(gamma); gamma = 0.999; end
    if nargin < 9 || isempty(eta_init); eta_init = 1; end
    if nargin < 10 || isempty(eta_main); eta_main = 0.01; end
    if nargin < 11 || isempty(rbaScale); rbaScale = 1; end

    % Hard IC/BC constraint
    U = dlT .* modelU(parameters, dlX, dlY, dlT) ...
        .* dlX .* (dlX-2) .* dlY .* (dlY-2) ...
        + 0.6 + 0.4*sin(pi()*dlX/2).*sin(pi()*dlY/2);

    % First derivatives
    gradientsU = dlgradient(sum(U,'all'), {dlX, dlY, dlT}, 'EnableHigherDerivatives', true);
    Ux = gradientsU{1};
    Uy = gradientsU{2};
    Ut = gradientsU{3};

    % Second derivatives
    Uxx = dlgradient(sum(Ux,'all'), dlX, 'EnableHigherDerivatives', true);
    Uyy = dlgradient(sum(Uy,'all'), dlY, 'EnableHigherDerivatives', true);

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

   
    % RBA update
    if epoch == 1
        eta = eta_init;
    else
        eta = eta_main;
    end

    epsDenom = 1e-12;
    r_norm = eta .* abs(f1) ./ (max(abs(f1), [], 'all') + epsDenom);
    rsum_raw = gamma * rsum_old + r_norm;
    rsum_new = dlarray(extractdata(rsum_raw), 'CB');
    rsum_normalized = rsum_new ./ max(rbaScale, epsDenom);

    % Weighted and unweighted residual losses
    lossF_unweighted = mean((f1).^2);
    lossF_weighted = mean((rsum_normalized .* f1).^2);
    lossF = lossF_weighted;
    loss = lossF;

    % Gradients
    gradients = dlgradient(loss, parameters);
end
