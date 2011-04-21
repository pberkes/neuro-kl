function result = mean_KL_estimate(alpha, beta)
% MEAN_KL_ESTIMATE computes a mean estimate of KL divergence
%
%   KL = MEAN_KL_ESTIMATE(ALPHA, BETA) computes the mean of a Bayesian
%   estimator of the KL divergence between two discrete distributions.
%
%   IMPORTANT: alpha and beta are not the values of the distribution,
%   but the parameters of the Dirichlet posterior of the two distributions.
%   Use KL_ESTIMATION to estimate KL directly from data sequences.

    alpha0=sum(alpha);
    beta0=sum(beta);

    result = mean_H_estimate(alpha) - ...
        ( sum((alpha/alpha0).*(psi(beta)-psi(beta0))) )/log(2);
