function result=mean_H_estimate(alpha)
% MEAN_H_ESTIMATE computes a mean estimate of the negative entropy
%
%   H = MEAN_H_ESTIMATE(ALPHA, BETA) computes the mean of a Bayesian
%   estimator of the negative entropy of a distribution
%
%   IMPORTANT: alpha does not contain the values of the distribution,
%   but the parameters of the Dirichlet posterior of the distribution.
%   Use H_ESTIMATION to estimate H directly from data sequences.

    alpha0=sum(alpha);

    result = ( sum(alpha.*psi(alpha+1))/alpha0 - psi(alpha0+1) )./log(2);
