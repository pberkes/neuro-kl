function plot_KL_estimation(KL_means, N)
% PLOT_KL_ESTIMATION visualize KL extrapolation results
%   
%
%   PLOT_KL_ESTIMATION(KL_means, N) plots the intermediate results of the
%    KL estimation process for visual inspection and diagnosis.
%    In the plot, the blue circles represent the mean KL estimation at
%    N/4, N/2, and N. The dotted line respresents the polynomial fitted
%    to the data. The red dashed line is the final value of the KL
%    estimation.
%
%   Parameters:
%   KL_means: mean KL estimates for N/4, N/2, and N;
%             returned by KL_ESTIMATION
%   N: vector with number of points used for extrapolation;
%             returned by KL_ESTIMATION

% Copyright (c) 2011 Pietro Berkes and Dmitriy Lisitsyn
% License: GPL v3

    [p,S,mu] = polyfit(N,(N.*N).*KL_means',2);
    kl = p(1)/(mu(2).^2);

    x = linspace(N(1)/2, N(end)*5, 100);
    y = polyval(p, x, S, mu);
    clf;
    hold on;
    plot(x, y./(x.^2), 'b:');
    plot(N, KL_means,'bo');
    plot([x(1), x(end)], [kl kl], 'r--');
    