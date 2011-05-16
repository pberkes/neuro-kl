function [Hest] = h_estimation(P, alpha, do_shuffle, number_channels)
% H_ESTIMATION estimate negative entropy of discrete states sequence
%
%   HEST = H_ESTIMATION(P, ALPHA, DO_SHUFFLE, NUMBER_CHANNELS) computes an
%   estimation of the negative entropy of a sequence of states.
%   The estimation is done using a Bayesian estimator, followed by an
%   extrapolation step to reduce biases in the estimation.
%
%   Parameters:
%   P: sequence of states
%   ALPHA: paramters of the Dirichlet prior (usually set to 1)
%   DO_SHUFFLE: if set to non-zero value, the data is shuffled before
%   NUMBER_CHANNELS: number of channels in the data

% Copyright (c) 2011 Pietro Berkes and Dmitriy Lisitsyn
% License: GPL v3


    parts=[4,2,1];

    N=length(P)./parts;

    H_results=cell(length(parts),1);
    H_means=zeros(length(parts),1);
    segment=cell(length(parts),1);

    for part=1:length(parts)
        segment{part}=floor([0 (1:parts(part)).*N(part)]);
    end

    % permute datapoints if requested
    if do_shuffle==1
        elSeq=randperm(length(P));
        P=P(elSeq);
    end

    for part=1:length(parts)
        for seg=1:parts(part)
            P_seg=P( (segment{part}(seg)+1):(segment{part}(seg+1)) );
            % transform state sequences to distributions
            p=states2distribution(P_seg, number_channels);
            % compute mean entropy for this part of the data
            H_results{part}(seg)=mean_H_estimate(p+alpha);
        end

        % mean of the means for each data size+
        H_means(part,1)=mean(H_results{part});
    end

    [p,S,mu] = polyfit(N,(N.*N).*H_means',2);
    Hest=p(1)/(mu(2).^2);
end
