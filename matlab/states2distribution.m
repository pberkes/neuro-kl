function distribution = states2distribution(states, number_channels)
% STATES2DISTRIBUTION convert an array of states to a distribution.
%
%   DISTRIBUTION = STATES2DISTRIBUTION(STATES, NUMBER_CHANNELS) takes an
%   array of neural states (as given by SPIKES2STATES) and the number
%   of channels in the data, and returns an unnormalized distribution
%   over states, i.e., the histogram over all states.
%
%   example:
%   spikes = [[1,0,1,0];[0,0,1,0];[1,0,1,0];[0,0,0,0];[1,0,1,0];[0,0,1,0]];
%   disp(states2distribution(spikes2states(spikes), 4)')
%      1   0   2   0   0   0   0   0   0   0   3   0   ...

% Copyright (c) 2011 Pietro Berkes and Dmitriy Lisitsyn
% License: GPL v3

    nbins = 2^number_channels;
    distribution = histc(states, (1:(nbins+1))-0.5);
    % the last bin returned by histc has a special meaning
    distribution = distribution(1:(end-1));
