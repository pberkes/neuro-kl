function result=spikes2states(spikes)
% SPIKES2STATES converts spikes trains into state numbers.
%
%   STATES = SPIKES2STATES(SPIKES, NUMBER_CHANNELS) takes a 2D binary
%   array (time x channels) and convert it to an array of states by
%   by numbering all possible neural activity patterns. The numbering
%   of the states goes from 1 to 2^(number of channels).
%
%   For example:
%   spikes = [[1,0,1,0];[0,0,1,0];[1,0,1,0];[0,0,0,0];[1,0,1,0];[0,0,1,0]];
%   disp(spikes2states(spikes)');
%     6     5     6     1     6     5

% Copyright (c) 2011 Pietro Berkes and Dmitriy Lisitsyn
% License: GPL v3


    number_channels = size(spikes, 2);

    conv = 2.^(0:(number_channels-1));
    result = spikes*conv' + 1;
    