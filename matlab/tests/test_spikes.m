% test functions to manipulate spikes trains

addpath('..');


% ----- spikes2states

spikes = [[1,0,1,0];[0,0,1,0];[1,0,1,0];[0,0,0,0];[1,0,1,0];[0,0,1,0];[1,1,1,1]];
states = spikes2states(spikes);

desired = [11,  3, 11,  1, 11,  3, 16];
for i=1:length(desired)
    assert(states(i)==desired(i));
end



% ----- states2distribution

distr = states2distribution(states, 4);

desired = [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1];
for i=1:length(desired)
    assert(distr(i)==desired(i));
end


% sample from distribution, reconstruct distribution
nchannels = 3;
real_distr = [0.1, 0.3, 0.05, 0.05, 0.2, 0.1, 0.1, 0.1];

states = randsample(1:2^nchannels,200000,true,real_distr);
% compute distribution
distr = states2distribution(states, nchannels) / length(states);

assert(max(abs(real_distr-distr)) < 1e-2);
