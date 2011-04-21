% test function to estimate entropy and KL divergence


% ---- entropy

nchannels = 4;
% uniform distribution, entropy = number of channels
alpha = zeros(2^nchannels, 1) + 10000;
hest = -mean_H_estimate(alpha);
assert(abs(hest-nchannels) < 1e-4);

% sample from distribution, reconstruct entropy
nchannels = 3;
distr = [0.1, 0.3, 0.05, 0.05, 0.2, 0.1, 0.1, 0.1];
real_entropy = -sum(distr.*log2(distr));

states = randsample(1:2^nchannels,100000,true,distr);
% compute histogram
distr = states2distribution(states, nchannels);

hest1 = - mean_H_estimate(distr);
assert(abs(real_entropy - hest1) < 1e-2);

hest2 = - h_estimation(states, 1, 1, nchannels);
assert(abs(real_entropy - hest2) < 1e-2);

hest3 = - h_estimation(states, 1, 0, nchannels);
assert(abs(real_entropy - hest3) < 1e-2);



% ---- KL divergence

% same states, KL divergence should converge to zero
states2 = states(randperm(length(states)));

kl1 = mean_KL_estimate(states2distribution(states, nchannels), ...
          states2distribution(states2, nchannels));
assert(kl1 < 1e-3);

kl2 = kl_estimation(states, states2, 1, 1, 1, nchannels);
assert(abs(kl2) < 1e-3);

kl3 = kl_estimation(states, states2, 1, 1, 0, nchannels);
assert(abs(kl3) < 1e-3);


% sample from distributions, reconstruct KL
nchannels = 3;
distr = [0.1, 0.3, 0.05, 0.05, 0.2, 0.1, 0.1, 0.1];
distr2 = [0.4, 0.2, 0.01, 0.09, 0.05, 0.05, 0.03, 0.17];
real_kl = sum( distr .* log2( distr./distr2 ) );

kl1 = mean_KL_estimate(distr*100000, distr2*100000);
assert(abs(real_kl - kl1) < 1e-2);

states = randsample(1:2^nchannels,100000,true,distr);
states2 = randsample(1:2^nchannels,100000,true,distr2);

kl2 = kl_estimation(states, states2, 1, 1, 1, nchannels);
assert(abs(real_kl - kl2) < 1e-2);

kl3 = kl_estimation(states, states2, 1, 1, 0, nchannels);
assert(abs(real_kl - kl3) < 1e-2);
