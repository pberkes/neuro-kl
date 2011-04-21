# Author: Pietro Berkes < pietro _DOT_ berkes _AT_ googlemail _DOT_ com >
# Copyright (c) 2011 Pietro Berkes
# License: GPL v3

import scipy
from scipy import log, log2, array, zeros
from scipy.special import digamma

def kl(p, q):
    """Compute the KL divergence between two discrete probability distributions

    The calculation is done directly using the Kullback-Leibler divergence,
    KL( p || q ) = sum_{x} p(x) log_2( p(x) / q(x) )

    Base 2 logarithm is used, so that returned values is measured in bits.
    """
    
    if (p==0.).sum()+(q==0.).sum() > 0:
        raise Exception, "Zero bins found"
    return (p*(log2(p) - log2(q))).sum()

def entropy(p):
    """Compute the negative entropy of a discrete probability distribution.

    The calculation is done directly using the entropy definition,
    H(p) = sum_{x} p(x) log_2( p(x) )
    
    Base 2 logarithm is used, so that returned values is measured in bits.
    """
    
    if (p==0.).sum() > 0:
        raise Exception, "Zero bins found"
    return (p*log2(p)).sum()

def mean_H_estimate(alpha):
    """Compute the mean of a Bayesian estimator of negative entropy.

    The argument, `alpha`, contains the parameters of a Dirichlet
    posterior over a distribution, `p`. The function returns
    the mean estimation < H(p) >_P, in bits.

    Input arguments:
    alpha -- parameters of the Dirichlet posterior over p
    """
    alpha0 = alpha.sum()
    res = (alpha*digamma(alpha+1)).sum()/alpha0 - digamma(alpha0+1)
    return res/log(2.)

def mean_KL_estimate(alpha, beta):
    """Compute the mean of a Bayesian estimator of KL divergence.

    The arguments, `alpha` and `beta` are the parameters of a Dirichlet
    posterior over two distributions, `p` and `q`. The function returns
    the mean estimation < < KL(p||q) >_P(alpha) >_Q(beta), in bits

    Input arguments:
    alpha -- parameters of the Dirichlet posterior over p
    beta -- parameters of the Dirichlet posterior over q

    Output:
    Entropy estimate
    """
    
    alpha0 = alpha.sum()
    beta0 = beta.sum()
    
    res = mean_H_estimate(alpha) \
          - (alpha/alpha0*(digamma(beta) - digamma(beta0))).sum()/log(2.)
    return res

def kl_estimation(p_dict, q_dict, npoints, alpha=1., Ns=None):
    """Compute an estimation of the KL divergence between two distributions.

    The estimation is done using a Bayesian estimator, followed by an
    extrapolation step to reduce biases in the estimation.

    Input arguments:
    p_dict -- a dictionary containing the distribution of states for the first
              distribution, as created by the function `states2dict`
    q_dict -- a dictionary containing the distribution of states for the second
              distribution, as created by the function `states2dict`
    npoints -- total number of points used to estimate the distribution
    alpha -- paramters of the Dirichlet prior (usually set to 1)

    Output:
    kl_est -- estimate of the KL divergence of p and q
    h_est -- estimate of the entropy of p
    """

    ns = array([4,2,1], dtype='int64')
    h_estimate = zeros((len(ns),))
    kl_estimate = zeros((len(ns),))
    for j, d in enumerate(ns):
        h_est = zeros((d,))
        kl_est = zeros((d,))
        for i in range(d):
            p = p_dict[d][i].flatten() # for joint distributions
            q = q_dict[d][i].flatten()
            h_est[i] = mean_H_estimate(p + alpha)
            kl_est[i] = mean_KL_estimate(p + alpha, q + alpha)
        h_estimate[j] = h_est.mean()
        kl_estimate[j] = kl_est.mean()
        
    # extrapolate
    if Ns is None:
        Ns = npoints/ns
    Ns = Ns.astype('d')
    h_extr = scipy.polyfit(Ns, Ns*Ns * h_estimate, 2)
    kl_extr = scipy.polyfit(Ns, Ns*Ns * kl_estimate, 2)
    return kl_extr[0], h_extr[0]

def h_estimation(p_dict, npoints, alpha=1., Ns=None):
    """Compute an estimation of the negative entropy of a distribution.

    The estimation is done using a Bayesian estimator, followed by an
    extrapolation step to reduce biases in the estimation.

    Input arguments:
    p_dict -- a dictionary containing the distribution of states, as created
              by the function `states2dict`
    npoints -- total number of points used to estimate the distribution
    alpha -- paramters of the Dirichlet prior (usually set to 1)
    """
    ns = array([4,2,1], dtype='int64')
    h_estimate = zeros((len(ns),))
    for j, d in enumerate(ns):
        h_est = zeros((d,))
        for i in range(d):
            p = p_dict[d][i].flatten() # for joint distributions
            h_est[i] = mean_H_estimate(p + alpha)
        h_estimate[j] = h_est.mean()
    # extrapolate
    if Ns is None:
        Ns = npoints/ns
    Ns = Ns.astype('d')
    h_extr = scipy.polyfit(Ns, Ns*Ns * h_estimate, 2)
    return h_extr[0]

# ################ DATA MANIPULATION

def spikes2states(spikes):
    """Convert a sequence of binarized spikes to a sequence of state numbers.

    Input arguments:
    spikes -- spikes trains: 2D binary array, each column a different unit,
              each row a time point
    """

    # check that the incoming array is binary
    if not scipy.all(scipy.logical_and(spikes>=0, spikes<=1)):
        raise ValueError('Input array must be binary')

    nchannels = spikes.shape[1]
    # convert binary sequence to decimal numbers
    pow2 = array([2**i for i in range(nchannels-1,-1,-1)])
    return (spikes*pow2).sum(axis=1)

def states2distr(states, nchannels, normed=False):
    """Return distribution over states.

    States are the decimal number of neural activity patterns, where the patterns
    are interpreted as binary words.

    E.g., if on 4 channels the activity pattern is 1 0 1 0 (spikes on
    channels 0 and 2, no spikes otherwise) the corresponding state is 10.

    See also 'spikes2states'.

    Input arguments:
    states -- array of states
    nchannels -- total number of channels (used to determine the maximum number
                 of states)
    normed -- if False return count of states, otherwise the fraction of the total

    Output:
    Array of length 2**nchannels, containing the histogram of states
    """
    bins = scipy.arange(2**nchannels+1)
    distr, ledges = scipy.histogram(states, bins=bins, normed=normed)
    return distr.astype('d')

def states2dict(states, nchannels, npoints=None, fractions=[1,2,4], shuffle=True):
    """Return dictionary with distribution over states for fractions of data.
    The distributions are *not* normalized, as required by other routines
    (e.g., KL estimation routines).

    This function is intented to be used with the KL and entropy estimation
    functions, kl_estimation and h_estimation.

    Input arguments:
    states -- array of states
    nchannels -- total number of channels (used to determine the maximum number
                 of states)
    npoints -- number of data points Default: None, meaning the full length of states
    fractions -- fractions of the data. For example, fractions=[1,2,4] will create
                 3 entries in the dictionary, based on the full data (N datapoints),
                 half the data (2 x N/2 points), and one quarter of the data
                 (4 x N/4 points). Default: [1,2,4]
    shuffle -- If True, data points are shuffled before computing the dictionaries
               to avoid trends in the data

    Output:
    Dictionary distr[fraction][distr_nr]. Keys are fractions (as given by input
    argument), values are lists of distributions.
    """
    if npoints is None:
        npoints = states.shape[0]
    if shuffle:
        states = states.copy()
        p = scipy.random.permutation(states.shape[0])
        states = scipy.take(states, p)
    distr = {}
    for d in fractions:
        distr[d] = [None]*d
        block_len = npoints//d
        for i in range(d):
            part_y = states[i*block_len:(i+1)*block_len]
            distr[d][i] = states2distr(part_y, nchannels, normed=False)
    _check_dict_consistency(distr, npoints)
    return distr

def spikes2indep_dict(spikes, npoints=None, fractions=[1,2,4]):
    """Return dictionary with distribution over states assuming independence.

    This function works like `states2dict`, but takes as input an array of
    spikes, and return a dictionary of states by removing all dependencies
    between channels.

    The distributions are *not* normalized, as required by other routines
    (e.g., KL estimation routines).
    
    Input arguments:
    spikes -- spikes trains: 2D binary array, each column a different unit,
              each row a time point
    nchannels -- total number of channels (used to determine the maximum number
                 of states)
    npoints -- number of data points Default: None, meaning the full length of states
    fractions -- fractions of the data. For example, fractions=[1,2,4] will create
                 3 entries in the dictionary, based on the full data (N datapoints),
                 half the data (2 x N/2 points), and one quarter of the data
                 (4 x N/4 points). Default: [1,2,4]
    shuffle -- If True, data points are shuffled before computing the dictionaries
               to avoid trends in the data

    Output:
    Dictionary distr[fraction][distr_nr]. Keys are fractions (as given by input
    argument), values are lists of distributions.
    """
    if npoints is None:
        npoints = spikes.shape[0]
    nchannels = spikes.shape[1]

    # p1[i] = p(channel_i = 1)
    p1 = spikes.sum(0).astype('d')/spikes.shape[0]
    # distribution over states given independence
    nbins = 2**nchannels
    indep_distr = zeros((nbins,))
    # cycle over states
    for s in range(nbins):
        # get binary pattern
        s_bin = scipy.binary_repr(s, width=nchannels) if s>0 else '0'*nchannels
        # compute probability for independent case
        prob = [(p1[k] if s_bin[k]=='1' else 1.-p1[k]) for k in range(nchannels) ]
        indep_distr[s] = scipy.prod(prob)
    # construct dictionary as for normal case
    distr = {}
    for d in fractions:
        l = npoints/d
        distr[d] = [indep_distr*l] * d
    _check_dict_consistency(distr, npoints)
    return distr

def _check_dict_consistency(distr, npoints):
    control = []
    if 2 in distr.keys(): control.append(2)
    if 4 in distr.keys(): control.append(4)
    for d in control:
        sm = 0.
        for i in range(d):
            sm += array(distr[d][i])
        assert scipy.sum(sm) - npoints < 1e-4
        assert scipy.all((sm.astype('int32') - array(distr[1][0])) < 1e-4)

def transition_matrix(y, nstates, dt=1):
    """
    Return transition matrix histogram (i.e., counts)
    """
    tr, tmp, tmp = scipy.histogram2d(y[:-dt], y[dt:], bins=range(nstates+1))
    return tr

def states2transition_dict(states, nchannels, dt=1, ds=[1,2,4], indep=False):
    """Return dictionary with transition probability distribution
    for N, N/2, N/4.
    indep -- if True, compute transition distribution assuming independence in time
    """
    npoints = states.shape[0]
    nstates = 2**nchannels
    #print 'npoints, nstates, nchannels', npoints, nstates, nchannels
    distr = {}
    for d in ds:
        distr[d] = [None]*d
        block_len = npoints/d
        for i in range(d):
            part_y = states[i*block_len:(i+1)*block_len]
            if not indep:
                # transition distribution
                distr[d][i] = transition_matrix(part_y, nstates, dt=dt)
            else:
                # distribution independent in time
                marg_distr = states2distr(part_y, nchannels)
                distr[d][i] = outer(marg_distr, marg_distr)*(part_y.shape[0]-1)
    return distr
