# Author: Pietro Berkes < pietro _DOT_ berkes _AT_ googlemail _DOT_ com >
# Copyright (c) 2011 Pietro Berkes
# License: GPL v3

import unittest
import kl_tools
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

class KLTest(unittest.TestCase):
    def test_spikes2states(self):
        patterns = np.array([[1, 0, 1, 0],
                             [1, 1, 1, 1],
                             [0, 0, 0, 0]])
        desired = np.array([10, 15, 0])
        assert_array_equal(desired,
                           kl_tools.spikes2states(patterns))
    
    def test_states2distr(self):
        nchannels = 4
        patterns = np.array([[1, 0, 1, 0],
                             [1, 0, 1, 0],
                             [1, 0, 1, 0],
                             [1, 1, 1, 1],
                             [0, 0, 0, 0]])
        states = kl_tools.spikes2states(patterns)

        # histogram over all possible states
        desired = np.zeros(16)
        desired[10] = 3
        desired[-1] = 1
        desired[0] = 1

        distr = kl_tools.states2distr(states, nchannels, normed=False)
        self.assertEqual(len(distr), 2**nchannels)
        assert_array_equal(desired, distr)

        distr = kl_tools.states2distr(states, nchannels, normed=True)
        self.assertEqual(len(distr), 2**nchannels)
        assert_array_equal(desired/5., distr)

    def test_states2dict(self):
        states = np.array([0]*10 + [1]*10 + [2]*10 + [3]*10)
        
        distr = kl_tools.states2dict(states, 2, states.shape[0], shuffle=False)
        assert_array_equal(distr[1][0], [10.,10.,10.,10.])
        assert_array_equal(distr[2][0], [10.,10.,0.,0.])
        assert_array_equal(distr[2][1], [0.,0.,10.,10.])
        for i in range(4):
            desired = np.zeros((4,))
            desired[i] = 10.
            assert_array_equal(distr[4][i], desired)
            
        # check that shuffling shuffles
        distr = kl_tools.states2dict(states, 2, states.shape[0], shuffle=True)
        assert_array_equal(distr[1][0], [10.,10.,10.,10.])
        self.assertFalse(np.all(distr[2][0] == [10.,10.,0.,0.]))
        self.assertEqual(distr[2][0].sum(), 20.)
        self.assertFalse(np.all(distr[2][1] == [0.,0.,10.,10.]))
        self.assertEqual(distr[2][1].sum(), 20.)
        for i in range(4):
            desired = np.zeros((4,))
            desired[i] = 10.
            self.assertFalse(np.all(distr[4][i] == desired))
            self.assertEqual(distr[4][i].sum(), 10.)

    def test_spikes2indep_dict(self):
        spikes = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [1, 0]])
        desired = np.array([8.,2.,12.,3.])/25.
        distr = kl_tools.spikes2indep_dict(spikes)
        assert_array_almost_equal(desired, distr[1][0]/5., 6)
        
    def test_transition_matrix(self):
        patterns = np.array([[0, 0],
                             [0, 1],
                             [0, 1],
                             [1, 1],
                             [0, 0],
                             [0, 1]])
        states = kl_tools.spikes2states(patterns)

        desired = np.zeros((4, 4))
        desired[0,1] = 2
        desired[1,1] = 1
        desired[1,3] = 1
        desired[3,0] = 1

        tr = kl_tools.transition_matrix(states, 4, dt=1)
        self.assertEqual(tr.shape, (4,4))
        assert_array_equal(desired, tr)

        desired = np.zeros((4, 4))
        desired[0,1] = 1
        desired[1,3] = 1
        desired[1,0] = 1
        desired[3,1] = 1

        tr = kl_tools.transition_matrix(states, 4, dt=2)
        assert_array_equal(desired, tr)

    def test_entropy(self):
        nchannels = 4
        # uniform distribution, entropy = number of channels
        alpha = np.zeros(2**nchannels) + 10000
        hest = -kl_tools.mean_H_estimate(alpha)
        assert(abs(hest-nchannels) < 1e-4)

        # sample from distribution, reconstruct entropy
        nchannels = 3
        distr = np.array([0.1, 0.3, 0.05, 0.05, 0.2, 0.1, 0.1, 0.1])
        real_entropy = -sum(distr*np.log2(distr))
        
        h = - kl_tools.entropy(distr)
        assert_almost_equal(real_entropy, h, 2)
        
        hest1 = - kl_tools.mean_H_estimate(distr*100000)
        assert_almost_equal(real_entropy, hest1, 2)

        # sample from distribution
        states = np.random.multinomial(1, distr, size=100000).argmax(1)
        distr = kl_tools.states2dict(states[:,None], nchannels)
        
        hest1 = - kl_tools.mean_H_estimate(distr[1][0])
        assert_almost_equal(real_entropy, hest1, 2)

        hest2 = - kl_tools.h_estimation(distr, states.shape[0]);
        assert_almost_equal(real_entropy, hest2, 2)

    def test_kl_zero(self):
        nchannels = 3
        distr = np.array([0.1, 0.3, 0.05, 0.05, 0.2, 0.1, 0.1, 0.1])
        # same states, KL divergence should converge to zero
        states = np.random.multinomial(1, distr, size=100000).argmax(1)
        states2 = np.random.permutation(states)

        distr = kl_tools.states2distr(states, nchannels)
        distr2 = kl_tools.states2distr(states2, nchannels)
        kl1 = kl_tools.mean_KL_estimate(distr, distr2)
        assert_almost_equal(kl1, 0., 3)

        distr = kl_tools.states2dict(states[:,None], nchannels)
        distr2 = kl_tools.states2dict(states2[:,None], nchannels)
        kl2, _ = kl_tools.kl_estimation(distr, distr2, 100000)
        assert_almost_equal(kl2, 0., 3)

    def test_kl(self):
        nchannels = 3
        distr = np.array([0.1, 0.3, 0.05, 0.05, 0.2, 0.1, 0.1, 0.1])
        distr2 = np.array([0.4, 0.2, 0.01, 0.09, 0.05, 0.05, 0.03, 0.17])
        real_kl = (distr * np.log2(distr/distr2)).sum()

        kl1 = kl_tools.kl(distr, distr2)
        assert_almost_equal(kl1, real_kl, 6)

        kl2 = kl_tools.mean_KL_estimate(distr*100000, distr2*100000)
        assert_almost_equal(kl2, real_kl, 3)

        # sample states
        states = np.random.multinomial(1, distr, size=100000).argmax(1)
        states2 = np.random.multinomial(1, distr2, size=100000).argmax(1)

        distr = kl_tools.states2dict(states[:,None], nchannels)
        distr2 = kl_tools.states2dict(states2[:,None], nchannels)
        kl3, _ = kl_tools.kl_estimation(distr, distr2, 100000)
        assert_almost_equal(kl3, real_kl, 2)


if __name__ == '__main__':
    unittest.main()

