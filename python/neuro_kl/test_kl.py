# Author: Pietro Berkes < pietro _DOT_ berkes _AT_ googlemail _DOT_ com >
# Copyright (c) 2011 Pietro Berkes
# License: GPL v3

import unittest
import kl_tools
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

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

    def kl_basic(self):
        p = np.array([0.7, 0.1, 0.2])
        self.AssertEqual(kl.kl(p, p), 0.)

if __name__ == '__main__':
    unittest.main()

