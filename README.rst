neuro-kl
======

`neuro-kl` contains a Python module and Matlab functions to compute an estimate
of the entropy and Kullback-Leibler divergence of distribution of simultaneously
recorded neural data.

The KL is estimated using a Bayesian method designed to deal with
relatively large distributions (2^16 elements), and is described in the
supplementary material of the paper

    Berkes, P., Orban, G., Lengyel, M., and Fiser, J. (2011).
    *Spontaneous cortical activity reveals hallmarks of an optimal
    internal model of the environment.*
    Science, 331:6013, 83–87.

in the "Dissimilarity of neural activity distributions" section. The supplementary material is freely available at
http://www.sciencemag.org/content/331/6013/83/suppl/DC1 .

License
=====

neuro-kl is released under the GPL v3. See LICENSE.txt .

Copyright (c) 2011, Pietro Berkes, Dmitriy Lisitsyn. All rights reserved.
