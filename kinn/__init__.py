#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# + {}
# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__author__ = {'Gabriel S. Gusmao' : 'gusmaogabriels@gmail.com'}
__version__ = '1.0'


def solve(problem):
    """Solve a validated JSON file or problem dictionary locally.

    Both training formulations belong to this package. Select ``method='fixed'``
    for the original fixed-weight objective, or ``method='mle'`` for the
    MLE covariance weighting and SVD extension (the default).
    """
    from .problem import load, validate
    normalized = validate(problem) if isinstance(problem, dict) else load(problem)
    if normalized['method'] == 'fixed':
        from .pareto import run
    else:
        from .rkinn import run
    return run(normalized)
