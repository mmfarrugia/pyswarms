#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from pyswarms
from pyswarms.double.constrained_optimizer import ConstrainedOptimizerPSO
from pyswarms.utils.functions.single_obj import sphere, zeroTest

from .abc_test_constrained_optimizer import ABCTestConstrainedOptimizer


class TestConstrainedOptimizer(ABCTestConstrainedOptimizer):
    @pytest.fixture
    def optimizer(self):
        return ConstrainedOptimizerPSO

    @pytest.fixture
    def optimizer_history(self, options):
        opt = ConstrainedOptimizerPSO(10, 2, options=options)
        opt.optimize(sphere, zeroTest, 1000)
        return opt

    @pytest.fixture
    def optimizer_reset(self, options):
        opt = ConstrainedOptimizerPSO(10, 2, options=options)
        opt.optimize(sphere, zeroTest, 10)
        opt.reset()
        return opt

    def test_global_correct_pos(self, options):
        """ Test to check global optimiser returns the correct position corresponding to the best cost """
        opt = ConstrainedOptimizerPSO(n_particles=10, dimensions=2, options=options)
        cost, pos = opt.optimize(sphere, zeroTest, iters=5)
        # find best pos from history
        min_cost_idx = np.argmin(opt.cost_history)
        min_pos_idx = np.argmin(sphere(opt.pos_history[min_cost_idx]))
        assert np.array_equal(opt.pos_history[min_cost_idx][min_pos_idx], pos)
