#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from pyswarms
from pyswarms.double.constrained_optimizer import ConstrainedOptimizerPSO
from pyswarms.utils.functions.single_obj import himmelblau, sphere, zeroTest, rosenbrock, n_disk_constraint

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
    
    @pytest.fixture
    def sum_under(self):
        """Objective function with arguments"""

        def sum_under_(x, s):
            return np.abs(np.sum(x, axis=1))-s

        return sum_under_
    
    def test_rosenbrock(self, options):
        """ Test to check constrained optimizer returns correct values when given dummy constraints"""
        opt = ConstrainedOptimizerPSO(10, 2, options=options)
        cost, pos = opt.optimize(rosenbrock, zeroTest, 10000)
        expected_pos = np.array([1., 1.])
        expected_cost = 0.
        assert pos == pytest.approx(expected_pos)
        assert cost == pytest.approx(expected_cost)

    def test_rosenbrock_disk(self, options):
        """ Test to check constrained optimizer returns correct values when given constraints"""
        opt = ConstrainedOptimizerPSO(10, 2, options=options)
        cost, pos = opt.optimize(rosenbrock, n_disk_constraint, 1000, r=2.)
        expected_pos = np.array([1., 1.])
        expected_cost = 0.
        assert pos == pytest.approx(expected_pos)
        assert cost == pytest.approx(expected_cost)

    def test_himmelblau(self, options):
        """ Test to check constrained optimizer returns correct values when given constraints"""
        x_max = 5 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = ConstrainedOptimizerPSO(10, 2, bounds=bounds, options=options)
        cost, pos = opt.optimize(himmelblau, zeroTest, 5000)
        expected_pos = np.array([1., 1.])
        expected_cost = 0.
        #assert pos == pytest.approx(expected_pos)
        assert cost == pytest.approx(expected_cost)

    def test_himmelblau_disk(self, options):
        """ Test to check constrained optimizer returns correct values when given constraints"""
        x_max = 5 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = ConstrainedOptimizerPSO(10, 2, bounds=bounds, options=options)
        cost, pos = opt.optimize(himmelblau, n_disk_constraint, 30000, r=3.)
        expected_pos = np.array([3., 2.])
        expected_cost = 0.
        assert cost == pytest.approx(expected_cost)
        assert pos == pytest.approx(expected_pos)

    def test_himmelblau_constrained(self, options, sum_under):
        """ Test to check constrained optimizer returns correct values when given constraints"""
        x_max = 5 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = ConstrainedOptimizerPSO(10, 2, bounds=bounds, options=options)
        cost, pos = opt.optimize(himmelblau, sum_under, 30000, s=2.)
        expected_pos_1 = np.array([-2.805118, 3.131312])
        expected_pos_2 = np.array([3.584428, -1.848126])
        expected_cost = 0.
        assert pos == pytest.approx(expected_pos_1) or pos == pytest.approx(expected_pos_2)
        assert cost == pytest.approx(expected_cost)

    def test_global_correct_pos(self, options):
        """ Test to check global optimiser returns the correct position corresponding to the best cost """
        opt = ConstrainedOptimizerPSO(n_particles=10, dimensions=2, options=options)
        cost, pos = opt.optimize(sphere, zeroTest, iters=5)
        # find best pos from history
        min_cost_idx = np.argmin(opt.cost_history)
        min_pos_idx = np.argmin(sphere(opt.pos_history[min_cost_idx]))
        assert np.array_equal(opt.pos_history[min_cost_idx][min_pos_idx], pos)
