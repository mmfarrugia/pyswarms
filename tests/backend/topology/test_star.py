#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import pytest

# Import from pyswarms
from pyswarms.backend.topology import Star

from .abc_test_topology import ABCTestTopology

np.random.seed(4135157)


class TestStarTopology(ABCTestTopology):
    @pytest.fixture
    def topology(self):
        return Star

    @pytest.fixture
    def options(self):
        return {}

    def test_compute_gbest_return_values(self, swarm, options, topology):
        """Test if compute_gbest() gives the expected return values"""
        topo = topology()
        expected_cost = 1.0002528364353296
        expected_pos = np.array(
            [9.90438476e-01, 2.50379538e-03, 1.87405987e-05]
        )
        pos, cost = topo.compute_gbest(swarm, **options)
        assert cost == pytest.approx(expected_cost)
        assert pos == pytest.approx(expected_pos)

    def test_compute_gbest_return_values_(
        self, constrained_swarm, options, topology
    ):
        """Test if compute_gbest() gives the expected return values"""
        topo = topology()
        expected_cost = 1.0002528364353296
        expected_pos = np.array(
            [9.90438476e-01, 2.50379538e-03, 1.87405987e-05]
        )
        pos, cost = topo.compute_gbest(constrained_swarm, **options)
        assert cost == pytest.approx(expected_cost)
        assert pos == pytest.approx(expected_pos)

    def test_compute_gbest_violation_return_values(
        self, constrained_swarm, options, topology
    ):
        """Test if compute_gbest() gives the expected return values"""
        topo = topology()
        expected_cost = -6.0
        expected_pos = np.array(
            [9.93740665e-01, -6.16501403e-03, -1.46096578e-02]
        )
        pos, cost = topo.compute_gbest_violation(constrained_swarm, **options)
        assert cost == pytest.approx(expected_cost)
        assert pos == pytest.approx(expected_pos)
