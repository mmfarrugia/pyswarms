#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard libraries
import re
import pytest
import random

random.seed(0)

# Import modules
import numpy as np

# Import from pyswarms
from pyswarms.backend.topology import Star
from pyswarms.utils.functions import single_obj as fx
from pyswarms.single import GlobalBestPSO, LocalBestPSO, GeneralOptimizerPSO
from pyswarms.constrained import ConstrainedOptimizerPSO

# Instantiate optimizers
optimizers = [GlobalBestPSO, LocalBestPSO, GeneralOptimizerPSO]
constrained_optimizers = [ConstrainedOptimizerPSO]
options = {"c1": 2, "c2": 2, "w": 0.7, "k": 3, "p": 2}
parameters = dict(
    n_particles=20,
    dimensions=10,
    options=options,
)


class TestToleranceOptions:
    @pytest.fixture(params=optimizers)
    def optimizer(self, request):
        global parameters
        if request.param.__name__ == "GeneralOptimizerPSO":
            return request.param, {**parameters, **{"topology": Star()}}
        return request.param, parameters

    @pytest.fixture(params=constrained_optimizers)
    def constrained_optimizer(self, request):
        global parameters
        if request.param.__name__ == "ConstrainedOptimizerPSO":
            return request.param, {**parameters, **{"topology": Star()}}
        return request.param, parameters

    def test_verbose(self, optimizer, capsys):
        """Test verbose run"""
        optm, params = optimizer
        opt = optm(**params)
        opt.optimize(fx.sphere, iters=100)
        out = capsys.readouterr().err
        count = len(re.findall(r"pyswarms", out))
        assert count > 0

    def test_verbose_(self, constrained_optimizer, capsys):
        """Test verbose run"""
        optm, params = constrained_optimizer
        opt = optm(**params)
        opt.optimize(fx.sphere, fx.zeroTest, iters=100)
        out = capsys.readouterr().err
        count = len(re.findall(r"pyswarms", out))
        assert count > 0

    def test_silent(self, optimizer, capsys):
        """Test silent run"""
        optm, params = optimizer
        opt = optm(**params)
        opt.optimize(fx.sphere, iters=100, verbose=False)
        out = capsys.readouterr()
        assert out.err == ""
        assert out.out == ""

    def test_silent_(self, constrained_optimizer, capsys):
        """Test silent run"""
        optm, params = constrained_optimizer
        opt = optm(**params)
        opt.optimize(fx.sphere, fx.zeroTest, iters=100, verbose=False)
        out = capsys.readouterr()
        assert out.err == ""
        assert out.out == ""
