# -*- coding: utf-8 -*-

"""
Swarm Operation Backend

This module abstracts various operations in the swarm such as updating
the personal best, finding neighbors, etc. You can use these methods
to specify how the swarm will behave.
"""

# Import standard library
import logging

# Import modules
import numpy as np
from pyswarms.backend.swarms import ConstrainedSwarm

from ..utils.reporter import Reporter
from .handlers import BoundaryHandler, VelocityHandler
from functools import partial


rep = Reporter(logger=logging.getLogger(__name__))


def compute_pbest(swarm):
    """Update the personal best score of a swarm instance

    You can use this method to update your personal best positions.

    .. code-block:: python

        import pyswarms.backend as P
        from pyswarms.backend.swarms import Swarm

        my_swarm = P.create_swarm(n_particles, dimensions)

        # Inside the for-loop...
        for i in range(iters):
            # It updates the swarm internally
            my_swarm.pbest_pos, my_swarm.pbest_cost = P.update_pbest(my_swarm)

    It updates your :code:`current_pbest` with the personal bests acquired by
    comparing the (1) cost of the current positions and the (2) personal
    bests your swarm has attained.

    If the cost of the current position is less than the cost of the personal
    best, then the current position replaces the previous personal best
    position.

    Parameters
    ----------
    swarm : pyswarms.backend.swarm.Swarm
        a Swarm instance

    Returns
    -------
    numpy.ndarray
        New personal best positions of shape :code:`(n_particles, n_dimensions)`
    numpy.ndarray
        New personal best costs of shape :code:`(n_particles,)`
    """
    try:
        # Infer dimensions from positions
        dimensions = swarm.dimensions
        
        # Create a 1-D and 2-D mask based from comparisons
        mask_cost = swarm.current_cost < swarm.pbest_cost
        mask_pos = np.repeat(mask_cost[:, np.newaxis], dimensions, axis=1)
        # Apply masks
        new_pbest_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
        new_pbest_cost = np.where(
            ~mask_cost, swarm.pbest_cost, swarm.current_cost
        )
    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    else:
        return (new_pbest_pos, new_pbest_cost)


def compute_velocity(swarm, clamp, vh, bounds=None):
    """Update the velocity matrix

    This method updates the velocity matrix using the best and current
    positions of the swarm. The velocity matrix is computed using the
    cognitive and social terms of the swarm. The velocity is handled
    by a :code:`VelocityHandler`.

    A sample usage can be seen with the following:

    .. code-block :: python

        import pyswarms.backend as P
        from pyswarms.swarms.backend import Swarm, VelocityHandler

        my_swarm = P.create_swarm(n_particles, dimensions)
        my_vh = VelocityHandler(strategy="invert")

        for i in range(iters):
            # Inside the for-loop
            my_swarm.velocity = compute_velocity(my_swarm, clamp, my_vh, bounds)

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    clamp : tuple of floats, optional
        a tuple of size 2 where the first entry is the minimum velocity
        and the second entry is the maximum velocity. It
        sets the limits for velocity clamping.
    vh : pyswarms.backend.handlers.VelocityHandler
        a VelocityHandler object with a specified handling strategy.
        For further information see :mod:`pyswarms.backend.handlers`.
    bounds : tuple of numpy.ndarray or list, optional
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.

    Returns
    -------
    numpy.ndarray
        Updated velocity matrix
    """
    try:
        # Prepare parameters
        swarm_size = swarm.position.shape
        c1 = swarm.options["c1"]
        c2 = swarm.options["c2"]
        w = swarm.options["w"]
        # Compute for cognitive and social terms, constraints handled differently
        if swarm is ConstrainedSwarm:
            
            cognitive = (
            c1
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.pbest_merged_pos - swarm.position)
            )
            social = (
                c2
                * np.random.uniform(0, 1, swarm_size)
                * (swarm.best_merged_pos - swarm.position)
            )
        else:
            cognitive = (
                c1
                * np.random.uniform(0, 1, swarm_size)
                * (swarm.pbest_pos - swarm.position)
            )
            social = (
                c2
                * np.random.uniform(0, 1, swarm_size)
                * (swarm.best_pos - swarm.position)
            )
        # Compute temp velocity (subject to clamping if possible)
        temp_velocity = (w * swarm.velocity) + cognitive + social
        updated_velocity = vh(
            temp_velocity, clamp, position=swarm.position, bounds=bounds
        )

    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    except KeyError:
        rep.logger.exception("Missing keyword in swarm.options")
        raise
    else:
        return updated_velocity


def compute_position(swarm, bounds, bh):
    """Update the position matrix

    This method updates the position matrix given the current position and the
    velocity. If bounded, the positions are handled by a
    :code:`BoundaryHandler` instance

    .. code-block :: python

        import pyswarms.backend as P
        from pyswarms.swarms.backend import Swarm, VelocityHandler

        my_swarm = P.create_swarm(n_particles, dimensions)
        my_bh = BoundaryHandler(strategy="intermediate")

        for i in range(iters):
            # Inside the for-loop
            my_swarm.position = compute_position(my_swarm, bounds, my_bh)

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    bounds : tuple of numpy.ndarray or list, optional
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.
    bh : pyswarms.backend.handlers.BoundaryHandler
        a BoundaryHandler object with a specified handling strategy
        For further information see :mod:`pyswarms.backend.handlers`.

    Returns
    -------
    numpy.ndarray
        New position-matrix
    """
    try:
        temp_position = swarm.position.copy()
        temp_position += swarm.velocity

        if bounds is not None:
            temp_position = bh(temp_position, bounds)

        position = temp_position
    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    else:
        return position


def compute_objective_function(swarm, objective_func, pool=None, **kwargs):
    """Evaluate particles using the objective function

    This method evaluates each particle in the swarm according to the objective
    function passed.

    If a pool is passed, then the evaluation of the particles is done in
    parallel using multiple processes.

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    objective_func : function
        objective function to be evaluated
    pool: multiprocessing.Pool
        multiprocessing.Pool to be used for parallel particle evaluation
    kwargs : dict
        arguments for the objective function

    Returns
    -------
    numpy.ndarray
        Cost-matrix for the given swarm
    """
    if pool is None:
        return objective_func(swarm.position, **kwargs)
    else:
        results = pool.map(
            partial(objective_func, **kwargs),
            np.array_split(swarm.position, pool._processes),
        )
        return np.concatenate(results)
    
#TODO another option for constraints, call it something else for now but could later replace the normal one
def compute_constrained_cost(swarm, objective_func, constraint_handler, pool=None, **kwargs):
    """Evaluate particles using the objective function

    This method evaluates each particle in the swarm according to the objective
    function passed.

    If a pool is passed, then the evaluation of the particles is done in
    parallel using multiple processes.

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    objective_func : function
        objective function to be evaluated
    pool: multiprocessing.Pool
        multiprocessing.Pool to be used for parallel particle evaluation
    kwargs : dict
        arguments for the objective function

    Returns
    -------
    numpy.ndarray
        Cost-matrix for the given swarm
    """
    try:
        temp_cost = swarm.current_cost.copy()

        if constraint_handler is not None:
            temp_cost = constraint_handler()

        amended_cost = temp_cost
    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    else:
        return amended_cost

    if pool is None:
        return objective_func(swarm.position, **kwargs)
    else:
        results = pool.map(
            partial(objective_func, **kwargs),
            np.array_split(swarm.position, pool._processes),
        )
        return np.concatenate(results)


def compute_objective_functions(swarm, objective_funcs, pool=None, **kwargs):
    """Evaluate particles using multiple objective functions
    
    Computes multiple objective functions by calling compute_objective_function
    on each objective_func in objective_funcs

    Note that arguments for all objective functions come from the same kwargs
    dictionary.

    If a pool is passed, then the evaluation of the particles is done in
    parallel using multiple processes by passing the pool argument to
    compute_objective_func.

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    objective_funcs : array of functions
        objective functions to be evaluated
    pool: multiprocessing.Pool
        multiprocessing.Pool to be used for parallel particle evaluation
    kwargs : dict
        arguments for the objective functions

    Returns
    -------
    costs : dict
        Cost-matrix dictionary for the given swarm
            key : objective_func
            value : cost-matrix for the given swarm
    """

    costs = {}
    for objective_func in objective_funcs:
        costs[objective_func] = compute_objective_function(swarm, objective_func, pool, kwargs)
    return costs

def compute_constraint_function(swarm, constraint_func, pool=None, **kwargs):
    """Evaluate particles using the constraint function

    This method evaluates each particle in the swarm according to the constraint
    function passed.

    If a pool is passed, then the evaluation of the particles is done in
    parallel using multiple processes.

    Note: This is fundamentally the same as compute_objective_function, but it does help
    delineate the two as this function can be called within the ConstraintHandler

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    objective_func : function
        objective function to be evaluated
    pool: multiprocessing.Pool
        multiprocessing.Pool to be used for parallel particle evaluation
    kwargs : dict
        arguments for the objective function

    Returns
    -------
    numpy.ndarray
        Cost-matrix for the given swarm
    """
    if pool is None:
        return constraint_func(swarm.position, **kwargs)
    else:
        results = pool.map(
            partial(constraint_func, **kwargs),
            np.array_split(swarm.position, pool._processes),
        )
        return np.concatenate(results)
    
def compute_pbest_constraints(swarm):
    """Update the personal best score of a swarm instance

    You can use this method to update your personal best positions.

    .. code-block:: python

        import pyswarms.backend as P
        from pyswarms.backend.swarms import Swarm

        my_swarm = P.create_swarm(n_particles, dimensions)

        # Inside the for-loop...
        for i in range(iters):
            # It updates the swarm internally
            my_swarm.pbest_pos, my_swarm.pbest_cost = P.update_pbest(my_swarm)

    It updates your :code:`current_pbest` with the personal bests acquired by
    comparing the (1) cost of the current positions and the (2) personal
    bests your swarm has attained.

    If the cost of the current position is less than the cost of the personal
    best, then the current position replaces the previous personal best
    position.

    Parameters
    ----------
    swarm : pyswarms.backend.swarm.Swarm
        a Swarm instance

    Returns
    -------
    numpy.ndarray
        New personal best positions of shape :code:`(n_particles, n_dimensions)`
    numpy.ndarray
        New personal best costs of shape :code:`(n_particles,)`
    """
    try:
        # Infer dimensions from positions
        dimensions = swarm.dimensions
        
        # Create a 1-D and 2-D mask based from comparisons
        mask_violation = swarm.current_violation < swarm.pbest_violation
        mask_pos = np.repeat(mask_violation[:, np.newaxis], dimensions, axis=1)
        # Apply masks
        new_pbest_pos = np.where(~mask_pos, swarm.pbest_violation_pos, swarm.position)
        new_pbest_violation = np.where(
            ~mask_violation, swarm.pbest_violation, swarm.current_violation
        )
    except AttributeError:
        rep.logger.exception(
            "Please pass a ConstrainedSwarm class. You passed {}".format(type(swarm))
        )
        raise
    else:
        return (new_pbest_pos, new_pbest_violation)
