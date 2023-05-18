# -*- coding: utf-8 -*-

r"""
An epsilon-constrained Particle Swarm Optimization (general PSO) algorithm.

It takes a set of candidate solutions, and tries to find the best
solution under constraints using a position-velocity update method 
where particles which satisfy constraints are moved to minimize the
objective function and particles violating constraints are moved to
minimize the constraint function. Uses a user specified topology.

The position update can be defined as:

.. math::

   x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

Where the position at the current timestep :math:`t` is updated using
the computed velocity at :math:`t+1`. Furthermore, the velocity update
is defined as:

.. math::

   v_{ij}(t + 1) = m * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                   + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

Here, :math:`c1` and :math:`c2` are the cognitive and social parameters
respectively. They control the particle's behavior given two choices: (1) to
follow its *personal best* or (2) follow the swarm's *global best* position.
Overall, this dictates if the swarm is explorative or exploitative in nature.
In addition, a parameter :math:`w` controls the inertia of the swarm's
movement.

An example usage is as follows:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.backend.topology import Pyramid
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters and topology
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    my_topology = Pyramid(static=False)

    # Call instance of GlobalBestPSO
    optimizer = ps.single.GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                        options=options, topology=my_topology)

    # Perform optimization
    stats = optimizer.optimize(fx.sphere, iters=100)

This algorithm was adapted from the earlier works of J. Kennedy and
R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.
"""

# Import standard library
import logging

# Import modules
import numpy as np
import multiprocessing as mp

from collections import deque, namedtuple
from pyswarms.backend.generators import create_epsilon_swarm

from pyswarms.double.general_optimizer import GeneralOptimizerPSO

from ..backend.operators import compute_pbest, compute_objective_function, compute_epsilon_functions, compute_constraint_function, compute_pbest_constraints
from ..backend.topology import Topology
from ..backend.handlers import BoundaryHandler, ConstraintHandler, VelocityHandler, OptionsHandler
from ..base import SwarmOptimizer
from ..utils.reporter import Reporter


class ConstrainedOptimizerPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        topology,
        bounds=None,
        oh_strategy=None,
        bh_strategy="periodic",
        velocity_clamp=None,
        vh_strategy="unmodified",
        ch_strategy=None,
        center=1.00,
        ftol=-np.inf,
        ftol_iter=1,
        init_pos=None,
    ):
        """Initialize the constrained swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}` or :code:`{'c1',
                'c2', 'w', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                if used with the :code:`Ring`, :code:`VonNeumann` or
                :code:`Random` topology the additional parameter k must be
                included
                * k : int
                    number of neighbors to be considered. Must be a positive
                    integer less than :code:`n_particles`
                if used with the :code:`Ring` topology the additional
                parameters k and p must be included
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the sum-of-absolute
                    values (or L1 distance) while 2 is the Euclidean (or L2)
                    distance.
                if used with the :code:`VonNeumann` topology the additional
                parameters p and r must be included
                * r: int
                    the range of the VonNeumann topology.  This is used to
                    determine the number of neighbours in the topology.
        topology : pyswarms.backend.topology.Topology
            a :code:`Topology` object that defines the topology to use in the
            optimization process. The currently available topologies are:
                * Star
                    All particles are connected
                * Ring (static and dynamic)
                    Particles are connected to the k nearest neighbours
                * VonNeumann
                    Particles are connected in a VonNeumann topology
                * Pyramid (static and dynamic)
                    Particles are connected in N-dimensional simplices
                * Random (static and dynamic)
                    Particles are connected to k random particles
                Static variants of the topologies remain with the same
                neighbours over the course of the optimization. Dynamic
                variants calculate new neighbours every time step.
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        oh_strategy : dict, optional, default=None(constant options)
            a dict of update strategies for each option.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity and
            the second entry is the maximum velocity. It sets the limits for
            velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : list (default is :code:`None`)
            an array of size :code:`dimensions`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        """
        super(GeneralOptimizerPSO, self).__init__(
            n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            ftol_iter=ftol_iter,
            init_pos=init_pos,
        )
        self.ToHistory = namedtuple(
            "ToHistory",
            [
                "best_cost",
                "mean_pbest_cost",
                "mean_neighbor_cost",
                "best_violation",
                "mean_pbest_violation",
                "mean_neighbor_violation",
                "position",
                "velocity",
            ],
        )
        if oh_strategy is None:
            oh_strategy = {}
        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology and check for type
        if not isinstance(topology, Topology):
            raise TypeError("Parameter `topology` must be a Topology object")
        else:
            self.top = topology
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.oh = OptionsHandler(strategy=oh_strategy)
        self.ch = ConstraintHandler(strategy=ch_strategy)
        self.name = __name__

    def _populate_history(self, hist):
        """Populate all history lists

        The :code:`cost_history`, :code:`mean_pbest_history`, and
        :code:`neighborhood_best` is expected to have a shape of
        :code:`(iters,)`,on the other hand, the :code:`pos_history`
        and :code:`velocity_history` are expected to have a shape of
        :code:`(iters, n_particles, dimensions)`

        Parameters
        ----------
        hist : collections.namedtuple
            Must be of the same type as self.ToHistory
        """
        self.cost_history.append(hist.best_cost)
        self.violation_history.append(hist.best_violation)
        self.mean_pbest_history.append(hist.mean_pbest_cost)
        self.mean_pbest_violation_history.append(hist.mean_pbest_violation)
        self.mean_neighbor_history.append(hist.mean_neighbor_cost)
        self.mean_neighbor_violation_history.append(hist.mean_neighbor_violation)
        self.pos_history.append(hist.position)
        self.velocity_history.append(hist.velocity)

#TODO: continue from here
    def optimize(
        self, objective_func, constraint_func, iters, n_processes=None, verbose=True, **kwargs
    ):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` and constraint violation :code: `g`
        for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        constraint_func : callable
            constraint function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """
        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )

        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position
        self.ch.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            # fmt: off
            self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=pool, **kwargs)
            self.swarm.current_violation = compute_constraint_function(self.swarm, constraint_func, pool, **kwargs)
            mask_epsilon = self.swarm.current_violation <= np.zeros(self.swarm.n_particles)
            self.swarm.costs_merged = np.where(mask_epsilon, self.swarm.current_cost, self.swarm.current_violation)
            #TODO: use a mask to put cost history as correct cost from objective or constraint
            # Compute personal best cost and position for each particle
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            # Compute personal best violation and position for each particle
            self.swarm.pbest_violation_pos, self.swarm.pbest_violation = compute_pbest_constraints(self.swarm)
            self.swarm.pbest_merged = np.where(mask_epsilon, self.swarm.pbest_cost, self.swarm.pbest_violation)
            self.swarm.pbest_merged_pos = np.where(mask_epsilon, self.swarm.pbest_pos, self.swarm.pbest_violation_pos)

            best_cost_yet_found = self.swarm.best_cost
            best_violation_yet_found = self.swarm.best_violation
            # fmt: on
            # Update swarm
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, **self.options
            )
            self.swarm.best_violation_pos, self.swarm.best_violation = self.top.compute_gbest_violation(
                self.swarm, **self.options
            )
            # Update the merged arrays of best values where the singular best value for cost or violation
            # is placed at each index based on whether the constraint function is satisfactorily minimized
            # on a per-particle basis
            self.swarm.best_merged_pos = np.where(mask_epsilon, self.swarm.best_pos, self.swarm.best_violation_pos)
            self.swarm.best_merged = np.where(mask_epsilon, self.swarm.best_cost, self.swarm.best_violation)

            # Print to console
            if verbose:
                self.rep.hook(best_cost=self.swarm.best_cost)
            # TODO might need to add a violation_hist, already have variable for violation_history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            )
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform options update
            self.swarm.options = self.oh(
                self.options, iternow=i, itermax=iters
            )
            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()
        return (final_best_cost, final_best_pos)
    
    def reset(self):
        """Reset the attributes of the optimizer

        All variables/atributes that will be re-initialized when this
        method is defined here. Note that this method
        can be called twice: (1) during initialization, and (2) when
        this is called from an instance.

        It is good practice to keep the number of resettable
        attributes at a minimum. This is to prevent spamming the same
        object instance with various swarm definitions.

        Normally, swarm definitions are as atomic as possible, where
        each type of swarm is contained in its own instance. Thus, the
        following attributes are the only ones recommended to be
        resettable:

        * Swarm position matrix (self.pos)
        * Velocity matrix (self.pos)
        * Best scores and positions (gbest_cost, gbest_pos, etc.)

        Otherwise, consider using positional arguments.
        """
        # Initialize history lists
        self.cost_history = []
        self.violation_history = [] #added
        self.mean_pbest_history = []
        self.mean_pbest_violation_history = []
        self.mean_neighbor_history = []
        self.mean_neighbor_violation_history = []
        self.pos_history = []
        self.velocity_history = []

        # Initialize the swarm
        self.swarm = create_epsilon_swarm(
            n_particles=self.n_particles,
            dimensions=self.dimensions,
            bounds=self.bounds,
            center=self.center,
            init_pos=self.init_pos,
            clamp=self.velocity_clamp,
            options=self.options,
        )
