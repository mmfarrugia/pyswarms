"""
The :mod:`pyswarms.single` module implements various techniques in
continuous single-objective optimization. These require only one
objective function that can be optimized in a continuous space.

.. note::
    PSO algorithms scale with the search space. This means that, by
    using larger boundaries, the final results are getting larger
    as well.

.. note::
    Please keep in mind that Python has a biggest float number.
    So using large boundaries in combination with exponentiation or
    multiplication can lead to an :code:`OverflowError`.
"""

from .constrained_optimizer import ConstrainedOptimizerPSO

__all__ = ["ConstrainedOptimizerPSO"]
