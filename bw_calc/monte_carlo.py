from .lca import LCA
from .utils import get_seed
from contextlib import contextmanager
from scipy.sparse.linalg import iterative
from stats_arrays.random import MCRandomNumberGenerator
import multiprocessing
import sys

try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve


class MonteCarloLCA(LCA):
    """Monte Carlo uncertainty analysis with separate `random number generators <http://en.wikipedia.org/wiki/Random_number_generation>`_ (RNGs) for each set of parameters."""
    def __init__(self, demand, data_objs, seed=None, *args, **kwargs):
        self.seed = seed or get_seed()
        super().__init__(demand, data_objs, seed=self.seed, *args, **kwargs)
        self.logger.info("Seeded RNGs", extra={'seed': self.seed})

    def load_data(self):
        self.load_lci_data()
        self.tech_rng = MCRandomNumberGenerator(self.tech_params, seed=self.seed)
        self.bio_rng = MCRandomNumberGenerator(self.bio_params, seed=self.seed)
        if self.lcia:
            self.load_lcia_data()
            self.cf_rng = MCRandomNumberGenerator(self.cf_params, seed=self.seed)
        # if self.weighting:
        #     self.load_weighting_data()
        #     self.weighting_rng = MCRandomNumberGenerator(self.weighting_params, seed=self.seed)
        if self.overrides:
            self.overrides.reset_sequential_indices()

    def __iter__(self):
        return self

    def __call__(self):
        return next(self)

    def __next__(self):
        if not hasattr(self, "tech_rng"):
            self.load_data()
        self.rebuild_technosphere_matrix(self.tech_rng.next())
        self.rebuild_biosphere_matrix(self.bio_rng.next())
        if self.lcia:
            self.rebuild_characterization_matrix(self.cf_rng.next())
        # if self.weighting:
        #     self.weighting_value = self.weighting_rng.next()

        if self.overrides:
            self.overrides.update_matrices()

        if not hasattr(self, "demand_array"):
            self.build_demand_array()

        self.lci_calculation()
        if self.lcia:
            self.lcia_calculation()
            # if self.weighting:
            #     self.weighting_calculation()
            return self.score
        else:
            return self.supply_array


class IterativeMonteCarloLCA(MonteCarloLCA):
    """Use iterative techniques instead of `LU factorization <http://en.wikipedia.org/wiki/LU_decomposition>`_ in Monte Carlo."""
    def __init__(self, demand, data_objs, iter_solver=iterative.cgs, *args, **kwargs):
        super().__init__(demand, data_objs, *args, **kwargs)
        self.iter_solver = iter_solver
        self.guess = None

    def solve_linear_system(self):
        if not self.iter_solver or self.guess is None:
            self.guess = spsolve(
                self.technosphere_matrix,
                self.demand_array)
            return self.guess
        else:
            solution, status = self.iter_solver(
                self.technosphere_matrix,
                self.demand_array,
                x0=self.guess,
                maxiter=1000)
            if status != 0:
                return spsolve(
                    self.technosphere_matrix,
                    self.demand_array
                )
            return solution


# class ComparativeMonteCarlo(IterativeMonteCarlo):
#     """First draft approach at comparative LCA"""
#     def __init__(self, demands, *args, **kwargs):
#         self.demands = demands
#         # Get all possibilities for database retrieval
#         demand_all = demands[0].copy()
#         for other in demands[1:]:
#             demand_all.update(other)
#         super(ComparativeMonteCarlo, self).__init__(demand_all, *args, **kwargs)

#     def load_data(self):
#         if not getattr(self, "method"):
#             raise ValueError("Must specify an LCIA method")

#         self.load_lci_data()
#         self.load_lcia_data()
#         self.tech_rng = MCRandomNumberGenerator(self.tech_params, seed=self.seed)
#         self.bio_rng = MCRandomNumberGenerator(self.bio_params, seed=self.seed)
#         self.cf_rng = MCRandomNumberGenerator(self.cf_params, seed=self.seed)

#     def __next__(self):
#         if not hasattr(self, "tech_rng"):
#             self.load_data()
#         self.rebuild_technosphere_matrix(self.tech_rng.next())
#         self.rebuild_biosphere_matrix(self.bio_rng.next())
#         self.rebuild_characterization_matrix(self.cf_rng.next())

#         if self.overrides:
#             self.overrides.update_matrices()

#         results = []
#         for demand in self.demands:
#             self.build_demand_array(demand)
#             self.lci_calculation()
#             self.lcia_calculation()
#             results.append(self.score)
#         return results


def single_worker(args):
    lca_args, iterations = args
    mc = MonteCarloLCA(*lca_args)
    return [next(mc) for x in range(iterations)]


def iterative_solving_worker(args):
    lca_args, iterations = args
    mc = IterativeMonteCarlo(*lca_args)
    return [next(mc) for x in range(iterations)]


class ParallelMonteCarlo(object):
    """Split a Monte Carlo calculation into parallel jobs"""
    def __init__(self, demand, data_objs, iterations=1000, chunk_size=None,
                 cpus=None, log_config=None):
        self.cpus = cpus or multiprocessing.cpu_count()
        if chunk_size:
            self.chunk_size = chunk_size
            self.num_jobs = iterations // chunk_size
            if iterations % self.chunk_size:
                self.num_jobs += 1
        else:
            self.num_jobs = self.cpus
            self.chunk_size = (iterations // self.num_jobs) + 1

    def calculate(self, worker=single_worker):
        with multiprocessing.Pool(processes=self.cpus) as pool:
            results = pool.map(
                worker,
                [
                    ((demand, data_objs), self.chunk_size)
                    for _ in range(self.num_jobs)
                ]
            )
        return [x for lst in results for x in lst]


# def multi_worker(args):
#     """Calculate a single Monte Carlo iteration for many demands.

#     ``args`` are in order:
#         * ``project``: Name of project
#         * ``demands``: List of demand dictionaries
#         * ``method``: LCIA method

#     Returns a list of results: ``[(demand dictionary, result)]``

#     """
#     project, demands, method = args
#     projects.set_current(project, writable=False)
#     mc = MonteCarloLCA(demands[0], method)
#     next(mc)
#     results = []
#     for demand in demands:
#         mc.redo_lcia(demand)
#         results.append((demand, mc.score))
#     return results


# class MultiMonteCarlo(object):
#     """
# This is a class for the efficient calculation of *many* demand vectors from
# each Monte Carlo iteration.

# Args:
#     * ``args`` is a list of demand dictionaries
#     * ``method`` is a LCIA method
#     * ``iterations`` is the number of Monte Carlo iterations desired
#     * ``cpus`` is the (optional) number of CPUs to use

# The input list can have complex demands, so ``[{('foo', 'bar'): 1, ('foo', 'baz'): 1}, {('foo', 'another'): 1}]`` is OK.

# Call ``.calculate()`` to generate results.

#     """
#     def __init__(self, demands, method, iterations, cpus=None):
#         clean_databases()
#         # Convert from activity proxies if necessary
#         self.demands = [{(k[0], k[1]): v for k, v in obj.items()}
#                         for obj in demands]
#         self.method = method
#         self.iterations = iterations
#         self.cpus = cpus or multiprocessing.cpu_count()

#     def merge_results(self, objs):
#         """Merge the results from each ``multi_worker`` worker.

#         ``[('a', [0,1]), ('a', [2,3])]`` becomes ``[('a', [0,1,2,3)]``.

#         """
#         r = {}
#         for obj in objs:
#             for key, value in obj:
#                 r.setdefault(frozenset(key.items()), []).append(value)
#         return [(dict(x), y) for x, y in r.items()]

#     def calculate(self, worker=multi_worker):
#         """Calculate Monte Carlo results for many demand vectors.

#         Returns a list of results with the format::

#             [(demand dictionary, [lca scores])]

#         There is no guarantee that the results are returned in the same order as the ``demand`` input variable.

#         """
#         with pool_adapter(multiprocessing.Pool(processes=self.cpus)) as pool:
#             results = pool.map(
#                 worker,
#                 [
#                     (projects.current, self.demands, self.method)
#                     for _ in range(self.iterations)
#                 ]
#             )
#         return self.merge_results(results)
