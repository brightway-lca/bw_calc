from bw_calc import MonteCarloLCA, IterativeMonteCarloLCA
from numbers import Number
from pathlib import Path
import numpy as np
import os
import platform
import pytest


no_pool = pytest.mark.skipif(platform.system() == "Windows",
    reason="fork() on Windows doesn't pass temp directory")

yes_docker = pytest.mark.skipif(bool(os.environ.get("BRIGHTWAY2_DOCKER")),
    reason="Project directory in CI Docker container is '/home/jovyan/data'")
no_docker = pytest.mark.skipif(not bool(os.environ.get("BRIGHTWAY2_DOCKER")),
    reason="Normal project directory")

fixtures_dir = Path(__file__, "..").resolve() / "fixtures"


def get_args():
    fp = fixtures_dir / "basic-calculation-package" / "basic-calculation-package.zip"
    return {3: 1}, [fp]


def test_plain_monte_carlo():
    mc = MonteCarloLCA(*get_args())
    assert next(mc) > 0


def test_monte_carlo_as_iterator():
    mc = MonteCarloLCA(*get_args())
    for x in mc:
        assert x > 0
        break

def test_iterative_solving():
    mc = IterativeMonteCarloLCA(*get_args())
    assert next(mc)


# @no_pool
# def test_multi_mc(background):
#     mc = MultiMonteCarlo(
#         [
#             {("test", "1"): 1},
#             {("test", "2"): 1},
#             {("test", "1"): 1, ("test", "2"): 1}
#         ],
#         ("a", "method"),
#         iterations=10
#     )
#     results = mc.calculate()
#     assert results


# @no_pool
# def test_multi_mc_not_same_answer(background):
#     activity_list = [
#             {("test", "1"): 1},
#             {("test", "2"): 1},
#             # {("test", "1"): 1, ("test", "2"): 1}
#         ]
#     mc = MultiMonteCarlo(
#         activity_list,
#         ("a", "method"),
#         iterations=10
#     )
#     results = mc.calculate()
#     assert len(results) == 2
#     for _, lst in results:
#         assert len(set(lst)) == len(lst)

#     lca = LCA(activity_list[0], ("a", "method"))
#     lca.lci()
#     lca.lcia()

#     def score(lca, func_unit):
#         lca.redo_lcia(func_unit)
#         return lca.score

#     static = [score(lca, func_unit) for func_unit in activity_list]
#     for a, b in zip(static, results):
#         assert a not in b[1]

# @no_pool
# def test_multi_mc_compound_func_units(background):
#     activity_list = [
#             {("test", "1"): 1},
#             {("test", "2"): 1},
#             {("test", "1"): 1, ("test", "2"): 1}
#         ]
#     mc = MultiMonteCarlo(
#         activity_list,
#         ("a", "method"),
#         iterations=10
#     )
#     results = mc.calculate()
#     assert len(results) == 3
#     assert activity_list == mc.demands

# @random_project
# def test_multi_mc_no_temp_dir():
#     mc = MultiMonteCarlo(
#         [
#             {("test", "1"): 1},
#             {("test", "2"): 1},
#             {("test", "1"): 1, ("test", "2"): 1}
#         ],
#         ("a", "method"),
#         iterations=10
#     )
#     results = mc.calculate()
#     print(results)
#     assert results
#     assert isinstance(results, list)
#     assert len(results)

# @no_pool
# def test_parallel_monte_carlo(background):
#     fu, method = get_args()
#     mc = ParallelMonteCarlo(fu, method, iterations=200)
#     results = mc.calculate()
#     print(results)
#     assert results

# @random_project
# def test_parallel_monte_carlo_no_temp_dir():
#     fu, method = get_args()
#     mc = ParallelMonteCarlo(fu, method, iterations=200)
#     results = mc.calculate()
#     print(results)
#     assert results
#     assert isinstance(results, list)
#     assert isinstance(results[0], Number)
#     assert results[0] > 0
