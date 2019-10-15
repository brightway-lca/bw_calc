from brightway_projects.processing import (
    create_calculation_package,
    dictionary_formatter,
)
from bw_calc import LCA
from bw_calc.errors import NoArrays, OutsideTechnosphere, NonsquareTechnosphere
from pathlib import Path
import numpy as np
import pytest

fixtures_dir = Path(__file__, "..").resolve() / "fixtures"


def test_basic_calculation_zipfile():
    fp = fixtures_dir / "basic-calculation-package" / "basic-calculation-package.zip"
    lca = LCA({3: 1}, [fp])
    lca.lci()
    lca.lcia()
    assert lca.score == 30
    lca.redo_lcia({4: 1})
    assert lca.score == 200 + 30 / 2


def test_basic_calculation_directory():
    fp = fixtures_dir / "basic-cp-directory"
    lca = LCA({3: 1}, [fp])
    lca.lci()
    lca.lcia()
    assert lca.score == 30
    lca.redo_lcia({4: 1})
    assert lca.score == 200 + 30 / 2


def test_basic_calculation_in_memory():
    resources = [
        {
            "name": "a",
            "matrix": "technosphere",
            "data": [
                dictionary_formatter({"row": 3, "col": 5, "amount": 1.0}),
                dictionary_formatter({"row": 4, "col": 6, "amount": 1.0}),
                dictionary_formatter({"row": 3, "col": 6, "amount": 0.5, "flip": True}),
            ],
        },
        {
            "name": "basic-biosphere",
            "matrix": "biosphere",
            "data": [
                dictionary_formatter({"row": 1, "col": 5, "amount": 3.0}),
                dictionary_formatter({"row": 2, "col": 6, "amount": 2.0}),
            ],
        },
        {
            "name": "basic-characterization",
            "matrix": "characterization",
            "data": [
                dictionary_formatter({"row": 1, "amount": 10.0}),
                dictionary_formatter({"row": 2, "amount": 100.0}),
            ],
        },
    ]
    fp = create_calculation_package(
        name="test-fixture-basic-matrices", resources=resources, path=None, compress=False
    )
    lca = LCA({3: 1}, [fp])
    lca.lci()
    lca.lcia()
    assert lca.score == 30
    lca.redo_lcia({4: 1})
    assert lca.score == 200 + 30 / 2


def test_basic_calculation_multiple_packages():
    pass


def test_empty_biosphere():
    resources = [
        {
            "name": "a",
            "matrix": "technosphere",
            "data": [
                dictionary_formatter({"row": 3, "col": 5, "amount": 1.0}),
                dictionary_formatter({"row": 4, "col": 6, "amount": 1.0}),
                dictionary_formatter({"row": 3, "col": 6, "amount": 0.5, "flip": True}),
            ],
        },
        {
            "name": "basic-characterization",
            "matrix": "characterization",
            "data": [
                dictionary_formatter({"row": 1, "amount": 10.0}),
                dictionary_formatter({"row": 2, "amount": 100.0}),
            ],
        },
    ]
    fp = create_calculation_package(
        name="test-fixture-basic-matrices", resources=resources, path=None, compress=False
    )
    lca = LCA({3: 1}, [fp])
    with pytest.raises(NoArrays):
        lca.lci()


# @bw2test
# def test_warning_empty_biosphere():
#     test_data = {
#         ("t", "1"): {
#             'exchanges': [{
#                 'amount': 0.5,
#                 'input': ('t', "2"),
#                 'type': 'technosphere',
#                 'uncertainty type': 0}],
#             'type': 'process',
#             'unit': 'kg'
#             },
#         ("t", "2"): {
#             'exchanges': [],
#             'type': 'process',
#             'unit': 'kg'
#             },
#         }
#     test_db = Database("t")
#     test_db.write(test_data)
#     lca = LCA({("t", "1"): 1})
#     with pytest.warns(UserWarning):
#         lca.lci()


def test_redo_lci_fails_if_activity_outside_technosphere():
    fp = fixtures_dir / "basic-calculation-package" / "basic-calculation-package.zip"
    lca = LCA({3: 1}, [fp])
    lca.lci()
    with pytest.raises(OutsideTechnosphere):
        lca.redo_lci({10: 1})


def test_passing_zero_key():
    resources = [
        {
            "name": "a",
            "matrix": "technosphere",
            "data": [
                dictionary_formatter({"row": 3, "col": 5, "amount": 1.0}),
                dictionary_formatter({"row": 0, "col": 6, "amount": 1.0}),
                dictionary_formatter({"row": 3, "col": 6, "amount": 0.5, "flip": True}),
            ],
        },
        {
            "name": "basic-biosphere",
            "matrix": "biosphere",
            "data": [
                dictionary_formatter({"row": 1, "col": 5, "amount": 3.0}),
                dictionary_formatter({"row": 2, "col": 6, "amount": 2.0}),
            ],
        },
        {
            "name": "basic-characterization",
            "matrix": "characterization",
            "data": [
                dictionary_formatter({"row": 1, "amount": 10.0}),
                dictionary_formatter({"row": 2, "amount": 100.0}),
            ],
        },
    ]
    fp = create_calculation_package(
        name="test-fixture-basic-matrices", resources=resources, path=None, compress=False
    )
    lca = LCA({0: 1}, [fp])
    lca.lci()
    lca.lcia()
    assert lca.score == 200 + 30 / 2


def test_non_one_production_values():
    resources = [
        {
            "name": "a",
            "matrix": "technosphere",
            "data": [
                # Modified production exchange
                dictionary_formatter({"row": 3, "col": 5, "amount": 2.0}),
                dictionary_formatter({"row": 4, "col": 6, "amount": 1.0}),
                dictionary_formatter({"row": 3, "col": 6, "amount": 0.5, "flip": True}),
            ],
        },
        {
            "name": "basic-biosphere",
            "matrix": "biosphere",
            "data": [
                dictionary_formatter({"row": 1, "col": 5, "amount": 3.0}),
                dictionary_formatter({"row": 2, "col": 6, "amount": 2.0}),
            ],
        },
        {
            "name": "basic-characterization",
            "matrix": "characterization",
            "data": [
                dictionary_formatter({"row": 1, "amount": 10.0}),
                dictionary_formatter({"row": 2, "amount": 100.0}),
            ],
        },
    ]
    fp = create_calculation_package(
        name="test-fixture-basic-matrices", resources=resources, path=None, compress=False
    )
    lca = LCA({3: 1}, [fp])
    lca.lci()
    lca.lcia()
    assert lca.score == 15


def test_substitution():
    resources = [
        {
            "name": "a",
            "matrix": "technosphere",
            "data": [
                dictionary_formatter({"row": 3, "col": 5, "amount": 1.0}),
                dictionary_formatter({"row": 4, "col": 6, "amount": 1.0}),
                # Substitution exchange
                dictionary_formatter({"row": 3, "col": 6, "amount": 0.5}),
            ],
        },
        {
            "name": "basic-biosphere",
            "matrix": "biosphere",
            "data": [
                dictionary_formatter({"row": 1, "col": 5, "amount": 3.0}),
                dictionary_formatter({"row": 2, "col": 6, "amount": 2.0}),
            ],
        },
        {
            "name": "basic-characterization",
            "matrix": "characterization",
            "data": [
                dictionary_formatter({"row": 1, "amount": 10.0}),
                dictionary_formatter({"row": 2, "amount": 100.0}),
            ],
        },
    ]
    fp = create_calculation_package(
        name="test-fixture-basic-matrices", resources=resources, path=None, compress=False
    )
    lca = LCA({4: 1}, [fp])
    lca.lci()
    lca.lcia()
    assert lca.score == 200 - 30 / 2


def test_circular_chains():
    resources = [
        {
            "name": "a",
            "matrix": "technosphere",
            "data": [
                dictionary_formatter({"row": 3, "col": 5, "amount": 1.0}),
                dictionary_formatter({"row": 4, "col": 5, "amount": 0.1, "flip": True}),
                dictionary_formatter({"row": 4, "col": 6, "amount": 1.0}),
                dictionary_formatter({"row": 3, "col": 6, "amount": 0.5, "flip": True}),
            ],
        },
        {
            "name": "basic-biosphere",
            "matrix": "biosphere",
            "data": [
                dictionary_formatter({"row": 1, "col": 5, "amount": 3.0}),
                dictionary_formatter({"row": 2, "col": 6, "amount": 2.0}),
            ],
        },
        {
            "name": "basic-characterization",
            "matrix": "characterization",
            "data": [
                dictionary_formatter({"row": 1, "amount": 10.0}),
                dictionary_formatter({"row": 2, "amount": 100.0}),
            ],
        },
    ]
    fp = create_calculation_package(
        name="test-fixture-basic-matrices", resources=resources, path=None, compress=False
    )
    lca = LCA({4: 1}, [fp])
    lca.lci()
    lca.lcia()
    assert np.allclose(lca.score, 226.31578965117728)


def test_non_square_technsphere():
    resources = [
        {
            "name": "a",
            "matrix": "technosphere",
            "data": [
                dictionary_formatter({"row": 3, "col": 5, "amount": 1.0}),
                dictionary_formatter({"row": 4, "col": 6, "amount": 1.0}),
                dictionary_formatter({"row": 4, "col": 7, "amount": 1.0}),
                dictionary_formatter({"row": 3, "col": 6, "amount": 0.5, "flip": True}),
            ],
        },
        {
            "name": "basic-biosphere",
            "matrix": "biosphere",
            "data": [
                dictionary_formatter({"row": 1, "col": 5, "amount": 3.0}),
                dictionary_formatter({"row": 2, "col": 6, "amount": 2.0}),
            ],
        },
        {
            "name": "basic-characterization",
            "matrix": "characterization",
            "data": [
                dictionary_formatter({"row": 1, "amount": 10.0}),
                dictionary_formatter({"row": 2, "amount": 100.0}),
            ],
        },
    ]
    fp = create_calculation_package(
        name="test-fixture-basic-matrices", resources=resources, path=None, compress=False
    )
    lca = LCA({4: 1}, [fp])
    with pytest.raises(NonsquareTechnosphere):
        lca.lci()


def test_fu_outside_technosphere():
    fp = fixtures_dir / "basic-calculation-package" / "basic-calculation-package.zip"
    lca = LCA({13: 1}, [fp])
    with pytest.raises(OutsideTechnosphere):
        lca.lci()


def test_missing_technosphere():
    resources = [
        {
            "name": "basic-biosphere",
            "matrix": "biosphere",
            "data": [
                dictionary_formatter({"row": 1, "col": 5, "amount": 3.0}),
                dictionary_formatter({"row": 2, "col": 6, "amount": 2.0}),
            ],
        },
        {
            "name": "basic-characterization",
            "matrix": "characterization",
            "data": [
                dictionary_formatter({"row": 1, "amount": 10.0}),
                dictionary_formatter({"row": 2, "amount": 100.0}),
            ],
        },
    ]
    fp = create_calculation_package(
        name="test-fixture-basic-matrices", resources=resources, path=None, compress=False
    )
    lca = LCA({3: 1}, [fp])
    with pytest.raises(NoArrays):
        lca.lci()


def test_multiple_lci_lcia_calculations():
    fp = fixtures_dir / "basic-calculation-package" / "basic-calculation-package.zip"
    lca = LCA({3: 1}, [fp])
    lca.lci()
    lca.lci()
    lca.lcia()
    lca.lcia()
    assert lca.score == 30
    lca.lcia()
    assert lca.score == 30
    lca.redo_lcia()
    assert lca.score == 30
    lca.redo_lci({4: 1})
    assert lca.score == 30
    lca.redo_lcia({4: 1})
    assert lca.score == 200 + 30 / 2
