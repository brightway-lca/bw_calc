from pathlib import Path
from bw_processing import create_calculation_package, dictionary_formatter
from copy import deepcopy

base_dir = Path(__file__, "..").resolve()


def generate_basic_matrices():
    directory = base_dir / "basic-calculation-package"
    directory.mkdir(mode=0o755, exist_ok=True)

    # Biosphere emissions: 1, 2
    # Products: 3, 4
    # Activities: 5, 6

    resources = [{
        'name': 'basic-technosphere',
        'matrix': 'technosphere',
        "path": "basic-technosphere.npy",
        'data': [
            dictionary_formatter({'row': 3, 'col': 5, 'amount': 1.}),
            dictionary_formatter({'row': 4, 'col': 6, 'amount': 1.}),
            dictionary_formatter({'row': 3, 'col': 6, 'amount': 0.5, "flip": True, 'uncertainty_type': 2, 'loc': 0, 'scale': 0.1}),
        ],
    }, {
        'name': 'basic-biosphere',
        'path': 'basic-biosphere.npy',
        'matrix': 'biosphere',
        'data': [
            dictionary_formatter({'row': 1, 'col': 5, 'amount': 3., 'uncertainty_type': 5, 'loc': 3., 'minimum': 1., 'maximum': 10}),
            dictionary_formatter({'row': 2, 'col': 6, 'amount': 2., 'uncertainty_type': 3, 'loc': 2., 'scale': 0.25, 'minimum': 0}),
        ],
    }, {
        'name': 'basic-characterization',
        'path': 'basic-characterization.npy',
        'matrix': 'characterization',
        'data': [
            dictionary_formatter({'row': 1, 'amount': 10., 'uncertainty_type': 7, 'minimum': 5, 'maximum': 15}),
            dictionary_formatter({'row': 2, 'amount': 100., 'uncertainty_type': 4, 'minimum': 75, 'maximum': 125}),
        ],
    }]
    fp = create_calculation_package(name="test-fixture-basic-matrices", resources=deepcopy(resources), path=directory)
    fp.rename(directory / "basic-calculation-package.zip")

    directory = base_dir / "basic-cp-directory"
    directory.mkdir(mode=0o755, exist_ok=True)
    create_calculation_package(name="test-fixture-basic-matrices-directory", resources=resources, path=directory, compress=False)


if __name__ == "__main__":
    generate_basic_matrices()
