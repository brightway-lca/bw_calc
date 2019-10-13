from pathlib import Path
from brightway_projects.processing import (create_calculation_package, dictionary_formatter)

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
        'data': [
            dictionary_formatter({'row': 3, 'col': 5, 'amount': 1.}),
            dictionary_formatter({'row': 4, 'col': 6, 'amount': 1.}),
            dictionary_formatter({'row': 3, 'col': 6, 'amount': 0.5, "flip": True}),
        ],
    }, {
        'name': 'basic-biosphere',
        'matrix': 'biosphere',
        'data': [
            dictionary_formatter({'row': 1, 'col': 5, 'amount': 3.}),
            dictionary_formatter({'row': 2, 'col': 6, 'amount': 2.}),
        ],
    }, {
        'name': 'basic-characterization',
        'matrix': 'characterization',
        'data': [
            dictionary_formatter({'row': 1, 'amount': 10.}),
            dictionary_formatter({'row': 2, 'amount': 100.}),
        ],
    }]
    fp = create_calculation_package(directory, "test-fixture-basic-matrices", resources)
    fp.rename(directory / "basic-calculation-package.zip")

    directory = base_dir / "basic-cp-directory"
    directory.mkdir(mode=0o755, exist_ok=True)
    create_calculation_package(directory, "test-fixture-basic-matrices-directory", resources, compress=False)


if __name__ == "__main__":
    generate_basic_matrices()
