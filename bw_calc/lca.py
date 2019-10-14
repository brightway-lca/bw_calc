# -*- coding: utf-8 -*-
from .errors import (
    NonsquareTechnosphere,
    OutsideTechnosphere,
)
# import pandas
from .log_utils import create_logger
from .matrices import MatrixBuilder
from .utils import filter_data_for_matrix, load_data_obj
from collections.abc import Mapping
from scipy import sparse
import logging
import numpy as np
import warnings

try:
    from pypardiso import factorized, spsolve
except ImportError:
    from scipy.sparse.linalg import factorized, spsolve
try:
    from overrides import PackagesDataLoader
except ImportError:
    PackagesDataLoader = None


class LCA(object):
    """A static LCI or LCIA calculation.

    Following the general philosophy of Brightway, and good software practices, there is a clear separation of concerns between retrieving and formatting data and doing an LCA. Building the necessary matrices is done with MatrixBuilder objects (:ref:`matrixbuilders`). The LCA class only does the LCA calculations themselves.

    """
    #############
    ### Setup ###
    #############

    def __init__(self, demand, data_objs, log_config=None, overrides=None, seed=None, ignore_override_seed=None):
        """Create a new LCA calculation.

        Args:
            * *demand* (dict): The demand or functional unit. Needs to be a dictionary to indicate amounts, e.g. ``{(77: 2.5}``.
            * *data_obj*

        Returns:
            A new LCA object

        """
        if not isinstance(demand, Mapping):
            raise ValueError("Demand must be a dictionary")

        if log_config:
            create_logger(**log_config)
        self.logger = logging.getLogger('bw_calc')

        self.demand = demand
        self.data_objs = [load_data_obj(o) for o in data_objs]

        if overrides and PackagesDataLoader is None:
            warnings.warn("Skipping overrides; `overrides` not installed")
            self.overrides = None
        elif overrides:
            # Iterating over a `Campaign` object will also return the presample filepaths
            self.overrides = PackagesDataLoader(
                dirpaths=overrides,
                seed=self.seed if ignore_override_seed else None,
                lca=self
            )
        else:
            self.overrides = None

        self.logger.info("Created LCA object", extra={
            'demand': self.demand,
            # 'database_filepath': self.database_filepath,
            # 'method': self.method,
            # 'method_filepath': self.method_filepath,
            # 'normalization': self.normalization,
            # 'normalization_filepath': self.normalization_filepath,
            # 'overrides': str(self.overrides),
            # 'weighting': self.weighting,
            # 'weighting_filepath': self.weighting_filepath,
        })

    def build_demand_array(self, demand=None):
        """Turn the demand dictionary into a *NumPy* array of correct size.

        Args:
            * *demand* (dict, optional): Demand dictionary. Optional, defaults to ``self.demand``.

        Returns:
            A 1-dimensional NumPy array

        """
        demand = demand or self.demand
        self.demand_array = np.zeros(len(self.product_dict))
        for key in demand:
            try:
                self.demand_array[self.product_dict[key]] = demand[key]
            except KeyError:
                if key in self.activity_dict:
                    raise ValueError((u"LCA can only be performed on products,"
                        u" not activities ({} is the wrong dimension)"
                        ).format(key)
                    )
                else:
                    raise OutsideTechnosphere("Can't find key {} in product dictionary".format(key))

    #########################
    ### Data manipulation ###
    #########################

    def reverse_dict(self):
        """Construct reverse dicts from technosphere and biosphere row and col indices to input values.

        Returns:
            (reversed ``self.activity_dict``, ``self.product_dict`` and ``self.biosphere_dict``)
        """
        rev_activity = {v: k for k, v in self.activity_dict.items()}
        rev_product = {v: k for k, v in self.product_dict.items()}
        rev_bio = {v: k for k, v in self.biosphere_dict.items()}
        return rev_activity, rev_product, rev_bio

    ######################
    ### Data retrieval ###
    ######################

    def load_lci_data(self):
        """Load data and create technosphere and biosphere matrices."""
        self.tech_params = filter_data_for_matrix(self.data_objs, "technosphere")
        self.product_dict, self.activity_dict, self.technosphere_matrix = MatrixBuilder.build(self.tech_params)
        self.bio_params = filter_data_for_matrix(self.data_objs, "biosphere")
        self.biosphere_dict, _, self.biosphere_matrix = MatrixBuilder.build(self.bio_params, col_dict=self.activity_dict)
        if len(self.activity_dict) != len(self.product_dict):
            raise NonsquareTechnosphere((
                "Technosphere matrix is not square: {} activities (columns) and {} products (rows). "
                "Use LeastSquaresLCA to solve this system, or fix the input "
                "data").format(len(self.activity_dict), len(self.product_dict))
            )

        # if not self.biosphere_dict:
        #     warnings.warn("No biosphere flows found. No inventory results can "
        #                   "be calculated, `lcia` will raise an error")

        # Only need to index here for traditional LCA
        if self.overrides:
            self.overrides.index_arrays(self)
            self.overrides.update_matrices(
                matrices=('technosphere_matrix', 'biosphere_matrix')
            )

    def load_lcia_data(self):
        """Load data and create characterization matrix.

        """
        self.cf_params = filter_data_for_matrix(self.data_objs, "characterization")
        _, _, self.characterization_matrix = MatrixBuilder.build(self.cf_params, self.biosphere_dict, one_d=True)

        if self.overrides:
            self.overrides.update_matrices(matrices=['characterization_matrix'])

    # def load_normalization_data(self):
    #     """Load normalization data."""
    #     self.normalization_params, _, _, self.normalization_matrix = \
    #         builder.build(
    #             self.normalization_filepath,
    #             "amount",
    #             "flow",
    #             "index",
    #             row_dict=self._biosphere_dict,
    #             one_d=True
    #         )
    #     if self.overrides:
    #         self.overrides.update_matrices(matrices=['normalization_matrix',])

    # def load_weighting_data(self):
    #     """Load weighting data, a 1-element array."""
    #     self.weighting_params = load_array(
    #         self.weighting_filepath
    #     )
    #     self.weighting_value = self.weighting_params['amount']

    #     # TODO: This won't work because weighting is a value not a matrix
    #     # if self.overrides:
    #     #     self.overrides.update_matrices(self, ['weighting_value',])

    ####################
    ### Calculations ###
    ####################

    def decompose_technosphere(self):
        """
Factorize the technosphere matrix into lower and upper triangular matrices, :math:`A=LU`. Does not solve the linear system :math:`Ax=B`.

Doesn't return anything, but creates ``self.solver``.

.. warning:: Incorrect results could occur if a technosphere matrix was factorized, and then a new technosphere matrix was constructed, as ``self.solver`` would still be the factorized older technosphere matrix. You are responsible for deleting ``self.solver`` when doing these types of advanced calculations.

        """
        self.solver = factorized(self.technosphere_matrix.tocsc())

    def solve_linear_system(self):
        """
Master solution function for linear system :math:`Ax=B`.

    To most numerical analysts, matrix inversion is a sin.

    -- Nicolas Higham, Accuracy and Stability of Numerical Algorithms, Society for Industrial and Applied Mathematics, Philadelphia, PA, USA, 2002, p. 260.

We use `UMFpack <http://www.cise.ufl.edu/research/sparse/umfpack/>`_, which is a very fast solver for sparse matrices.

If the technosphere matrix has already been factorized, then the decomposed technosphere (``self.solver``) is reused. Otherwise the calculation is redone completely.

        """
        if hasattr(self, "solver"):
            return self.solver(self.demand_array)
        else:
            return spsolve(
                self.technosphere_matrix,
                self.demand_array)

    def lci(self, factorize=False):
        """
Calculate a life cycle inventory.

#. Load LCI data, and construct the technosphere and biosphere matrices.
#. Build the demand array
#. Solve the linear system to get the supply array and life cycle inventory.

Args:
    * *factorize* (bool, optional): Factorize the technosphere matrix. Makes additional calculations with the same technosphere matrix much faster. Default is ``False``; not useful is only doing one LCI calculation.
    * *builder* (``MatrixBuilder`` object, optional): Default is ``bw2calc.matrices.TechnosphereBiosphereMatrixBuilder``, which is fine for most cases. Custom matrix builders can be used to manipulate data in creative ways before building the matrices.

.. warning:: Custom matrix builders should inherit from ``TechnosphereBiosphereMatrixBuilder``, because technosphere inputs need to have their signs flipped to be negative, as we do :math:`A^{-1}f` directly instead of :math:`(I - A^{-1})f`.

Doesn't return anything, but creates ``self.supply_array`` and ``self.inventory``.

        """
        self.load_lci_data()
        self.build_demand_array()
        if factorize:
            self.decompose_technosphere()
        self.lci_calculation()

    def lci_calculation(self):
        """The actual LCI calculation.

        Separated from ``lci`` to be reusable in cases where the matrices are already built, e.g. ``redo_lci`` and Monte Carlo classes.

        """
        self.supply_array = self.solve_linear_system()
        # Turn 1-d array into diagonal matrix
        count = len(self.activity_dict)
        self.inventory = self.biosphere_matrix * \
            sparse.spdiags([self.supply_array], [0], count, count)

    def lcia(self):
        """
Calculate the life cycle impact assessment.

#. Load and construct the characterization matrix
#. Multiply the characterization matrix by the life cycle inventory

Args:
    * *builder* (``MatrixBuilder`` object, optional): Default is ``bw2calc.matrices.MatrixBuilder``, which is fine for most cases. Custom matrix builders can be used to manipulate data in creative ways before building the characterization matrix.

Doesn't return anything, but creates ``self.characterized_inventory``.

        """
        assert hasattr(self, "inventory"), "Must do lci first"
        self.load_lcia_data()
        self.lcia_calculation()

    def lcia_calculation(self):
        """The actual LCIA calculation.

        Separated from ``lcia`` to be reusable in cases where the matrices are already built, e.g. ``redo_lcia`` and Monte Carlo classes.

        """
        self.characterized_inventory = \
            self.characterization_matrix * self.inventory

    def normalize(self):
        """Multiply characterized inventory by flow-specific normalization factors."""
        assert hasattr(self, "characterized_inventory"), "Must do lcia first"
        if not hasattr(self, "normalization_matrix"):
            self.load_normalization_data()
        self.normalization_calculation()

    def normalization_calculation(self):
        """The actual normalization calculation.

        Creates ``self.normalized_inventory``."""
        self.normalized_inventory = \
            self.normalization_matrix * self.characterized_inventory

    # def weight(self):
    #     """Multiply characterized inventory by weighting value.

    #     Can be done with or without normalization."""
    #     assert hasattr(self, "characterized_inventory"), "Must do lcia first"
    #     if not hasattr(self, "weighting_value"):
    #         self.load_weighting_data()

    # def weighting_calculation(self):
    #     """The actual weighting calculation.

    #     Multiples weighting value by normalized inventory, if available, otherwise by characterized inventory.

    #     Creates ``self.weighted_inventory``."""
    #     if hasattr(self, "normalized_inventory"):
    #         obj = self.normalized_inventory
    #     else:
    #         obj = self.characterized_inventory
    #     self.weighted_inventory = self.weighting_value[0] * obj

    @property
    def score(self):
        """
The LCIA score as a ``float``.

Note that this is a `property <http://docs.python.org/2/library/functions.html#property>`_, so it is ``foo.lca``, not ``foo.score()``
        """
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        # if self.weighting:
        #     assert hasattr(self, "weighted_inventory"), "Must do weighting first"
        #     return float(self.weighted_inventory.sum())
        return float(self.characterized_inventory.sum())

    #########################
    ### Redo calculations ###
    #########################

    def rebuild_technosphere_matrix(self, vector):
        """Build a new technosphere matrix using the same row and column indices, but different values. Useful for Monte Carlo iteration or sensitivity analysis.

        Args:
            * *vector* (array): 1-dimensional NumPy array with length (# of technosphere parameters), in same order as ``self.tech_params``.

        Doesn't return anything, but overwrites ``self.technosphere_matrix``.

        """
        self.technosphere_matrix = MatrixBuilder.build_matrix(
            self.tech_params, self.product_dict, self.activity_dict, new_data=vector
        )

    def rebuild_biosphere_matrix(self, vector):
        """Build a new biosphere matrix using the same row and column indices, but different values. Useful for Monte Carlo iteration or sensitivity analysis.

        Args:
            * *vector* (array): 1-dimensional NumPy array with length (# of biosphere parameters), in same order as ``self.bio_params``.

        Doesn't return anything, but overwrites ``self.biosphere_matrix``.

        """
        self.biosphere_matrix = MatrixBuilder.build_matrix(
            self.bio_params, self.biosphere_dict, self.activity_dict, new_data=vector
        )

    def rebuild_characterization_matrix(self, vector):
        """Build a new characterization matrix using the same row and column indices, but different values. Useful for Monte Carlo iteration or sensitivity analysis.

        Args:
            * *vector* (array): 1-dimensional NumPy array with length (# of characterization parameters), in same order as ``self.cf_params``.

        Doesn't return anything, but overwrites ``self.characterization_matrix``.

        """
        self.characterization_matrix = MatrixBuilder.build_matrix(
            self.cf_params, self.biosphere_dict, one_d=True, new_data=vector
        )

    def redo_lci(self, demand=None):
        """Redo LCI with same databases but different demand.

        Args:
            * *demand* (dict): A demand dictionary.

        Doesn't return anything, but overwrites ``self.demand_array``, ``self.supply_array``, and ``self.inventory``.

        .. warning:: If you want to redo the LCIA as well, use ``redo_lcia(demand)`` directly.

        """
        assert hasattr(self, "inventory"), "Must do lci first"
        if demand is not None:
            self.build_demand_array(demand)
        self.lci_calculation()
        self.logger.info("Redoing LCI", extra={'demand': demand or self.demand})

    def redo_lcia(self, demand=None):
        """Redo LCIA, optionally with new demand.

        Args:
            * *demand* (dict, optional): New demand dictionary. Optional, defaults to ``self.demand``.

        Doesn't return anything, but overwrites ``self.characterized_inventory``. If ``demand`` is given, also overwrites ``self.demand_array``, ``self.supply_array``, and ``self.inventory``.

        """
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        if demand is not None:
            self.redo_lci(demand)
        self.lcia_calculation()
        self.logger.info("Redoing LCIA", extra={'demand': demand or self.demand})

    # def to_dataframe(self, cutoff=200):
    #     """Return all nonzero elements of characterized inventory as Pandas dataframe"""
    #     assert pandas, "This method requires the `pandas` (http://pandas.pydata.org/) library"
    #     assert hasattr(self, "characterized_inventory"), "Must do LCIA calculation first"

    #     from bw2data import get_activity

    #     coo = self.characterized_inventory.tocoo()
    #     stacked = np.vstack([np.abs(coo.data), coo.row, coo.col, coo.data])
    #     stacked.sort()
    #     rev_activity, _, rev_bio = self.reverse_dict()
    #     length = stacked.shape[1]

    #     data = []
    #     for x in range(min(cutoff, length)):
    #         if stacked[3, length - x - 1] == 0.:
    #             continue
    #         activity = get_activity(rev_activity[stacked[2, length - x - 1]])
    #         flow = get_activity(rev_bio[stacked[1, length - x - 1]])
    #         data.append((
    #             activity['name'],
    #             flow['name'],
    #             activity.get('location'),
    #             stacked[3, length - x - 1]
    #         ))
    #     return pandas.DataFrame(
    #         data,
    #         columns=['Activity', 'Flow', 'Region', 'Amount']
    #     )

    ####################
    ### Contribution ###
    ####################

    # def top_emissions(self, **kwargs):
    #     """Call ``bw2analyzer.ContributionAnalyses.annotated_top_emissions``"""
    #     try:
    #         from bw2analyzer import ContributionAnalysis
    #     except ImportError:
    #         raise ImportError("`bw2analyzer` is not installed")
    #     return ContributionAnalysis().annotated_top_emissions(self, **kwargs)

    # def top_activities(self, **kwargs):
    #     """Call ``bw2analyzer.ContributionAnalyses.annotated_top_processes``"""
    #     try:
    #         from bw2analyzer import ContributionAnalysis
    #     except ImportError:
    #         raise ImportError("`bw2analyzer` is not installed")
    #     return ContributionAnalysis().annotated_top_processes(self, **kwargs)
