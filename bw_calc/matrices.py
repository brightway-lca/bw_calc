from .indexing import index_with_searchsorted, index_with_arrays
from .utils import load_array, MAX_SIGNED_32BIT_INT
from scipy import sparse
import numpy as np


class MatrixBuilder(object):
    """
The class, and its subclasses, load structured arrays, manipulate them, and generate `SciPy sparse matrices <http://docs.scipy.org/doc/scipy/reference/sparse.html>`_.

Matrix builders use an array of row indices, an array of column indices, and an array of values to create a `coordinate (coo) matrix <http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html>`_, which is then converted to a `compressed sparse row (csr) matrix <http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_.

See the following for more information on structured arrays:

* `NumPy structured arrays <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html#numpy.recarray>`_
* `Intermediate and processed data <https://docs.brightwaylca.org/intro.html#intermediate-and-processed-data>`_

These classes are not instantiated, and have only `classmethods <https://docs.python.org/2/library/functions.html#classmethod>`__. They are not really true classes, but more organizational. In other words, you should use:

.. code-block:: python

    MatrixBuilder.build(args)

and not:

.. code-block:: python

    mb = MatrixBuilder()
    mb.build(args)

    """

    @classmethod
    def build(cls, array, row_dict=None, col_dict=None, one_d=False, drop_missing=True):
        """
Build a sparse matrix from NumPy structured array(s).

See more detailed documentation at :ref:`building-matrices`.

This method does the following:

TODO: Update

#. Load and concatenate the :ref:`structured arrays files <building-matrices>` in filepaths ``paths`` using the function :func:`.utils.load_arrays` into a parameter array.
#. If not ``row_dict``, use :meth:`.build_dictionary` to build ``row_dict`` from the parameter array column ``"row_value"``.
#. Using the ``"row_value"`` and the ``row_dict``, use the method :meth:`.add_matrix_indices` to add matrix indices to the ``"row_index"`` column.
#. If not ``one_d``, do the same to ``col_dict`` and ``"col_index"``, using ``"col_value"``.
#. If not ``one_d``, use :meth:`.build_matrix` to build a sparse matrix using ``"amount"`` for the matrix data values, and ``"row_index"`` and ``"col_index"`` for row and column indices.
#. Else if ``one_d``, use :meth:`.build_diagonal_matrix` to build a diagonal matrix using ``"amount"`` for diagonal matrix data values and ``"row_index"`` as row/column indices.
#. Return the loaded parameter arrays from step 1, row and column dicts from steps 2 & 4, and matrix from step 5 or 6.

Args:
    * *paths* (list): List of array filepaths to load.
    * *"amount"* (str): Label of column in parameter arrays with matrix data values.
    * *"row_value"* (str): Label of column in parameter arrays with row ID values, i.e. the integer values returned from ``mapping``.
    * *"row_index"* (str): Label of column in parameter arrays where matrix row indices will be stored.
    * *"col_value"* (str, optional): Label of column in parameter arrays with column ID values, i.e. the integer values returned from ``mapping``. Not needed for diagonal matrices.
    * *"col_index"* (str, optional): Label of column in parameter arrays where matrix column indices will be stored. Not needed for diagonal matrices.
    * *row_dict* (dict, optional): Mapping dictionary linking ``"row_value"`` values to ``"row_index"`` values. Will be built if not given.
    * *col_dict* (dict, optional): Mapping dictionary linking ``"col_value"`` values to ``"col_index"`` values. Will be built if not given.
    * *one_d* (bool): Build diagonal matrix.
    * *drop_missing* (bool): Remove rows from the parameter array which aren't mapped by ``row_dict`` or ``col_dict``. Default is ``True``. Advanced use only.

Returns:
    A :ref:`numpy parameter array <building-matrices>`, the row mapping dictionary, the column mapping dictionary, and a COO sparse matrix.

        """
        if not row_dict:
            row_dict = index_with_searchsorted(
                array["row_value"],
                array["row_index"]
            )
        else:
            index_with_arrays(
                array["row_value"],
                array["row_index"],
                row_dict
            )

        if one_d:
            # Eliminate references to row data which isn't used;
            # Unused data remains MAX_SIGNED_32BIT_INT values
            if drop_missing:
                array = array[np.where(array["row_index"] != MAX_SIGNED_32BIT_INT)]
            matrix = cls.build_matrix(array, row_dict, one_d=True)
        else:
            if not col_dict:
                col_dict = index_with_searchsorted(
                    array["col_value"],
                    array["col_index"]
                )
            else:
                index_with_arrays(
                    array["col_value"],
                    array["col_index"],
                    col_dict
                )

            if drop_missing:
                array = array[np.where(array["row_index"] != MAX_SIGNED_32BIT_INT)]
                array = array[np.where(array["col_index"] != MAX_SIGNED_32BIT_INT)]

            matrix = cls.build_matrix(array, row_dict, col_dict)
        return row_dict, col_dict, matrix

    @classmethod
    def build_matrix(cls, array, row_dict, col_dict=None, one_d=False, new_data=None):
        """Build sparse matrix."""
        vector = (array["amount"] if new_data is None else new_data).copy()
        assert vector.shape[0] == array.shape[0], "Incompatible data & indices"
        vector[array["flip"]] *= -1
        # coo_matrix construction is coo_matrix((values, (rows, cols)),
        # (row_count, col_count))
        if one_d:
            return sparse.coo_matrix((
                vector.astype(np.float64),
                (array["row_index"], array["row_index"])),
                (len(row_dict), len(row_dict))).tocsr()
        else:
            return sparse.coo_matrix((
                vector.astype(np.float64),
                (array["row_index"], array["col_index"])),
                (len(row_dict), len(col_dict))).tocsr()
