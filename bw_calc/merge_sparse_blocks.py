import numpy as N
from scipy import sparse
# Modified from http://www.scipy.org/scipy/scipy/attachment/ticket/602/merge_sparse_blocks.py

def merge_sparse_blocks(block_mats, lookup_dicts, format='csr', dtype=N.float64):
    """.. centered:: Merge several sparse matrix blocks into a single sparse matrix

Input Params
============

block_mats -- Iterable of sparse matrices
lookup_dicts -- Iterable of lookup dictionaries (arbitrary value -> array index)
format -- Desired sparse format of output matrix
dtype -- Desired dtype, defaults to N.float64

Output
======

Merged matrix, to array indices dictionary, from array indices dictionary

Example
=======

Combine two matrices, and check that they are combined correctly, and that the lookup dictionaries are correct as well.

Matrix a:   [[0  1   2   3]
            [1  1   0   0]
            [2  0   2   0]
            [3  0   0   3]]

Matrix b:   [[4   5   6   7]
            [5  5   0   0]
            [6  0   6   0]
            [7  0   0   7]]

Lookup dict a: {20: 0, 21: 1, 22: 2, 23: 3}

Lookup dict b: {30: 0, 31: 1, 32: 2, 33: 3}

    >>> a = sparse.lil_matrix((4,4))
    >>> a[:,0] = range(4)
    >>> a[0,:] = range(4)
    >>> a.setdiag(range(4))

    >>> b = sparse.lil_matrix((4,4))
    >>> b[:,0] = range(4,8)
    >>> b[0,:] = range(4,8)
    >>> b.setdiag(range(4,8))

    >>> lookup_a = {20: 0, 21: 1, 22: 2, 23: 3}
    >>> lookup_b = {30: 0, 31: 1, 32: 2, 33: 3}

    >>> c, forwards, backwards = merge_sparse_blocks((a,b), (lookup_a, lookup_b))
    >>> assert isinstance(c, sparse.sparse.csr_matrix)
    >>> print forwards[20]
    0
    >>> print forwards[30]
    4
    >>> print backwards[1]
    21
    >>> print backwards[5]
    31
    >>> print N.array(c.todense())
    [[ 0.  1.  2.  3.  0.  0.  0.  0.]
     [ 1.  1.  0.  0.  0.  0.  0.  0.]
     [ 2.  0.  2.  0.  0.  0.  0.  0.]
     [ 3.  0.  0.  3.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  4.  5.  6.  7.]
     [ 0.  0.  0.  0.  5.  5.  0.  0.]
     [ 0.  0.  0.  0.  6.  0.  6.  0.]
     [ 0.  0.  0.  0.  7.  0.  0.  7.]]
    """
    def cumulative_sum(l):
         cum = list(l[:]) # Change if tuple to allow assignment
         csum_0 = 0
         csum_1 = 0
         csum_2 = 0
         for i, (x, y, z) in enumerate(l):
             csum_0 += x
             csum_1 += y
             csum_2 += z
             cum[i] = (csum_0, csum_1, csum_2)
         return cum

    block_mats = [mat.tocoo() for mat in block_mats]
    # Set up a triple of x offset, y offset, nnz (data length) offset
    offsets = cumulative_sum([(0,0,0)] + [(mat.shape[0], mat.shape[1], mat.nnz) for mat in block_mats]) # Last pair is not an offset, but final shape
    final_shape = offsets[-1]
    data = N.empty(final_shape[2], dtype=dtype) # Values in matrix
    row = N.empty(final_shape[2], dtype=N.intc) # Row indices
    col = N.empty(final_shape[2], dtype=N.intc) # Col indices
    for index, bm in enumerate(block_mats):
        data[offsets[index][2]:offsets[index+1][2]] = bm.data
        row[offsets[index][2]:offsets[index+1][2]] = bm.row+offsets[index][0]
        col[offsets[index][2]:offsets[index+1][2]] = bm.col+offsets[index][1] # Check x and y even though matrices are square

    merged_mat = sparse.coo_matrix((data, (row, col)), dtype=dtype)

    if format != 'coo':
        merged_mat = getattr(merged_mat, 'to'+format)()
        # if format == 'csc' or format == 'csr':
        #     if not merged_mat.has_sorted_indices: merged_mat.sort_indices()

    values = N.empty(final_shape[0], dtype=N.intc)
    keys = N.empty(final_shape[0], dtype=N.intc)
    for index, dic in enumerate(lookup_dicts):
        values[offsets[index][0]:offsets[index+1][0]] = N.array(dic.values()) + offsets[index][0]
        keys[offsets[index][0]:offsets[index+1][0]] = dic.keys()

    lookup_forward_dict = dict(zip(keys, values))
    lookup_backwards_dict = dict(zip(values, keys))

    return merged_mat, lookup_forward_dict, lookup_backwards_dict # Forward is into array

def merge_sparse_proxies(first, second):
    pass

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()