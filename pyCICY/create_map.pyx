# cython: infer_types=True
import numpy as np
cimport cython
from cython.parallel import prange


DTYPE = np.intc
@cython.boundscheck(False)
@cython.wraparound(False)

cdef create_map(int[:,::1] V1basis, int[:,::1] V2basis, int[:,::1] fbasis, int[:] fmoduli):
    
    cdef Py_sice_t V1dim = V1basis.shape[0]
    cdef Py_sice_t V2dim = V2basis.shape[0]
    cdef Py_sice_t fdim = fbasis.shape[0]
    cdef Py_size_t n_vars = V1basis.shape[1]

    # smatrix
    result = np.zeros((V2dim, V1dim), dtype=DTYPE)
    cdef int[:, :] smatrix = result

    # tmp monomial
    tmp = np.zeros(n_vars, dtype=DTYPE)
    cdef int[:] monomial = tmp

    # bool for success
    cdef bool allowed = True
    cdef bool success = True

    cdef Py_ssize_t i, j, k, n

    for i in prange(V1dim, nogil=True):
        # loop over all moduli 'monomials'
        for j in range(fdim):
            # loop over all maps
            allowed = True
            for n in range(n_vars):
                # if y is bigger we simply multiply
                if fbasis[j][n] >= 0:
                    monomial[n] = V1basis[i][n]+fbasis[j][n]
                else:
                    # if smaller we see if taking the partial derivative yields zero
                    if abs(fbasis[j][n]) > V1basis[i][n]:
                        allowed = False
                        break
                    else:
                        monomial[n] = V1basis[i][n]+fbasis[j][n]

            if allowed:
                for k in range(V2dim):
                    success = True
                    for n in range(n_vars):
                        if not monomial[n] == V2basis[k][n]:
                            success = False
                            break
                    if success:
                        # can only be one matching; can we save k and run
                        # the next loop from (k_old,V2dim,1)?
                        smatrix[k][i] = fmoduli[j]

    return result