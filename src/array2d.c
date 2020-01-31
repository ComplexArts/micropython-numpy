#include <math.h>
#include <string.h>

#include "py/runtime.h"

#include "ndarray.h"

#include "array.h"
#include "binary.h"

#define perm_typecode NDARRAY_UINT16
#define TOL 1e-6
typedef uint16_t perm_t;

// Solve L x[:,k] = b[:,k] or U x[:,k] = b[:,k] in place
// Solution is in B
void
array2d_tr_solve(size_t m, size_t n, mp_float_t *ptrA,
                 size_t p, size_t k, mp_float_t *ptrB,
                 bool upper, bool unit_diag) {

    mp_int_t i, j; // i cannot be unsigned
    if (upper) {

        // m >= n
        // assume all zeros below n

        for (i = n - 1; i >= 0; i--) { // last nonzero row is n
            for (j = i + 1; j < n; j++)
                ptrB[i * p + k] -= ptrA[i * n + j] * ptrB[j * p + k];

            if (!unit_diag)
                ptrB[i * p + k] /= ptrA[i * n + i];
        }

    } else { // lower

        // m <= n
        // assume all zeros to the right of m

        for (i = 0; i < m; i++) {
            for (j = 0; j < i; j++)
                ptrB[i * p + k] -= ptrA[i * n + j] * ptrB[j * p + k];

            if (!unit_diag)
                ptrB[i * p + k] /= ptrA[i * n + i];

        }
    }
}

// QR

void
array2d_prod_Q_x(size_t m, size_t n, mp_float_t *A, mp_float_t *tau, size_t p, size_t l, mp_float_t *x, bool tp) {
    // A = Q R,    Q = H_1 H_2 ... H_n,     H_k = (I - beta_k v_k v_k^T),
    //
    // A \in R^{m x n},  Q \in R^{m x m},  R \in R^{m \times n}
    //
    // perform the product x[:,l] = Q x[:,l], x \in R^{n}
    //
    // y = H_1 H_2 ... H_n x = H_1 x_1, x_1 = H_2 x_2, ... x_n-1 = H_n x
    //
    // perform the product x[:,l] = Q^T x[:,l], x \in R^{n}
    //
    // y = H_n ... H_2 H_1 x = H_n x_n, x_n = H_n-1 x_n-1, ... x_2 = H_1 x
    //
    // In both cases:
    //
    // x_k-1 = H_k x_k = (I - beta_k v_k v_k^T) x_k,   x_k-1 = x_k - beta_k (v_k^T x_k) v_k
    size_t i, j, k;
    mp_int_t inc;
    mp_float_t rho, gamma;

    if (tp) {
        j = 0;
        inc = 1;
    } else {
        j = m - 1;
        inc = -1;
    }

    for (i = 0; i < m; i++, j += inc) {

        rho = x[j * p + l];
        for (k = j + 1; k < m; k++) {
            rho += A[k * n + j] * x[k * p + l];
        }

        gamma = tau[j] * rho;
        x[j * p + l] -= gamma;
        for (k = j + 1; k < m; k++) {
            x[k * p + l] -= gamma * A[k * n + j];
        }

    }
}

mp_float_t
array2d_house(size_t n, size_t ldx, mp_float_t *x, mp_float_t tol) {

    // sigma = ||x(2:n)||^2
    // v = [1; x(2:n)]
    size_t i;
    mp_float_t sigma = 0., beta = 0., mu, v0, v02, x0 = x[0], xi;
    for (i = 1; i < n; i++) {
        xi = x[i*ldx];
        sigma += xi * xi;
    }

#ifdef NDARRAY_DEBUG
    printf("n = %ld, x0 = %f, sigma = %f\n", n, x0, sigma);
#endif

    if (sigma > tol) {

        mu = sqrt(x0*x0 + sigma);
        v0 = (x0 <= 0 ? x0 - mu : -sigma/(x0 + mu));
        v02 = v0*v0;
        beta = 2*v02/(sigma + v02);

#ifdef NDARRAY_DEBUG
        printf("mu = %f\n", mu);
        printf("v0 = %f\n", v0);
        printf("beta = %f\n", beta);
#endif

        // v = v / v0
        for (i = 1; i < n; i++) {
            x[i * ldx] /= v0;
        }

        // store mu on v[0]
        x[0] = mu;

    } else if (x0 < 0) {
        beta = 2.;

        // flip sign on v[0]
        x[0] *= -1;
    }

    return beta;
}

void
array2d_perform_qr(ndarray_obj_t *A, ndarray_obj_t *tau, mp_float_t tol) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY2D_PERFORM_QR\n");
#endif

    // initialize
    mp_float_t beta, rho;
    size_t m = A->shape[0], n = A->shape[1];
    size_t k, j, l, r;

    // get pointer to data
    mp_float_t *ptrA = (mp_float_t*) array_get_ptr_array(A->typecode, A->array->items, 0);

    // get pointer to tau
    mp_float_t *ptrTau = (mp_float_t*) array_get_ptr_array(tau->typecode, tau->array->items, 0);

#ifdef NDARRAY_DEBUG
    printf("A = ");
    mp_obj_print_helper(&mp_sys_stdout_print, A, PRINT_REPR);
    printf("\n");
#endif

    // iterate
    r = m >= n ? n : m;
    for (j = 0; j < r; j++) {

        // house
        ptrTau[j] = beta = array2d_house(m - j, n, ptrA + j * n + j, tol);

        // v = [1; A[j+1:m,j]]
        // A[j:m, j:n] = (I - beta v v^T) A[j:m, j:n]
        // A[j:m, j:n] -= beta [1; A[j+1:m,j]] u^T, u^T = (v^T A[j:m, j:n])

        if (beta > tol) {

            for (l = j + 1; l < n; l++) {

                rho = ptrA[j * n + l];
                for (k = j + 1; k < m; k++) {
                    rho += ptrA[k * n + j] * ptrA[k * n + l];
                }

                ptrA[j * n + l] -= beta * rho;
                for (k = j + 1; k < m; k++) {
                    ptrA[k * n + l] -= beta * rho * ptrA[k * n + j];
                }

            }

        }

#ifdef NDARRAY_DEBUG
        printf("A = ");
        mp_obj_print_helper(&mp_sys_stdout_print, A, PRINT_REPR);
        printf("\n");
#endif

    }

}

mp_obj_t
array2d_qr(mp_obj_t self_obj) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }

    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_obj);

    if (self->dims != 2) {
        mp_raise_ValueError("expected matrix");
        return mp_const_none;
    }

    // cast as float
    ndarray_obj_t *qr_ndarray = array_astype(self, NDARRAY_FLOAT, true);

    // create tau vector
    size_t *shape = m_new(size_t, 1);
    shape[0] = qr_ndarray->shape[0];
    ndarray_obj_t *tau_ndarray = array_create_new(1, shape, NDARRAY_FLOAT, false);

    // perform qr decomposition
    array2d_perform_qr(qr_ndarray, tau_ndarray, TOL);

    // return qr
    return MP_OBJ_FROM_PTR(qr_ndarray);

}

mp_obj_t
array2d_qr_solve(mp_obj_t lhs_obj, mp_obj_t rhs_obj) {

    // are lhs and rhs ndarray?
    if (!MP_OBJ_IS_TYPE(lhs_obj, &ndarray_type) || !MP_OBJ_IS_TYPE(rhs_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }

    ndarray_obj_t *self_ndarray = MP_OBJ_TO_PTR(lhs_obj);
    ndarray_obj_t *b_ndarray = MP_OBJ_TO_PTR(rhs_obj);

    if (self_ndarray->dims != 2) {
        mp_raise_ValueError("expected matrix");
        return mp_const_none;
    }


    if (b_ndarray->dims != 1 && b_ndarray->dims != 2) {
        mp_raise_ValueError("b must be a 1d or 2d array");
        return mp_const_none;
    }

    if (self_ndarray->shape[0] != b_ndarray->shape[0]) {
        mp_raise_ValueError("b does not have compatible dimensions");
        return mp_const_none;
    }

    // cast as float, copy if already float
    ndarray_obj_t *qr_ndarray = array_astype(self_ndarray, NDARRAY_FLOAT, true);

    // create tau vector
    size_t *shape = m_new(size_t, 1);
    shape[0] = qr_ndarray->shape[0];
    ndarray_obj_t *tau_ndarray = array_create_new(1, shape, NDARRAY_FLOAT, false);

    // perform qr decomposition
    array2d_perform_qr(qr_ndarray, tau_ndarray, TOL);

    // get pointer to qr decomposition
    mp_float_t *ptrQR = (mp_float_t*) array_get_ptr_array(qr_ndarray->typecode, qr_ndarray->array->items, 0);

    // calculate rank
    size_t m = qr_ndarray->shape[0], n = qr_ndarray->shape[1];
    size_t r = (m <= n ? m : n);
    size_t rank = 0;
    for (size_t i = 0; i < r; i++, rank++) {
        if (fabs(ptrQR[i * n + i]) < TOL) {
            break;
        }
    }

    // immediate return in case it is singular
    if (n > m || rank < r) {
        mp_raise_ValueError("matrix is rank deficient to working precision, svd not implemented yet");
        return mp_const_none;
    }

    // get pointer to tau
    mp_float_t *ptrTau = (mp_float_t*) array_get_ptr_array(tau_ndarray->typecode, tau_ndarray->array->items, 0);

    // initialize solution, copy
    ndarray_obj_t *x_ndarray = array_astype(b_ndarray, NDARRAY_FLOAT, true);

    // get pointer to solution
    mp_float_t *ptrX = (mp_float_t*) array_get_ptr_array(x_ndarray->typecode, x_ndarray->array->items, 0);

    // solve
    mp_int_t p = b_ndarray->dims == 1 ? 1 : b_ndarray->shape[1];
    for (size_t j = 0; j < p; j++) {

        // calcualate y = Q^T b
        array2d_prod_Q_x(m, n, ptrQR, ptrTau, p, j, ptrX, true);

        // solve R x = y, R is upper
        array2d_tr_solve(m, n, ptrQR, p, j, ptrX, true, false);

    }

    // trim solution
    mp_obj_t result_obj;
    if (m > n) {

        // create view
        size_t newdims = x_ndarray->dims;
        mp_bound_slice_t *newslice = m_new(mp_bound_slice_t, newdims);
        newslice[0].start = 0;  newslice[0].stop = n;  newslice[0].step = 1;
        if (newdims == 2) {
            newslice[1] = x_ndarray->slice[1];
        }
        result_obj = MP_OBJ_FROM_PTR(array_create_new_view(x_ndarray, newdims, newslice, false));

    } else {

        // no need to trim
        result_obj = MP_OBJ_FROM_PTR(x_ndarray);

    }

    // calculate residues
    mp_obj_t residues_obj = array_dot(lhs_obj, result_obj);
    residues_obj = binary_op(MP_BINARY_OP_SUBTRACT, residues_obj, rhs_obj);

    // build tuple
    mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(mp_obj_new_tuple(2, NULL));
    tuple->items[0] = result_obj;
    tuple->items[1] = residues_obj;

    // return tuple
    return MP_OBJ_FROM_PTR(tuple);

}

// LU

mp_int_t
array2d_perform_lu(ndarray_obj_t *A, ndarray_obj_t *P, mp_float_t tol) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY2D_PERFORM_LU\n");
#endif

    // initialize
    mp_float_t maxA, absA;
    perm_t argMaxA, i, j, k;
    size_t m = A->shape[0], n = A->shape[1];
    mp_float_t buffer[n];
    perm_t number_of_permutations = 0;

    // get pointer to data
    mp_float_t *ptrA = (mp_float_t*) array_get_ptr_array(A->typecode, A->array->items, 0);

    // get pointer to permutation and initialize to identity
    perm_t *ptrP = (perm_t*) array_get_ptr_array(P->typecode, P->array->items, 0);
    for (i = 0; i < m; i++)
        ptrP[i] = i;

#ifdef NDARRAY_DEBUG
    printf("A = ");
    mp_obj_print_helper(&mp_sys_stdout_print, A, PRINT_REPR);
    printf("\n");

    printf("P = ");
    mp_obj_print_helper(&mp_sys_stdout_print, P, PRINT_REPR);
    printf("\n");
#endif

    // iterate
    for (i = 0; i < m; i++) {

        // pivot
        maxA = 0.0;
        argMaxA = i;
        for (k = i; k < m; k++)
            if ((absA = fabs(ptrA[k * n  + i])) > maxA) {
                maxA = absA;
                argMaxA = k;
            }

#ifdef NDARRAY_DEBUG
        printf("i = %d, argMaxA = %d\n", i, argMaxA);
#endif

        // stop if pivot is too small
        if (maxA < tol)
            return -i-1;

        // permute rows if necessary
        if (argMaxA != i) {

            // swap permutation
            j = ptrP[i];
            ptrP[i] = ptrP[argMaxA];
            ptrP[argMaxA] = j;

            // swap rows of A
            memcpy(buffer, ptrA + i * n, n * sizeof(mp_float_t));
            memcpy(ptrA + i * n, ptrA + argMaxA * n, n * sizeof(mp_float_t));
            memcpy(ptrA + argMaxA * n, buffer, n * sizeof(mp_float_t));

            // add to number of permutations
            number_of_permutations++;
        }

#ifdef NDARRAY_DEBUG
        printf("A = ");
        mp_obj_print_helper(&mp_sys_stdout_print, A, PRINT_REPR);
        printf("\n");

        printf("P = ");
        mp_obj_print_helper(&mp_sys_stdout_print, P, PRINT_REPR);
        printf("\n");
#endif

        // reduce
        for (j = i + 1; j < m; j++) {
            ptrA[j*n + i] /= ptrA[i*n + i];

            for (k = i + 1; k < n; k++)
                ptrA[j*n + k] -= ptrA[j*n + i] * ptrA[i*n + k];
        }

    }

    return number_of_permutations;
}

mp_obj_t
array2d_lu(mp_obj_t self_obj) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }

    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_obj);

    if (self->dims != 2) {
        mp_raise_ValueError("expected matrix");
        return mp_const_none;
    }

    // cast as float, copy if already float
    ndarray_obj_t *lu = array_astype(self, NDARRAY_FLOAT, true);

    // create permutation vector
    size_t *shape = m_new(size_t, 1);
    shape[0] = lu->shape[0];
    ndarray_obj_t *p = array_create_new(1, shape, perm_typecode, false);

    // perform lu decomposition
    array2d_perform_lu(lu, p, TOL);

    // return lu and p as tuple
    mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(mp_obj_new_tuple(2, NULL));
    tuple->items[0] = MP_OBJ_FROM_PTR(lu);
    tuple->items[1] = MP_OBJ_FROM_PTR(p);
    return tuple;

}

mp_obj_t
array2d_lu_det(mp_obj_t self_obj) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }

    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_obj);

    if (self->dims != 2) {
        mp_raise_ValueError("expected matrix");
        return mp_const_none;
    }

    if (self->shape[0] != self->shape[1]) {
        mp_raise_ValueError("expected square matrix");
        return mp_const_none;
    }

    // cast as float
    ndarray_obj_t *lu = array_astype(self, NDARRAY_FLOAT, true);

    // create permutation vector
    size_t *shape = m_new(size_t, 1);
    size_t n = shape[0] = lu->shape[0];
    ndarray_obj_t *p = array_create_new(1, shape, perm_typecode, false);

    // perform lu decomposition
    perm_t number_of_permutations = array2d_perform_lu(lu, p, TOL);

    // immediate return in case it is singular
    if (number_of_permutations < 0)
        return mp_obj_new_float(0);

    // get pointer to data
    mp_float_t *ptrLU = (mp_float_t*) array_get_ptr_array(lu->typecode, lu->array->items, 0);

    // calculate determinant
    mp_float_t det = *ptrLU;
    for (size_t i = 1; i < n; i++)
        det *= ptrLU[(n + 1) * i];

    // adjust sign
    if (number_of_permutations % 2 != 0)
        det *= -1.;

    return mp_obj_new_float(det);
}

mp_obj_t
array2d_lu_solve(mp_obj_t lhs_obj, mp_obj_t rhs_obj) {

    // are lhs and rhs ndarray?
    if (!MP_OBJ_IS_TYPE(lhs_obj, &ndarray_type) || !MP_OBJ_IS_TYPE(rhs_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }

    ndarray_obj_t *self_ndarray = MP_OBJ_TO_PTR(lhs_obj);
    ndarray_obj_t *b_ndarray = MP_OBJ_TO_PTR(rhs_obj);

    if (self_ndarray->dims != 2) {
        mp_raise_ValueError("expected matrix");
        return mp_const_none;
    }

    if (self_ndarray->shape[0] != self_ndarray->shape[1]) {
        mp_raise_ValueError("expected square matrix");
        return mp_const_none;
    }

    if (b_ndarray->dims != 1 && b_ndarray->dims != 2) {
        mp_raise_ValueError("b must be a 1d or 2d array");
        return mp_const_none;
    }

    if (self_ndarray->shape[0] != b_ndarray->shape[0]) {
        mp_raise_ValueError("b does not have compatible dimensions");
        return mp_const_none;
    }

    // cast as float
    ndarray_obj_t *lu_ndarray = array_astype(self_ndarray, NDARRAY_FLOAT, true);

    // create permutation vector
    size_t *shape = m_new(size_t, 1);
    size_t m = shape[0] = lu_ndarray->shape[0], n = m;
    ndarray_obj_t *p_ndarray = array_create_new(1, shape, perm_typecode, false);

    // perform lu decomposition
    mp_int_t number_of_permutations = array2d_perform_lu(lu_ndarray, p_ndarray, TOL);

    // immediate return in case it is singular
    if (number_of_permutations < 0) {
        mp_raise_ValueError("matrix is singular to working precision");
        return mp_const_none;
    }

    // get pointer to lu decomposition
    mp_float_t *ptrLU = (mp_float_t*) array_get_ptr_array(lu_ndarray->typecode, lu_ndarray->array->items, 0);

    // get pointer to lu permutation
    perm_t *ptrP = (perm_t*) array_get_ptr_array(p_ndarray->typecode, p_ndarray->array->items, 0);

    // get pointer to rhs
    mp_float_t *ptrB = (mp_float_t*) array_get_ptr_array(b_ndarray->typecode, b_ndarray->array->items, 0);

    // initialize solution
    ndarray_obj_t *x_ndarray = array_astype(b_ndarray, NDARRAY_FLOAT, true);

    // get pointer to solution
    mp_float_t *ptrX = (mp_float_t*) array_get_ptr_array(x_ndarray->typecode, x_ndarray->array->items, 0);

    // copy and permute x
    // TODO: this is really a advanced indexing operation, take care of slices
    mp_int_t p = b_ndarray->dims == 1 ? 1 : b_ndarray->shape[1];
    for (mp_int_t j = 0; j < p; j++) {
        for (mp_int_t i = 0; i < m; i++) {
            ptrX[i * p + j] = ptrB[ptrP[i] * p + j];
        }
    }

    // solve
    for (mp_int_t j = 0; j < p; j++) {

        // solve L y = b, L is lower and unit diagonal
        array2d_tr_solve(m, n, ptrLU, p, j, ptrX, false, true);

        // solve U x = y, U is upper
        array2d_tr_solve(m, n, ptrLU, p, j, ptrX, true, false);

    }

    return MP_OBJ_FROM_PTR(x_ndarray);
}

mp_obj_t
array2d_lu_inv(mp_obj_t self_obj) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }

    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_obj);

    if (self->dims != 2) {
        mp_raise_ValueError("expected matrix");
        return mp_const_none;
    }

    if (self->shape[0] != self->shape[1]) {
        mp_raise_ValueError("expected square matrix");
        return mp_const_none;
    }

    // cast as float
    ndarray_obj_t *lu = array_astype(self, NDARRAY_FLOAT, true);

    // create permutation vector
    size_t *shape = m_new(size_t, 1);
    size_t m = shape[0] = lu->shape[0];
    ndarray_obj_t *p = array_create_new(1, shape, perm_typecode, false);

    // perform lu decomposition
    mp_int_t number_of_permutations = array2d_perform_lu(lu, p, TOL);

    // immediate return in case it is singular
    if (number_of_permutations < 0) {
        mp_raise_ValueError("matrix is singular to working precision");
        return mp_const_none;
    }

    // get pointer to lu decomposition
    mp_float_t *ptrLU = (mp_float_t*) array_get_ptr_array(lu->typecode, lu->array->items, 0);

    // get pointer to lu permutation
    perm_t *ptrP = (perm_t*) array_get_ptr_array(p->typecode, p->array->items, 0);

    // initialize solution
    size_t *x_shape = m_new(size_t, 2);
    x_shape[0] = x_shape[1] = m;
    ndarray_obj_t *x = array_create_new(2, x_shape, NDARRAY_FLOAT, false);

    // get pointer to solution
    mp_float_t *ptrX = (mp_float_t*) array_get_ptr_array(x->typecode, x->array->items, 0);

    // solve
    for (mp_int_t j = 0; j < m; j++) {

        // solve
        for (mp_int_t i = 0; i < m; i++) {
            if (ptrP[i] == j)
                ptrX[i * m + j] = 1.;
            else
                ptrX[i * m + j] = 0.;

            for (mp_int_t k = 0; k < i; k++)
                ptrX[i * m + j] -= ptrLU[i * m + k] * ptrX[k * m + j];
        }

        for (mp_int_t i = m - 1; i >= 0; i--) {
            for (mp_int_t k = i + 1; k < m; k++)
                ptrX[i * m + j] -= ptrLU[i * m + k] * ptrX[k * m + j];

            ptrX[i * m + j] /= ptrLU[i * m + i];
        }
    }

    return MP_OBJ_FROM_PTR(x);
}
