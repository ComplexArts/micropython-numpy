#include <math.h>
#include <string.h>

#include "py/runtime.h"

#include "ndarray.h"

#include "array.h"

#define perm_typecode NDARRAY_UINT16
#define TOL 1e-6
typedef uint16_t perm_t;

void
array2d_prod(size_t m, size_t n, size_t j, mp_float_t *v, mp_float_t *A, mp_float_t *u) {
    // perform the product u^T = v^T A[j:m, j:n]
    size_t i, k;
    for (k = j; k < n; k++) {
        u[k] = 0.;
        for (i = j; i<m; i++) {
            u[k] += v[i] * A[i * n + k];
        }
    }
}

mp_float_t
array2d_house(size_t n, size_t ldx, mp_float_t *x, mp_float_t *v, mp_float_t tol) {

    // sigma = ||x(2:n)||^2
    // v = [1; x(2:n)]
    size_t i;
    v[0] = 1.;
    mp_float_t sigma = 0., beta = 0., mu, v0, v02, x0 = x[0], xi;
    for (i = 1; i < n; i++) {
        xi = x[i*ldx];
        v[i] = xi;
        sigma += xi * xi;
    }

#ifdef NDARRAY_DEBUG
    printf("sigma = %f\n", sigma);
#endif

    if (sigma > tol) {
        mu = sqrt(x0*x0 + sigma);
#ifdef NDARRAY_DEBUG
        printf("mu = %f\n", mu);
#endif
        v0 = (x0 <= 0 ? x0 - mu : -sigma/(x0 + mu));
        v02 = v0*v0;
        beta = 2*v02/(sigma + v02);
#ifdef NDARRAY_DEBUG
        printf("beta = %f\n", beta);
#endif
        // v = v / v0
        for (i = 1; i < n; i++) {
            v[i] /= v0;
        }
    }

    return beta;
}

void
array2d_perform_qr(ndarray_obj_t *A, mp_float_t tol) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY2D_PERFORM_QR\n");
#endif

    // initialize
    mp_float_t beta;
    size_t m = A->shape[0], n = A->shape[1];
    size_t i, j, k, r;
    mp_float_t v[m], u[n];

    // get pointer to data
    mp_float_t *ptrA = (mp_float_t*) array_get_ptr_array(A->typecode, A->array->items, 0);

#ifdef NDARRAY_DEBUG
    printf("A = ");
    mp_obj_print_helper(&mp_sys_stdout_print, A, PRINT_REPR);
    printf("\n");
#endif

    // iterate
    r = m >= n ? n : m;
    for (j = 0; j < r; j++) {

        // house
        beta = array2d_house(m - j, n, ptrA + j * n + j, v + j, tol);

        // A[j:m, j:n] = (I - beta v v^T) A[j:m, j:n]
        //             = A[j:m, j:n] - beta v (v^T A[j:m, j:n])
        // A[j:m, j:n] -= beta v u^T, u^T = (v^T A[j:m, j:n])

        if (beta > 0) {
            // calculate u^T = (v^T A[j:m, j:n])
            array2d_prod(m, n, j, v, ptrA, u);

            // calculate A[j:m, j:n] -= beta v u^T, u^T = (v^T A[j:m, j:n])
            for (i = j; i < m; i++) {
                for (k = j; k < n; k++) {
                    ptrA[i * n + k] -= beta * v[i] * u[k];
                }
            }

            if (j <= m) {
                // calculate A[j + 1 : m, j] = v[2: m - j + 1]
                for (i = j + 1; i < m; i++) {
                    ptrA[i * n + j] = v[i];
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

mp_int_t
array2d_perform_lu(ndarray_obj_t *A, ndarray_obj_t *P, mp_float_t tol) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY2D_PERFORM_LU\n");
#endif

    // initialize
    mp_float_t maxA, absA;
    perm_t argMaxA, i, j, k;
    size_t n = A->shape[0], m = A->shape[1];
    mp_float_t buffer[m];
    perm_t number_of_permutations = 0;

    // get pointer to data
    mp_float_t *ptrA = (mp_float_t*) array_get_ptr_array(A->typecode, A->array->items, 0);

    // get pointer to permutation and initialize to identity
    perm_t *ptrP = (perm_t*) array_get_ptr_array(P->typecode, P->array->items, 0);
    for (i = 0; i < n; i++)
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
    for (i = 0; i < n; i++) {

        // pivot
        maxA = 0.0;
        argMaxA = i;
        for (k = i; k < n; k++)
            if ((absA = fabs(ptrA[k *m  + i])) > maxA) {
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
            memcpy(buffer, ptrA + i * m, m * sizeof(mp_float_t));
            memcpy(ptrA + i * m, ptrA + argMaxA * m, m * sizeof(mp_float_t));
            memcpy(ptrA + argMaxA * m, buffer, m * sizeof(mp_float_t));

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
        for (j = i + 1; j < n; j++) {
            ptrA[j*m + i] /= ptrA[i*m + i];

            for (k = i + 1; k < m; k++)
                ptrA[j*m + k] -= ptrA[j*m + i] * ptrA[i*m + k];
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

    // cast as float
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

    ndarray_obj_t *self = MP_OBJ_TO_PTR(lhs_obj);
    ndarray_obj_t *b = MP_OBJ_TO_PTR(rhs_obj);

    if (self->dims != 2) {
        mp_raise_ValueError("expected matrix");
        return mp_const_none;
    }

    if (self->shape[0] != self->shape[1]) {
        mp_raise_ValueError("expected square matrix");
        return mp_const_none;
    }

    if (b->dims != 1 && b->dims != 2) {
        mp_raise_ValueError("b must be a 1d or 2d array");
        return mp_const_none;
    }

    if (self->shape[1] != b->shape[0]) {
        mp_raise_ValueError("b does not have compatible dimensions");
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

    // get pointer to rhs
    mp_float_t *ptrB = (mp_float_t*) array_get_ptr_array(b->typecode, b->array->items, 0);

    // initialize solution
    ndarray_obj_t *x = array_astype(b, NDARRAY_FLOAT, true);

    // get pointer to solution
    mp_float_t *ptrX = (mp_float_t*) array_get_ptr_array(x->typecode, x->array->items, 0);

    // solve
    mp_int_t n = b->dims == 1 ? 1 : b->shape[1];
    for (mp_int_t j = 0; j < n; j++) {

        // solve
        for (mp_int_t i = 0; i < m; i++) {
            ptrX[i * n + j] = ptrB[ptrP[i] * n + j];

            for (mp_int_t k = 0; k < i; k++)
                ptrX[i * n + j] -= ptrLU[i * m + k] * ptrX[k * n + j];
        }

        for (mp_int_t i = m - 1; i >= 0; i--) {
            for (mp_int_t k = i + 1; k < m; k++)
                ptrX[i * n + j] -= ptrLU[i * m + k] * ptrX[k * n + j];

            ptrX[i * n + j] /= ptrLU[i * m + i];
        }
    }

    return MP_OBJ_FROM_PTR(x);
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
    ndarray_obj_t *qr = array_astype(self, NDARRAY_FLOAT, true);

    // perform qr decomposition
    array2d_perform_qr(qr, TOL);

    // return qr
    return MP_OBJ_FROM_PTR(qr);

}
