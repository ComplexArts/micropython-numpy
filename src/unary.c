#include <math.h>
#include "py/runtime.h"

#include "unary.h"

#include "binary.h"
#include "scalar.h"
#include "array.h"

#define ABS(x) (x < 0 ? -x : x)

#define UNARY_OP_INNER_LOOP(ptr, count, op) do {\
    for (size_t i = 0; i < count; i++) { \
        *ptr = op(*ptr); \
        ptr++; \
    } \
} while(0)

#define UNARY_OP_LOOP(ptr, count, op) do {\
    if (op == MP_UNARY_OP_INVERT) \
        if (array->typecode == NDARRAY_BOOL) \
            UNARY_OP_INNER_LOOP(ptr, count, !); \
        else \
            UNARY_OP_INNER_LOOP(ptr, count, ~); \
    else if (op == MP_UNARY_OP_NEGATIVE) \
        UNARY_OP_INNER_LOOP(ptr, count, -); \
    else if (op == MP_UNARY_OP_ABS) \
        UNARY_OP_INNER_LOOP(ptr, count, ABS); \
} while(0)

void
unary_array_op(mp_obj_array_t *array, size_t count,
               mp_unary_op_t op) {
    void *ptr = array_get_ptr_array(array->typecode, array->items, 0);
    if (array->typecode == NDARRAY_BOOL || array->typecode == NDARRAY_UINT8) {
        uint8_t* ptr_typed = (uint8_t *)ptr;
        UNARY_OP_LOOP(ptr_typed, count, op);
    } else if(array->typecode == NDARRAY_INT8) {
        int8_t* ptr_typed = (int8_t *)ptr;
        UNARY_OP_LOOP(ptr_typed, count, op);
    } else if(array->typecode == NDARRAY_UINT16) {
        uint16_t* ptr_typed = (uint16_t *)ptr;
        UNARY_OP_LOOP(ptr_typed, count, op);
    } else if(array->typecode == NDARRAY_INT16) {
        int16_t* ptr_typed = (int16_t *)ptr;
        UNARY_OP_LOOP(ptr_typed, count, op);
    } else {
        mp_float_t* ptr_typed = (mp_float_t *)ptr;
        if (op == MP_UNARY_OP_NEGATIVE)
            UNARY_OP_INNER_LOOP(ptr_typed, count, -);
        else if (op == MP_UNARY_OP_ABS)
            UNARY_OP_INNER_LOOP(ptr_typed, count, fabs);
    }
}

mp_obj_t
unary_op(mp_unary_op_t op, mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ndarray_obj_t *destination;

#ifdef NDARRAY_DEBUG
    printf("UNARY_OP = %d\n", op);
#endif

    switch (op) {
        case MP_UNARY_OP_LEN:
            return mp_obj_new_int(self->shape[0]);

        case MP_UNARY_OP_BOOL:

            if (self->typecode == NDARRAY_BOOL)
                // result self in case it is already bool
                return self_in;

            // otherwise cast to bool
            destination = array_create_new(self->dims, self->shape, NDARRAY_BOOL, false);
            array_astype_recursive(0, self, 0, destination, 0);

            return MP_OBJ_FROM_PTR(destination);

        case MP_UNARY_OP_INT:
            if (self->typecode == NDARRAY_INT8 || self->typecode == NDARRAY_INT16)
                // result self in case it is already int
                return self_in;
            // otherwise cast to int
            destination = array_create_new(self->dims, self->shape, NDARRAY_INT16, false);
            array_astype_recursive(0, self, 0, destination, 0);
            return MP_OBJ_FROM_PTR(destination);

        case MP_UNARY_OP_INVERT:
            if(self->array->typecode == NDARRAY_FLOAT) {
                mp_raise_ValueError("operation is not supported for floats");
                return mp_const_none;
            }
            // break to operate
            break;

        case MP_UNARY_OP_NEGATIVE:
            if(self->array->typecode == NDARRAY_BOOL) {
                mp_raise_ValueError("operation is not supported for booleans, use ~ instead");
                return mp_const_none;
            }
            // break to operate
            break;

        case MP_UNARY_OP_POSITIVE:
            return array_copy(self_in);

        case MP_UNARY_OP_ABS:
            if((self->array->typecode == NDARRAY_UINT8) || (self->array->typecode == NDARRAY_UINT16)
               || (self->array->typecode == NDARRAY_BOOL)) {
                return array_copy(self_in);
            }
            // break to operate
            break;

        default: return MP_OBJ_NULL; // operator not supported
    }

    // copy, operate, and return
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(array_copy(self_in));

#ifdef NDARRAY_DEBUG
    printf("COPY:\n");
    ndarray_print_debug(ndarray);
#endif

    // no need to recurse, array is always packed after copy
    unary_array_op(ndarray->array, ndarray->array->len, op);

    return MP_OBJ_FROM_PTR(ndarray);
}

// UNARY FUNCTION

#define UNARY_FUN_AND(x, y) x = (x && y)
#define UNARY_FUN_OR(x, y)  x = (x || y)
#define UNARY_FUN_MAX(x, y) x = (y >= x ? y : x)
#define UNARY_FUN_MIN(x, y) x = (y <= x ? y : x)
#define UNARY_FUN_SUM(x, y) x += y
#define UNARY_FUN_PROD(x, y) x *= y

#define UNARY_FUN_INNER_LOOP(dest, ptr, step, count, op) do { \
    for (size_t i = 0; i < count; i++) { \
         op(dest, *ptr); \
         ptr += step; \
    } \
} while(0)

#define UNARY_FUN_LOOP(dest, ptr, step, count, op) do {\
    if (op == UNARY_OP_ALL) \
        UNARY_FUN_INNER_LOOP(dest, ptr, step, count, UNARY_FUN_AND); \
    else if (op == UNARY_OP_ANY) \
        UNARY_FUN_INNER_LOOP(dest, ptr, step, count, UNARY_FUN_OR); \
    else if (op == UNARY_OP_MAX) \
        UNARY_FUN_INNER_LOOP(dest, ptr, step, count, UNARY_FUN_MAX); \
    else if (op == UNARY_OP_MIN) \
        UNARY_FUN_INNER_LOOP(dest, ptr, step, count, UNARY_FUN_MIN); \
    else if (op == UNARY_OP_SUM) \
        UNARY_FUN_INNER_LOOP(dest, ptr, step, count, UNARY_FUN_SUM); \
    else /* if (op == UNARY_OP_SUM) */ \
        UNARY_FUN_INNER_LOOP(dest, ptr, step, count, UNARY_FUN_PROD); \
} while(0)

#define UNARY_FUN_LOOP_FLOAT(dest, ptr, step, count, op) do {\
    if (op == UNARY_OP_MAX) \
        UNARY_FUN_INNER_LOOP(dest, ptr, step, count, UNARY_FUN_MAX); \
    else if (op == UNARY_OP_MIN) \
        UNARY_FUN_INNER_LOOP(dest, ptr, step, count, UNARY_FUN_MIN); \
    else if (op == UNARY_OP_SUM) \
        UNARY_FUN_INNER_LOOP(dest, ptr, step, count, UNARY_FUN_SUM); \
    else /* if (op == UNARY_OP_SUM) */ \
        UNARY_FUN_INNER_LOOP(dest, ptr, step, count, UNARY_FUN_PROD); \
} while(0)

mp_obj_t
unary_fun_array(mp_obj_t dest_obj,
                        mp_obj_array_t *array, size_t offset, mp_int_t step, size_t count,
                        unary_op_t op) {

#ifdef NDARRAY_DEBUG
    printf("SCALAR_ARRAY_OP, op = %d, typecode = '%c', offset = %ld, step = %ld, count = %ld\n", op, array->typecode, offset, step, count);
#endif

    void *ptr = array_get_ptr_array(array->typecode, array->items, offset);
    if (array->typecode == NDARRAY_FLOAT) {
        mp_float_t dest = mp_obj_get_float(dest_obj);
        mp_float_t* ptr_typed = (mp_float_t *)ptr;
        UNARY_FUN_LOOP_FLOAT(dest, ptr_typed, step, count, op);
        return mp_obj_new_float(dest);
    } else {
        if (array->typecode == NDARRAY_BOOL) {
            uint8_t dest = mp_obj_get_int(dest_obj);
            uint8_t * ptr_typed = (uint8_t *) ptr;
            UNARY_FUN_LOOP(dest, ptr_typed, step, count, op);
            return mp_obj_new_bool(dest);
        } else if (array->typecode == NDARRAY_UINT8) {
            uint8_t dest = mp_obj_get_int(dest_obj);
            uint8_t * ptr_typed = (uint8_t *) ptr;
            UNARY_FUN_LOOP(dest, ptr_typed, step, count, op);
            return mp_obj_new_int_from_uint(dest);
        } else if (array->typecode == NDARRAY_INT8) {
            int8_t dest = mp_obj_get_int(dest_obj);
            int8_t *ptr_typed = (int8_t *) ptr;
            UNARY_FUN_LOOP(dest, ptr_typed, step, count, op);
            return mp_obj_new_int(dest);
        } else if (array->typecode == NDARRAY_UINT16) {
            uint16_t dest = mp_obj_get_int(dest_obj);
            uint16_t *ptr_typed = (uint16_t *) ptr;
            UNARY_FUN_LOOP(dest, ptr_typed, step, count, op);
            return mp_obj_new_int_from_uint(dest);
        } else /* if (array->typecode == NDARRAY_INT16) */ {
            int16_t dest = mp_obj_get_int(dest_obj);
            int16_t *ptr_typed = (int16_t *) ptr;
            UNARY_FUN_LOOP(dest, ptr_typed, step, count, op);
            return mp_obj_new_int(dest);
        }
    }
}

mp_obj_t
unary_fun_recursive(size_t axis, size_t level,
                    ndarray_obj_t *origin, mp_uint_t origin_offset,
                    mp_obj_t destination, mp_uint_t destination_offset,
                    unary_op_t op) {

    // get current slice and shape
    mp_bound_slice_t origin_slice = origin->slice[level];
    size_t n = origin->shape[level];

#ifdef NDARRAY_DEBUG
    printf("UNARY_FUN_RECURSIVE axis = %ld, level = %ld, offset = %ld, destination_offset = %ld, count = %ld, op = %d\n",
           axis, level, origin_offset, destination_offset, n, op);
#endif

    // increment level
    level++;

    // calculate remaining origin depth
    mp_uint_t origin_level_depth = 1;
    for (mp_uint_t i = level; i < origin->dims; i++) {
        origin_level_depth *= origin->array_shape[i];
    }

    // calculate remaining destination depth
    mp_uint_t destination_level_depth = 0;
    ndarray_obj_t *dest_ndarray = NULL;
    if (destination != MP_OBJ_NULL && mp_obj_is_type(destination, &ndarray_type)) {
        destination_level_depth = 1;
        dest_ndarray = MP_OBJ_TO_PTR(destination);
        for (mp_uint_t i = level; i < dest_ndarray->dims; i++) {
            destination_level_depth *= dest_ndarray->array_shape[i];
        }
    }

    if (axis >= 0 && level - 1 == axis) {

#ifdef NDARRAY_DEBUG
        printf("AXIS == LEVEL\n");
#endif

        if (level < origin->dims) {

#ifdef NDARRAY_DEBUG
            printf("UPPER LEVEL\n");
#endif

            if (n == 1) {

#ifdef NDARRAY_DEBUG
                printf("N = 1, COPY\n");
#endif

                // upper level
                array_copy_recursive(level,
                                     origin, origin_offset,
                                     level - 1,
                                     dest_ndarray, destination_offset);

            } else {

#ifdef NDARRAY_DEBUG
                printf("N = %ld > 1, OPERATE\n", n);
#endif

                // operate first two items
                mp_obj_t lhs = origin, rhs = origin;
                mp_uint_t lhs_index = origin_slice.start, rhs_index = lhs_index + origin_slice.step;
                mp_uint_t lhs_offset = origin_offset + lhs_index * origin_level_depth,
                        rhs_offset = origin_offset + rhs_index * origin_level_depth;

                mp_binary_op_t binary_op;
                binary_op_fun_t binary_op_fun;
                if (op == UNARY_OP_ALL) {
                    binary_op = MP_BINARY_OP_AND;
                    binary_op_fun = binary_op_and;
                } else if (op == UNARY_OP_ANY) {
                    binary_op = MP_BINARY_OP_OR;
                    binary_op_fun = binary_op_or;
                } else if (op == UNARY_OP_MAX) {
                    binary_op = NDARRAY_BINARY_OP_MAX;
                    binary_op_fun = binary_op_max;
                } else if (op == UNARY_OP_MIN) {
                    binary_op = NDARRAY_BINARY_OP_MIN;
                    binary_op_fun = binary_op_min;
                } else if (op == UNARY_OP_SUM) {
                    binary_op = MP_BINARY_OP_ADD;
                    binary_op_fun = binary_op_add;
                } else /* if (op == UNARY_OP_PROD) */ {
                    binary_op = MP_BINARY_OP_MULTIPLY;
                    binary_op_fun = binary_op_multiply;
                }

                binary_op_recursive(level,
                                    dest_ndarray, destination_offset,
                                    lhs, lhs_offset,
                                    rhs, rhs_offset,
                                    binary_op, binary_op_fun);

                // set lhs to destination
                lhs = destination;
                lhs_offset = destination_offset;

                // iterate remaining operands
                for (mp_uint_t i = 2; i < n; i++) {
                    rhs_index += origin_slice.step;
                    rhs_offset = origin_offset + rhs_index * origin_level_depth;
                    binary_op_recursive(level,
                                        dest_ndarray, destination_offset,
                                        lhs, lhs_offset,
                                        rhs, rhs_offset,
                                        binary_op, binary_op_fun);
                }

            }
        } else {

#ifdef NDARRAY_DEBUG
            printf("LAST LEVEL\n");
#endif

            if (n == 1) {

#ifdef NDARRAY_DEBUG
                printf("N = 1, COPY\n");
#endif

                // last level
                array_copy_recursive(level - 1,
                                     origin, origin_offset,
                                     level - 2,
                                     dest_ndarray, destination_offset);

            } else {

#ifdef NDARRAY_DEBUG
                printf("N = %ld > 1, OPERATE\n", n);
#endif

                // apply operator
                origin_offset += origin_slice.start * origin_level_depth;
                // set first value
                mp_obj_t dest_value = array_get_val_array(origin->typecode, origin->array->items, origin_offset);
                // advance and decrement number of items
                origin_offset += origin_slice.step * origin_level_depth;
                n--;
                dest_value = unary_fun_array(dest_value,
                                             origin->array,
                                             origin_offset,
                                             origin_slice.step * origin_level_depth, n,
                                             op);
                // store at destination
                array_set_val_array(dest_ndarray->typecode,
                                    dest_ndarray->array->items, destination_offset, dest_value);

            }

        }

        // do not recurse anymore
        return destination;

    } else if (level < origin->dims) {

        // upper level
        mp_uint_t index = origin_slice.start;
        for (mp_uint_t i = 0; i < n; i++) {
            destination = unary_fun_recursive(axis, level,
                                              origin, origin_offset + index * origin_level_depth,
                                              destination,
                                              destination_offset + index * destination_level_depth, op);
            index += origin_slice.step;
        }

    } else {

        // last level, operate
        origin_offset += origin_slice.start * origin_level_depth;
        // set value?
        if (destination == MP_OBJ_NULL) {
            destination = array_get_val_array(origin->typecode, origin->array->items, origin_offset);
            // advance and decrement number of items
            origin_offset += origin_slice.step * origin_level_depth;
            n--;
        }
        destination = unary_fun_array(destination,
                                      origin->array,
                                      origin_offset,
                                      origin_slice.step * origin_level_depth, n,
                                      op);

    }

    return destination;
}

mp_obj_t
unary_fun(unary_op_t op, mp_obj_t self_in, mp_int_t axis) {

    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);

    // is axis out of bound?
    if (axis >= 0 && axis > self->dims - 1) {
        mp_raise_ValueError("Axis is out of bounds");
        return mp_const_none;
    }

    // is axis only one?
    if (axis == 0 && self->dims == 1)
        // same as None
        axis = -1;

    if (axis < 0 && self->size == self->array_size) {

#ifdef NDARRAY_DEBUG
        printf("UNARY_FUN: ARRAY IS PACKED, typecode = %c\n", self->array->typecode);
        ndarray_print_debug(self);
        mp_obj_print_helper(&mp_sys_stdout_print, self_in, PRINT_REPR);
        printf("\n");
#endif

        // array is packed, iterate all entries
        mp_obj_t destination = array_get_val_array(self->array->typecode, self->array->items, 0);

#ifdef NDARRAY_DEBUG
        printf("UNARY_FUN: destination = ");
        mp_obj_print_helper(&mp_sys_stdout_print, destination, PRINT_REPR);
        printf("\n");
#endif

        return unary_fun_array(destination, self->array, 1, 1, self->array->len - 1, op);

    } else {

#ifdef NDARRAY_DEBUG
        printf("UNARY_FUN: ARRAY IS NOT PACKED\n");
        ndarray_print_debug(MP_OBJ_TO_PTR(self));
#endif

        mp_obj_t destination = MP_OBJ_NULL;
        if (axis >= 0) {

            // create destination array flattened at axis, do not initialize
            size_t dims = self->dims - 1;
            size_t *shape = m_new(size_t, dims);
            size_t j = 0;
            for (size_t i = 0; i < self->dims; i++) {
                if (i == axis)
                    // skip axis
                    continue;
                shape[j++] = self->shape[i];
            }

            ndarray_obj_t *ndarray = array_create_new(dims, shape, self->typecode, false);
            destination = MP_OBJ_FROM_PTR(ndarray);

        }

        // recurse
        return unary_fun_recursive(axis, 0, self, 0, destination, 0, op);

    }

}

// UNARY_FUN

mp_obj_t
unary_fun_create_result(mp_obj_t lhs, char typecode) {

    // result must be float
    size_t dims;
    size_t *shape;

    if (!mp_obj_is_type(lhs, &ndarray_type)) {

        mp_raise_ValueError("operand must be an ndarray");
        return mp_const_none;

    } else {

        // left-hand side is an ndarray
        ndarray_obj_t *lhs_ndarray = MP_OBJ_TO_PTR(lhs);

        // typecode
        if (typecode == NDARRAY_UNKNOWN)
            typecode = lhs_ndarray->typecode;

        // result array shape is the same as lhs
        dims = lhs_ndarray->dims;
        shape = m_new(size_t, dims);
        for (size_t i = 0; i < dims; i++)
            shape[i] = lhs_ndarray->shape[i];

    }

    // create result array
    return MP_OBJ_FROM_PTR(array_create_new(dims, shape, typecode, false));

}

#define UNARY_FUN_GENERIC_LOOP(ptr_result, ptr1, step1, count, op) do {\
    for (size_t i = 0; i < count; i++) { \
        *ptr_result++ = op(*ptr1); \
        ptr1 += step1; \
    } \
} while(0)

#define UNARY_TYPECAST_RHS(self_ptr_typed, lhs_typecode, lhs_ptr, lhs_step, count, op) do { \
    if (lhs_typecode == NDARRAY_BOOL || lhs_typecode == NDARRAY_UINT8) { \
        uint8_t *(lhs_ptr_typed) = (uint8_t *) (lhs_ptr); \
        UNARY_FUN_GENERIC_LOOP(self_ptr_typed, lhs_ptr_typed, lhs_step, count, op); \
    } else if (lhs_typecode == NDARRAY_INT8) { \
        int8_t *(lhs_ptr_typed) = (int8_t *) (lhs_ptr); \
        UNARY_FUN_GENERIC_LOOP(self_ptr_typed, lhs_ptr_typed, lhs_step, count, op); \
    } else if (lhs_typecode == NDARRAY_UINT16) { \
        uint16_t *(lhs_ptr_typed) = (uint16_t *) (lhs_ptr); \
        UNARY_FUN_GENERIC_LOOP(self_ptr_typed, lhs_ptr_typed, lhs_step, count, op); \
    } else if (lhs_typecode == NDARRAY_INT16) { \
        int16_t *(lhs_ptr_typed) = (int16_t *) (lhs_ptr); \
        UNARY_FUN_GENERIC_LOOP(self_ptr_typed, lhs_ptr_typed, lhs_step, count, op); \
    } else { \
        mp_float_t *(lhs_ptr_typed) = (mp_float_t *) (lhs_ptr); \
        UNARY_FUN_GENERIC_LOOP(self_ptr_typed, lhs_ptr_typed, lhs_step, count, op); \
    } \
} while(0)

typedef mp_float_t (*mp_float_fun_t)(mp_float_t);

void
unary_generic_fun_recursive(size_t level,
                            ndarray_obj_t *self_ndarray, size_t self_offset,
                            mp_obj_t lhs, size_t lhs_offset,
                            mp_float_fun_t fun) {

#ifdef NDARRAY_DEBUG
    printf("UNARY_GENERIC_FUN_RECURSIVE level = %ld, self_offset = %ld, lhs_offset = %ld\n",
           level, self_offset, lhs_offset);
#endif

    // get current slices and shapes
    mp_uint_t self_depth = 1;
    mp_bound_slice_t self_slice = self_ndarray->slice[level];
    size_t n = self_ndarray->shape[level];

    // calculate remaining depth
    for (mp_uint_t i = level + 1; i < self_ndarray->dims; i++) {
        self_depth *= self_ndarray->array_shape[i];
    }

    // left hand side
    mp_uint_t lhs_depth = 1;
    ndarray_obj_t *lhs_ndarray = MP_OBJ_TO_PTR(lhs);
    mp_bound_slice_t lhs_slice = lhs_ndarray->slice[level];
    n = lhs_ndarray->shape[level];

    // calculate remaining depth
    for (mp_uint_t i = level + 1; i < lhs_ndarray->dims; i++) {
        lhs_depth *= lhs_ndarray->array_shape[i];
    }

    // increment level
    level++;

    mp_uint_t self_index = self_slice.start, lhs_index = lhs_slice.start;
    if (level < self_ndarray->dims) {

        // upper level, recurse
        for (mp_uint_t i = 0; i < n; i++) {
            unary_generic_fun_recursive(level,
                                        self_ndarray, self_offset + self_index * self_depth,
                                        lhs, lhs_offset + lhs_index * lhs_depth,
                                        fun);
            self_index += self_slice.step;
            lhs_index += lhs_slice.step;
        }

    } else {

        // last level, operate
        mp_float_t *self_ptr_typed = array_get_ptr_array(self_ndarray->typecode, self_ndarray->array->items,
                                                         self_offset + self_index * self_depth);

        void *lhs_ptr = array_get_ptr_array(lhs_ndarray->typecode, lhs_ndarray->array->items,
                                            lhs_offset + lhs_index * lhs_depth);

        mp_int_t lhs_step = lhs_slice.step * lhs_depth;
        UNARY_TYPECAST_RHS(self_ptr_typed, lhs_ndarray->typecode, lhs_ptr, lhs_step, n, fun);

    }

}

#define UNARY_FUN_FLOAT_1(py_name, c_name) \
    mp_obj_t unary_fun_ ## py_name(mp_obj_t lhs) { \
        if (!mp_obj_is_type(lhs, &ndarray_type)) { \
            mp_raise_ValueError("expected ndarray"); \
            return mp_const_none; \
        } \
        /* typecode defaults to NDARRAY_FLOAT */ \
        mp_obj_t result = unary_fun_create_result(lhs, NDARRAY_FLOAT); \
        ndarray_obj_t *result_ndarray = MP_OBJ_TO_PTR(result); \
        /* operate */ \
        unary_generic_fun_recursive(0, result_ndarray, 0, lhs, 0, MICROPY_FLOAT_C_FUN(c_name)); \
        return result; \
}

UNARY_FUN_FLOAT_1(acos, acos);
UNARY_FUN_FLOAT_1(asin, asin);
UNARY_FUN_FLOAT_1(atan, atan);

UNARY_FUN_FLOAT_1(sin, sin);
UNARY_FUN_FLOAT_1(cos, cos);
UNARY_FUN_FLOAT_1(tan, tan);

UNARY_FUN_FLOAT_1(acosh, acosh);
UNARY_FUN_FLOAT_1(asinh, asinh);
UNARY_FUN_FLOAT_1(atanh, atanh);

UNARY_FUN_FLOAT_1(sinh, sinh);
UNARY_FUN_FLOAT_1(cosh, cosh);
UNARY_FUN_FLOAT_1(tanh, tanh);

UNARY_FUN_FLOAT_1(ceil, ceil);
UNARY_FUN_FLOAT_1(floor, floor);

UNARY_FUN_FLOAT_1(erf, erf);
UNARY_FUN_FLOAT_1(erfc, erfc);

UNARY_FUN_FLOAT_1(exp, exp);
UNARY_FUN_FLOAT_1(expm1, expm1);

UNARY_FUN_FLOAT_1(gamma, tgamma);
UNARY_FUN_FLOAT_1(lgamma, lgamma);

UNARY_FUN_FLOAT_1(log, log);
UNARY_FUN_FLOAT_1(log10, log10);
UNARY_FUN_FLOAT_1(log2, log2);

UNARY_FUN_FLOAT_1(sqrt, sqrt);
