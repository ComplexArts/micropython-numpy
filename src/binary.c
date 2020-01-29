#include "py/runtime.h"

#include "binary.h"
#include "scalar.h"
#include "array.h"

// Binary operations

#define BINARY_ADD(x, y, z) z++ = (x + y)
#define BINARY_SUBTRACT(x, y, z) z++ = (x - y)
#define BINARY_MULTIPLY(x, y, z) z++ = (x * y)
#define BINARY_DIVIDE(x, y, z) z++ = (x / y)
#define BINARY_EQUAL(x, y, z) z++ = (x == y)
#define BINARY_NOT_EQUAL(x, y, z) z++ = (x != y)
#define BINARY_LESS(x, y, z) z++ = (x < y)
#define BINARY_LESS_EQUAL(x, y, z) z++ = (x <= y)
#define BINARY_MORE(x, y, z) z++ = (x > y)
#define BINARY_MORE_EQUAL(x, y, z) z++ = (x >= y)

#define BINARY_AND(x, y, z) z++ = (x && y)
#define BINARY_OR(x, y, z) z++ = (x || y)
#define BINARY_MAX(x, y, z) z++ = (x >= y ? x : y)
#define BINARY_MIN(x, y, z) z++ = (x <= y ? x : y)

#define BINARY_DOT(x, y, z) z += (x * y)

#define BINARY_OP_INNER_LOOP(ptr_result, ptr1, step1, ptr2, step2, count, op) do {\
    for (size_t i = 0; i < count; i++) { \
        op(*ptr1, *ptr2, *ptr_result); \
        ptr1 += step1; \
        ptr2 += step2; \
    } \
} while(0)

#define NDARRAY_TYPECAST_RHS(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op) do { \
    if (rhs_typecode == NDARRAY_BOOL || rhs_typecode == NDARRAY_UINT8) { \
        uint8_t *(rhs_ptr_typed) = (uint8_t *) (rhs_ptr); \
        BINARY_OP_INNER_LOOP(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_ptr_typed, rhs_step, count, op); \
    } else if (rhs_typecode == NDARRAY_INT8) { \
        int8_t *(rhs_ptr_typed) = (int8_t *) (rhs_ptr); \
        BINARY_OP_INNER_LOOP(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_ptr_typed, rhs_step, count, op); \
    } else if (rhs_typecode == NDARRAY_UINT16) { \
        uint16_t *(rhs_ptr_typed) = (uint16_t *) (rhs_ptr); \
        BINARY_OP_INNER_LOOP(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_ptr_typed, rhs_step, count, op); \
    } else if (rhs_typecode == NDARRAY_INT16) { \
        int16_t *(rhs_ptr_typed) = (int16_t *) (rhs_ptr); \
        BINARY_OP_INNER_LOOP(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_ptr_typed, rhs_step, count, op); \
    } else { \
        mp_float_t *(rhs_ptr_typed) = (mp_float_t *) (rhs_ptr); \
        BINARY_OP_INNER_LOOP(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_ptr_typed, rhs_step, count, op); \
    } \
} while(0)

#define NDARRAY_TYPECAST_LHS_RHS(self_ptr_typed, lhs_typecode, lhs_ptr, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op) do { \
    if (lhs_typecode == NDARRAY_BOOL || lhs_typecode == NDARRAY_UINT8) { \
        uint8_t *(lhs_ptr_typed) = (uint8_t *) (lhs_ptr); \
        NDARRAY_TYPECAST_RHS(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op); \
    } else if (lhs_typecode == NDARRAY_INT8) { \
        int8_t *(lhs_ptr_typed) = (int8_t *) (lhs_ptr); \
        NDARRAY_TYPECAST_RHS(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op); \
    } else if (lhs_typecode == NDARRAY_UINT16) { \
        uint16_t *(lhs_ptr_typed) = (uint16_t *) (lhs_ptr); \
        NDARRAY_TYPECAST_RHS(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op); \
    } else if (lhs_typecode == NDARRAY_INT16) { \
        int16_t *(lhs_ptr_typed) = (int16_t *) (lhs_ptr); \
        NDARRAY_TYPECAST_RHS(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op); \
    } else { \
        mp_float_t *(lhs_ptr_typed) = (mp_float_t *) (lhs_ptr); \
        NDARRAY_TYPECAST_RHS(self_ptr_typed, lhs_ptr_typed, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op); \
    } \
} while(0)

#define NDARRAY_TYPECAST(self_typecode, self_ptr, lhs_typecode, lhs_ptr, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op) do { \
    if (self_typecode == NDARRAY_BOOL || self_typecode == NDARRAY_UINT8) { \
        uint8_t *(self_ptr_typed) = (uint8_t *) (self_ptr); \
        NDARRAY_TYPECAST_LHS_RHS(self_ptr_typed, lhs_typecode, lhs_ptr, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op); \
    } else if (self_typecode == NDARRAY_INT8) { \
        int8_t *(self_ptr_typed) = (int8_t *) (self_ptr); \
        NDARRAY_TYPECAST_LHS_RHS(self_ptr_typed, lhs_typecode, lhs_ptr, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op); \
    } else if (self_typecode == NDARRAY_UINT16) { \
        uint16_t *(self_ptr_typed) = (uint16_t *) (self_ptr); \
        NDARRAY_TYPECAST_LHS_RHS(self_ptr_typed, lhs_typecode, lhs_ptr, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op); \
    } else if (self_typecode == NDARRAY_INT16) { \
        int16_t *(self_ptr_typed) = (int16_t *) (self_ptr); \
        NDARRAY_TYPECAST_LHS_RHS(self_ptr_typed, lhs_typecode, lhs_ptr, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op); \
    } else { \
        mp_float_t *(self_ptr_typed) = (mp_float_t *) (self_ptr); \
        NDARRAY_TYPECAST_LHS_RHS(self_ptr_typed, lhs_typecode, lhs_ptr, lhs_step, rhs_typecode, rhs_ptr, rhs_step, count, op); \
    } \
} while(0)


#define BINARY_OP_FUN(name, bin_op) \
void \
binary_op_## name(char self_typecode, void *self_ptr, \
                  char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step, \
                  char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count, \
                  mp_binary_op_t op) { \
    NDARRAY_TYPECAST(self_typecode, self_ptr, \
                     lhs_typecode, lhs_ptr, lhs_step, \
                     rhs_typecode, rhs_ptr, rhs_step, count, \
                     bin_op); \
}

BINARY_OP_FUN(add, BINARY_ADD);
BINARY_OP_FUN(subtract, BINARY_SUBTRACT);
BINARY_OP_FUN(multiply, BINARY_MULTIPLY);
BINARY_OP_FUN(divide, BINARY_DIVIDE);
BINARY_OP_FUN(equal, BINARY_EQUAL);
BINARY_OP_FUN(not_equal, BINARY_NOT_EQUAL);
BINARY_OP_FUN(less, BINARY_LESS);
BINARY_OP_FUN(less_equal, BINARY_LESS_EQUAL);
BINARY_OP_FUN(more, BINARY_MORE);
BINARY_OP_FUN(more_equal, BINARY_MORE_EQUAL);
BINARY_OP_FUN(and, BINARY_AND);
BINARY_OP_FUN(or, BINARY_OR);
BINARY_OP_FUN(max, BINARY_MAX);
BINARY_OP_FUN(min, BINARY_MIN);
BINARY_OP_FUN(dot, BINARY_DOT);

char
binary_upcast(char lhs_typecode, char rhs_typecode) {

    // Applies the following upcast rules:
    //
    // type  op type   => type
    //
    // bool|uint8  op uintY        => uintY
    // bool|uint8  op intY         => intY
    // uintX       op bool|uint8   => uintX
    // intX        op bool|uint8   => intX
    //
    // uintX op uintY  => uint(X>Y)
    // uintX op intY   => int(X>Y)
    // intX  op uintY  => int(X>Y)
    // intX  op intY   => int(X>Y)
    //
    // intX|bool op float  => float

    char typecode;
    if (lhs_typecode == rhs_typecode)
        // same types
        typecode = lhs_typecode;
    else {
        // upcast
        if (lhs_typecode == NDARRAY_FLOAT || rhs_typecode == NDARRAY_FLOAT)
            typecode = NDARRAY_FLOAT;
        else if (lhs_typecode == NDARRAY_BOOL || lhs_typecode == NDARRAY_UINT8)
            typecode = rhs_typecode;
        else if (rhs_typecode == NDARRAY_BOOL || rhs_typecode == NDARRAY_UINT8)
            typecode = lhs_typecode;
        else if (lhs_typecode == NDARRAY_UINT16 && rhs_typecode == NDARRAY_INT16)
            typecode = NDARRAY_INT16;
        else if (lhs_typecode == NDARRAY_INT16 && rhs_typecode == NDARRAY_UINT16)
            typecode = NDARRAY_INT16;
        else
            typecode = NDARRAY_FLOAT;
    }

    return typecode;
}


mp_obj_t
binary_create_op_result(mp_obj_t lhs, mp_obj_t rhs, mp_binary_op_t op) {

    // create result of scalar binary operator

    char typecode, lhs_typecode, rhs_typecode;
    size_t dims;
    size_t *shape;

    if (mp_obj_is_type(lhs, &ndarray_type)) {

        // left-hand side is an ndarray
        ndarray_obj_t *lhs_ndarray = MP_OBJ_TO_PTR(lhs);
        lhs_typecode = lhs_ndarray->typecode;

        if (mp_obj_is_type(rhs, &ndarray_type)) {

            // right-hand side is an ndarray as well

            // get ndarrays
            ndarray_obj_t *rhs_ndarray = MP_OBJ_TO_PTR(rhs);
            rhs_typecode = lhs_ndarray->typecode;

            // check dimensions
            if (lhs_ndarray->dims != rhs_ndarray->dims) {
                mp_raise_ValueError("operands must have the same dimensions");
                return mp_const_none;
            }

            // check shape
            for (size_t i = 0; i < lhs_ndarray->dims; i++) {
                if (lhs_ndarray->shape[i] != rhs_ndarray->shape[i]) {
                    mp_raise_ValueError("operands must have the same shape");
                    return mp_const_none;
                }
            }

        } else {

            ndarray_scalar_t scalar = ndarray_get_scalar(rhs);
            rhs_typecode = scalar.typecode;

        }

        // result array shape is the same as lhs
        dims = lhs_ndarray->dims;
        shape = m_new(size_t, dims);
        for (size_t i = 0; i < dims; i++)
            shape[i] = lhs_ndarray->shape[i];

    } else if (mp_obj_is_type(rhs, &ndarray_type)) {

        // left-hand side is scalar
        ndarray_scalar_t scalar = ndarray_get_scalar(lhs);
        lhs_typecode = scalar.typecode;

        // right-hand side is array
        ndarray_obj_t *rhs_ndarray = MP_OBJ_TO_PTR(rhs);
        rhs_typecode = rhs_ndarray->typecode;

        // result array shape is the same as rhs
        dims = rhs_ndarray->dims;
        shape = m_new(size_t, dims);
        for (size_t i = 0; i < dims; i++)
            shape[i] = rhs_ndarray->shape[i];

    } else {

        mp_raise_ValueError("at least one of the operands must be an ndarray");
        return mp_const_none;

    }

    // boolean operations?
    if (op == MP_BINARY_OP_EQUAL || op == MP_BINARY_OP_NOT_EQUAL || op == MP_BINARY_OP_LESS ||
        op == MP_BINARY_OP_LESS_EQUAL || op == MP_BINARY_OP_MORE || op == MP_BINARY_OP_MORE_EQUAL) {
        typecode = NDARRAY_BOOL;
    } else
        typecode = binary_upcast(lhs_typecode, rhs_typecode);

#ifdef NDARRAY_DEBUG
    printf("BINARY OP = %d\n", op);
    printf("typecode = %c\n", typecode);
    printf("dims = %ld\n", dims);
    printf("shape = ");
    for (int i = 0; i < dims; i++)
        printf("%ld, ", shape[i]);
    printf("\n");
#endif

    // create result array
    return MP_OBJ_FROM_PTR(array_create_new(dims, shape, typecode, false));

}

void
binary_op_recursive(size_t level,
                    ndarray_obj_t *self_ndarray, size_t self_offset,
                    mp_obj_t lhs, size_t lhs_offset,
                    mp_obj_t rhs, size_t rhs_offset,
                    mp_binary_op_t op,
                    binary_op_fun_t op_fun) {

#ifdef NDARRAY_DEBUG
    printf("BINARY_OP_RECURSIVE level = %ld, self_offset = %ld, lhs_offset = %ld, rhs_offset = %ld, op = %d\n",
           level, self_offset, lhs_offset, rhs_offset, op);
#endif

    // get current slices and shapes
    mp_uint_t self_depth = 1, lhs_depth = 1, rhs_depth = 1;
    mp_bound_slice_t lhs_slice = { .start=0, .step=1, .stop=0 }, rhs_slice = { .start=0, .step=1, .stop=0 };
    mp_bound_slice_t self_slice = self_ndarray->slice[level];
    size_t n = self_ndarray->shape[level];

    // calculate remaining depth
    for (mp_uint_t i = level + 1; i < self_ndarray->dims; i++) {
        self_depth *= self_ndarray->array_shape[i];
    }

    ndarray_obj_t *lhs_ndarray = NULL;
    if (mp_obj_is_type(lhs, &ndarray_type)) {

        // left-hand side is array
        lhs_ndarray = MP_OBJ_TO_PTR(lhs);
        lhs_slice = lhs_ndarray->slice[level];
        n = lhs_ndarray->shape[level];

        // calculate remaining depth
        for (mp_uint_t i = level + 1; i < lhs_ndarray->dims; i++) {
            lhs_depth *= lhs_ndarray->array_shape[i];
        }

    }

    ndarray_obj_t *rhs_ndarray = NULL;
    if (mp_obj_is_type(rhs, &ndarray_type)) {

        // right-hand side is array
        rhs_ndarray = MP_OBJ_TO_PTR(rhs);
        rhs_slice = rhs_ndarray->slice[level];
        n = rhs_ndarray->shape[level];

        // calculate remaining depth
        for (mp_uint_t i = level + 1; i < rhs_ndarray->dims; i++) {
            rhs_depth *= rhs_ndarray->array_shape[i];
        }

    }

    // increment level
    level++;

    mp_uint_t self_index = self_slice.start, lhs_index = lhs_slice.start, rhs_index = rhs_slice.start;
    if (level < self_ndarray->dims) {

        // upper level, recurse
        for (mp_uint_t i = 0; i < n; i++) {
            binary_op_recursive(level,
                                self_ndarray, self_offset + self_index * self_depth,
                                lhs, lhs_offset + lhs_index * lhs_depth,
                                rhs, rhs_offset + rhs_index * rhs_depth,
                                op, op_fun);
            self_index += self_slice.step;
            lhs_index += lhs_slice.step;
            rhs_index += rhs_slice.step;
        }

    } else {

        // last level, operate
        void *lhs_ptr, *rhs_ptr,
                *self_ptr = array_get_ptr_array(self_ndarray->typecode, self_ndarray->array->items,
                                                  self_offset + self_index * self_depth);

        ndarray_scalar_t lhs_scalar;
        char lhs_typecode;
        if (mp_obj_is_type(lhs, &ndarray_type)) {
            lhs_typecode = lhs_ndarray->typecode;
            lhs_ptr = array_get_ptr_array(lhs_typecode, lhs_ndarray->array->items,
                                            lhs_offset + lhs_index * lhs_depth);
        } else {

            lhs_scalar = ndarray_get_scalar(lhs);
            lhs_typecode = lhs_scalar.typecode;
            lhs_ptr = &lhs_scalar.base_type;
            lhs_slice.step = 0;

        }

        ndarray_scalar_t rhs_scalar;
        char rhs_typecode;
        if (mp_obj_is_type(rhs, &ndarray_type)) {
            rhs_typecode = rhs_ndarray->typecode;
            rhs_ptr = array_get_ptr_array(rhs_typecode, rhs_ndarray->array->items,
                                            rhs_offset + rhs_index * rhs_depth);
        } else {

            rhs_scalar = ndarray_get_scalar(rhs);
            rhs_typecode = rhs_scalar.typecode;
            rhs_ptr = &rhs_scalar.base_type;
            rhs_slice.step = 0;

        }

        (*op_fun)(self_ndarray->typecode, self_ptr,
                  lhs_typecode, lhs_ptr, lhs_slice.step,
                  rhs_typecode, rhs_ptr, rhs_slice.step, n, op);

    }

}

mp_obj_t
binary_op(mp_binary_op_t op, mp_obj_t lhs, mp_obj_t rhs) {

    // reverse ops
    if (op == MP_BINARY_OP_REVERSE_ADD)
        return binary_op(MP_BINARY_OP_ADD, rhs, lhs);
    else if (op == MP_BINARY_OP_REVERSE_MULTIPLY)
        return binary_op(MP_BINARY_OP_MULTIPLY, rhs, lhs);
    else if (op == MP_BINARY_OP_REVERSE_SUBTRACT)
        return binary_op(MP_BINARY_OP_SUBTRACT, rhs, lhs);
    else if (op == MP_BINARY_OP_REVERSE_TRUE_DIVIDE)
        return binary_op(MP_BINARY_OP_TRUE_DIVIDE, rhs, lhs);

    // check for supported ops
    binary_op_fun_t op_fun;
    if (op == MP_BINARY_OP_ADD)
        op_fun = binary_op_add;
    else if (op == MP_BINARY_OP_MULTIPLY)
        op_fun = binary_op_multiply;
    else if (op == MP_BINARY_OP_SUBTRACT)
        op_fun = binary_op_subtract;
    else if (op == MP_BINARY_OP_TRUE_DIVIDE)
        op_fun = binary_op_divide;
    else if (op == MP_BINARY_OP_EQUAL)
        op_fun = binary_op_equal;
    else if (op == MP_BINARY_OP_NOT_EQUAL)
        op_fun = binary_op_not_equal;
    else if (op == MP_BINARY_OP_LESS)
        op_fun = binary_op_less;
    else if (op == MP_BINARY_OP_MORE)
        op_fun = binary_op_more;
    else if (op == MP_BINARY_OP_LESS_EQUAL)
        op_fun = binary_op_less_equal;
    else if (op == MP_BINARY_OP_MORE_EQUAL)
        op_fun = binary_op_more_equal;
    else if (op == MP_BINARY_OP_AND)
        op_fun = binary_op_and;
    else if (op == MP_BINARY_OP_OR)
        op_fun = binary_op_or;
    else
        return MP_OBJ_NULL;

    // check dimensions, upcast and create return object
    mp_obj_t result = binary_create_op_result(lhs, rhs, op);

    // recurse to operate
    ndarray_obj_t *result_ndarray = MP_OBJ_FROM_PTR(result);
    binary_op_recursive(0, result_ndarray, 0, lhs, 0, rhs, 0, op, op_fun);
#ifdef NDARRAY_DEBUG
    mp_obj_print_helper(&mp_sys_stdout_print, result, PRINT_REPR);
    printf("\n");
#endif
    return result;

}

mp_obj_t
binary_op_helper(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args, mp_binary_op_t op) {
    static const mp_arg_t allowed_args[] = {
            { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = mp_const_none} },
            { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = mp_const_none} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    mp_obj_t lhs = args[0].u_obj;
    mp_obj_t rhs = args[1].u_obj;

    return binary_create_op_result(lhs, rhs, op);
}

mp_obj_t
binary_maximum(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {

    // call helper to check dimensions, upcast and create return object
    mp_obj_t result = binary_op_helper(n_args, pos_args, kw_args, NDARRAY_BINARY_OP_MAX);

    // recurse to operate
    ndarray_obj_t *result_ndarray = MP_OBJ_FROM_PTR(result);
    binary_op_recursive(0, result_ndarray, 0, pos_args[0], 0, pos_args[1], 0, NDARRAY_BINARY_OP_MAX, binary_op_max);

    return result;

}

mp_obj_t
binary_minimum(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {

    // call helper to check dimensions, upcast and create return object
    mp_obj_t result = binary_op_helper(n_args, pos_args, kw_args, NDARRAY_BINARY_OP_MIN);

    // recurse to operate
    ndarray_obj_t *result_ndarray = MP_OBJ_FROM_PTR(result);
    binary_op_recursive(0, result_ndarray, 0, pos_args[0], 0, pos_args[1], 0, NDARRAY_BINARY_OP_MIN, binary_op_min);

    return result;

}
