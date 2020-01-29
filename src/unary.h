#ifndef _UNARY_H_
#define _UNARY_H_

#include <ndarray.h>

typedef enum {
    UNARY_OP_ALL,
    UNARY_OP_ANY,
    UNARY_OP_MAX,
    UNARY_OP_MIN,
    UNARY_OP_SUM,
    UNARY_OP_PROD,
    UNARY_OP_GEN
} unary_op_t;

typedef
void (*unary_op_fun_t)(char self_typecode, void *self_ptr,
                       char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
                       char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
                       mp_binary_op_t op);

void
unary_array_op(mp_obj_array_t *array, size_t count,
               mp_unary_op_t op);

mp_obj_t
unary_op(mp_unary_op_t op, mp_obj_t self_in);


mp_obj_t
unary_fun_array(mp_obj_t dest_obj,
                mp_obj_array_t *array, size_t offset, mp_int_t step, size_t count,
                unary_op_t op);

mp_obj_t
unary_fun_recursive(size_t axis, size_t level,
                    ndarray_obj_t *origin, mp_uint_t origin_offset,
                    mp_obj_t destination, mp_uint_t destination_offset,
                    unary_op_t op);

mp_obj_t
unary_fun(unary_op_t op, mp_obj_t self_in, mp_int_t axis);

mp_obj_t unary_fun_acos(mp_obj_t in);
mp_obj_t unary_fun_asin(mp_obj_t in);
mp_obj_t unary_fun_atan(mp_obj_t in);

mp_obj_t unary_fun_sin(mp_obj_t in);
mp_obj_t unary_fun_cos(mp_obj_t in);
mp_obj_t unary_fun_tan(mp_obj_t in);

mp_obj_t unary_fun_acosh(mp_obj_t in);
mp_obj_t unary_fun_asinh(mp_obj_t in);
mp_obj_t unary_fun_atanh(mp_obj_t in);

mp_obj_t unary_fun_sinh(mp_obj_t in);
mp_obj_t unary_fun_cosh(mp_obj_t in);
mp_obj_t unary_fun_tanh(mp_obj_t in);

mp_obj_t unary_fun_ceil(mp_obj_t in);
mp_obj_t unary_fun_floor(mp_obj_t in);

mp_obj_t unary_fun_erf(mp_obj_t in);
mp_obj_t unary_fun_erfc(mp_obj_t in);

mp_obj_t unary_fun_exp(mp_obj_t in);
mp_obj_t unary_fun_expm1(mp_obj_t in);

mp_obj_t unary_fun_gamma(mp_obj_t in);
mp_obj_t unary_fun_lgamma(mp_obj_t in);

mp_obj_t unary_fun_log(mp_obj_t in);
mp_obj_t unary_fun_log10(mp_obj_t in);
mp_obj_t unary_fun_log2(mp_obj_t in);

mp_obj_t unary_fun_sqrt(mp_obj_t in);


#endif