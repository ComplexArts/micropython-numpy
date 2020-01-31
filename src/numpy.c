#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "py/runtime.h"
#include "py/binary.h"
#include "py/obj.h"
#include "py/objarray.h"

#include "ndarray.h"
#include "array.h"
#include "array2d.h"
#include "unary.h"
#include "binary.h"

#define NUMPY_VERSION 0.100
#define NUMPY_MAJOR 0
#define NUMPY_MINOR 1
#define NUMPY_RELEASE 0

typedef struct _mp_obj_float_t {
    mp_obj_base_t base;
    mp_float_t value;
} mp_obj_float_t;

mp_obj_float_t numpy_version = {{&mp_type_float}, NUMPY_VERSION};

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_shape_obj, ndarray_shape);
MP_DEFINE_CONST_FUN_OBJ_1(ndarray_size_obj, ndarray_size);
MP_DEFINE_CONST_FUN_OBJ_1(ndarray_ndim_obj, ndarray_ndim);

MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_flatten_obj, 1, ndarray_flatten);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_copy_obj, 1, ndarray_copy);
MP_DEFINE_CONST_FUN_OBJ_1(ndarray_asbytearray_obj, ndarray_asbytearray);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_astype_obj, 2, ndarray_astype);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_reshape_obj, 2, ndarray_reshape);

MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_all_obj, 1, ndarray_all);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_any_obj, 1, ndarray_any);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_min_obj, 1, ndarray_min);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_max_obj, 1, ndarray_max);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_sum_obj, 1, ndarray_sum);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_prod_obj, 1, ndarray_prod);

MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_zeros_obj, 1, ndarray_zeros);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_ones_obj, 1, ndarray_ones);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_full_obj, 1, ndarray_full);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_empty_obj, 1, ndarray_empty);

MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_eye_obj, 1, ndarray_eye);

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_fabs_obj, ndarray_fabs);
MP_DEFINE_CONST_FUN_OBJ_2(ndarray_issubsctype_obj, ndarray_issubsctype);
MP_DEFINE_CONST_FUN_OBJ_2(ndarray_array_equal_obj, ndarray_array_equal);

MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_flip_obj, 1, ndarray_flip);
MP_DEFINE_CONST_FUN_OBJ_1(ndarray_flipud_obj, ndarray_flipud);
MP_DEFINE_CONST_FUN_OBJ_1(ndarray_fliplr_obj, ndarray_fliplr);

MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_concatenate_obj, 1, ndarray_concatenate);
MP_DEFINE_CONST_FUN_OBJ_1(ndarray_vstack_obj, ndarray_vstack);
MP_DEFINE_CONST_FUN_OBJ_1(ndarray_hstack_obj, ndarray_hstack);

MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_minimum_obj, 2, binary_minimum);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_maximum_obj, 2, binary_maximum);

MP_DEFINE_CONST_FUN_OBJ_1(array_transpose_obj, array_transpose);
MP_DEFINE_CONST_FUN_OBJ_2(array_dot_obj, array_dot);

MP_DEFINE_CONST_FUN_OBJ_1(array2d_lu_obj, array2d_lu);
MP_DEFINE_CONST_FUN_OBJ_1(array2d_lu_det_obj, array2d_lu_det);
MP_DEFINE_CONST_FUN_OBJ_2(array2d_lu_solve_obj, array2d_lu_solve);
MP_DEFINE_CONST_FUN_OBJ_1(array2d_lu_inv_obj, array2d_lu_inv);

MP_DEFINE_CONST_FUN_OBJ_1(array2d_qr_obj, array2d_qr);
MP_DEFINE_CONST_FUN_OBJ_2(array2d_qr_solve_obj, array2d_qr_solve);

MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_acos_obj, unary_fun_acos);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_asin_obj, unary_fun_asin);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_atan_obj, unary_fun_atan);

MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_sin_obj, unary_fun_sin);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_cos_obj, unary_fun_cos);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_tan_obj, unary_fun_tan);

MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_acosh_obj, unary_fun_acosh);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_asinh_obj, unary_fun_asinh);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_atanh_obj, unary_fun_atanh);

MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_sinh_obj, unary_fun_sinh);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_cosh_obj, unary_fun_cosh);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_tanh_obj, unary_fun_tanh);

MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_ceil_obj, unary_fun_ceil);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_floor_obj, unary_fun_floor);

MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_erf_obj, unary_fun_erf);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_erfc_obj, unary_fun_erfc);

MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_exp_obj, unary_fun_exp);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_expm1_obj, unary_fun_expm1);

MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_gamma_obj, unary_fun_gamma);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_lgamma_obj, unary_fun_lgamma);

MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_log_obj, unary_fun_log);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_log10_obj, unary_fun_log10);
MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_log2_obj, unary_fun_log2);

MP_DEFINE_CONST_FUN_OBJ_1(unary_fun_sqrt_obj, unary_fun_sqrt);

STATIC const mp_rom_map_elem_t numpy_ndarray_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_shape), MP_ROM_PTR(&ndarray_shape_obj) },
    { MP_ROM_QSTR(MP_QSTR_size), MP_ROM_PTR(&ndarray_size_obj) },
    { MP_ROM_QSTR(MP_QSTR_ndim), MP_ROM_PTR(&ndarray_ndim_obj) },
    { MP_ROM_QSTR(MP_QSTR_flatten), MP_ROM_PTR(&ndarray_flatten_obj) },
    { MP_ROM_QSTR(MP_QSTR_copy), MP_ROM_PTR(&ndarray_copy_obj) },
    { MP_ROM_QSTR(MP_QSTR_astype), MP_ROM_PTR(&ndarray_astype_obj) },
    { MP_ROM_QSTR(MP_QSTR_asbytearray), MP_ROM_PTR(&ndarray_asbytearray_obj) },
    { MP_ROM_QSTR(MP_QSTR_reshape), MP_ROM_PTR(&ndarray_reshape_obj) },
    { MP_ROM_QSTR(MP_QSTR_all), MP_ROM_PTR(&ndarray_all_obj) },
    { MP_ROM_QSTR(MP_QSTR_any), MP_ROM_PTR(&ndarray_any_obj) },
    { MP_ROM_QSTR(MP_QSTR_max), MP_ROM_PTR(&ndarray_max_obj) },
    { MP_ROM_QSTR(MP_QSTR_min), MP_ROM_PTR(&ndarray_min_obj) },
    { MP_ROM_QSTR(MP_QSTR_sum), MP_ROM_PTR(&ndarray_sum_obj) },
    { MP_ROM_QSTR(MP_QSTR_prod), MP_ROM_PTR(&ndarray_prod_obj) },
    { MP_ROM_QSTR(MP_QSTR_transpose), MP_ROM_PTR(&array_transpose_obj) },
    { MP_ROM_QSTR(MP_QSTR_dot), MP_ROM_PTR(&array_dot_obj) },
    //{ MP_ROM_QSTR(MP_QSTR_sort), MP_ROM_PTR(&numerical_sort_inplace_obj) },
};

STATIC MP_DEFINE_CONST_DICT(numpy_ndarray_locals_dict, numpy_ndarray_locals_dict_table);

const mp_obj_type_t ndarray_type = {
    { &mp_type_type },
    .name = MP_QSTR_ndarray,
    .print = ndarray_print,
    .make_new = ndarray_make_new,
    .subscr = ndarray_subscr,
    .getiter = ndarray_getiter,
    .unary_op = unary_op,
    .binary_op = binary_op,
    .locals_dict = (mp_obj_dict_t*)&numpy_ndarray_locals_dict,
};

STATIC const mp_map_elem_t numpy_globals_table[] = {
    { MP_OBJ_NEW_QSTR(MP_QSTR___name__), MP_OBJ_NEW_QSTR(MP_QSTR_numpy) },
    { MP_ROM_QSTR(MP_QSTR___version__), MP_ROM_PTR(&numpy_version) },

    { MP_OBJ_NEW_QSTR(MP_QSTR_ndarray), (mp_obj_t)&ndarray_type },
    { MP_OBJ_NEW_QSTR(MP_QSTR_array), (mp_obj_t)&ndarray_type },

    { MP_ROM_QSTR(MP_QSTR_shape), (mp_obj_t)&ndarray_shape_obj },
    { MP_ROM_QSTR(MP_QSTR_size), (mp_obj_t)&ndarray_size_obj },
    { MP_ROM_QSTR(MP_QSTR_ndim), (mp_obj_t)&ndarray_ndim_obj },

    { MP_ROM_QSTR(MP_QSTR_all), (mp_obj_t)&ndarray_all_obj },
    { MP_ROM_QSTR(MP_QSTR_any), (mp_obj_t)&ndarray_any_obj },
    { MP_ROM_QSTR(MP_QSTR_max), (mp_obj_t)&ndarray_max_obj },
    { MP_ROM_QSTR(MP_QSTR_min), (mp_obj_t)&ndarray_min_obj },
    { MP_ROM_QSTR(MP_QSTR_sum), (mp_obj_t)&ndarray_sum_obj },
    { MP_ROM_QSTR(MP_QSTR_prod), (mp_obj_t)&ndarray_prod_obj },

    { MP_ROM_QSTR(MP_QSTR_dot), (mp_obj_t)&array_dot_obj },
    { MP_ROM_QSTR(MP_QSTR_lu), (mp_obj_t)&array2d_lu_obj },
    { MP_ROM_QSTR(MP_QSTR_det), (mp_obj_t)&array2d_lu_det_obj },
    { MP_ROM_QSTR(MP_QSTR_inv), (mp_obj_t)&array2d_lu_inv_obj },
    { MP_ROM_QSTR(MP_QSTR_solve), (mp_obj_t)&array2d_lu_solve_obj },

    { MP_ROM_QSTR(MP_QSTR_qr), (mp_obj_t)&array2d_qr_obj },
    { MP_ROM_QSTR(MP_QSTR_qr_solve), (mp_obj_t)&array2d_qr_solve_obj },
    { MP_ROM_QSTR(MP_QSTR_lstsq), (mp_obj_t)&array2d_qr_solve_obj },

    { MP_ROM_QSTR(MP_QSTR_concatenate), (mp_obj_t)&ndarray_concatenate_obj },
    { MP_ROM_QSTR(MP_QSTR_vstack), (mp_obj_t)&ndarray_vstack_obj },
    { MP_ROM_QSTR(MP_QSTR_hstack), (mp_obj_t)&ndarray_hstack_obj },

    { MP_ROM_QSTR(MP_QSTR_empty), (mp_obj_t)&ndarray_empty_obj },
    { MP_ROM_QSTR(MP_QSTR_full), (mp_obj_t)&ndarray_full_obj },
    { MP_ROM_QSTR(MP_QSTR_zeros), (mp_obj_t)&ndarray_zeros_obj },
    { MP_ROM_QSTR(MP_QSTR_ones), (mp_obj_t)&ndarray_ones_obj },
    { MP_ROM_QSTR(MP_QSTR_eye), (mp_obj_t)&ndarray_eye_obj },

    { MP_ROM_QSTR(MP_QSTR_fabs), (mp_obj_t)&ndarray_fabs_obj },
    { MP_ROM_QSTR(MP_QSTR_issubsctype), (mp_obj_t)&ndarray_issubsctype_obj },
    { MP_ROM_QSTR(MP_QSTR_array_equal), (mp_obj_t)&ndarray_array_equal_obj },

    { MP_ROM_QSTR(MP_QSTR_flip), (mp_obj_t)&ndarray_flip_obj },
    { MP_ROM_QSTR(MP_QSTR_fliplr), (mp_obj_t)&ndarray_fliplr_obj },
    { MP_ROM_QSTR(MP_QSTR_flipud), (mp_obj_t)&ndarray_flipud_obj },

    { MP_ROM_QSTR(MP_QSTR_minimum), (mp_obj_t)&ndarray_minimum_obj },
    { MP_ROM_QSTR(MP_QSTR_maximum), (mp_obj_t)&ndarray_maximum_obj },

    { MP_OBJ_NEW_QSTR(MP_QSTR_acos), (mp_obj_t)&unary_fun_acos_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_asin), (mp_obj_t)&unary_fun_asin_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_atan), (mp_obj_t)&unary_fun_atan_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_sin), (mp_obj_t)&unary_fun_sin_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_cos), (mp_obj_t)&unary_fun_cos_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_tan), (mp_obj_t)&unary_fun_tan_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_acosh), (mp_obj_t)&unary_fun_acosh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_asinh), (mp_obj_t)&unary_fun_asinh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_atanh), (mp_obj_t)&unary_fun_atanh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_sinh), (mp_obj_t)&unary_fun_sinh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_cosh), (mp_obj_t)&unary_fun_cosh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_tanh), (mp_obj_t)&unary_fun_tanh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_ceil), (mp_obj_t)&unary_fun_ceil_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_floor), (mp_obj_t)&unary_fun_floor_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_erf), (mp_obj_t)&unary_fun_erf_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_erfc), (mp_obj_t)&unary_fun_erfc_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_exp), (mp_obj_t)&unary_fun_exp_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_expm1), (mp_obj_t)&unary_fun_expm1_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_gamma), (mp_obj_t)&unary_fun_gamma_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_lgamma), (mp_obj_t)&unary_fun_lgamma_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_log), (mp_obj_t)&unary_fun_log_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_log10), (mp_obj_t)&unary_fun_log10_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_log2), (mp_obj_t)&unary_fun_log2_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_sqrt), (mp_obj_t)&unary_fun_sqrt_obj },
    // class constants
    { MP_ROM_QSTR(MP_QSTR_bool_), MP_ROM_INT(NDARRAY_BOOL) },
    { MP_ROM_QSTR(MP_QSTR_uint8), MP_ROM_INT(NDARRAY_UINT8) },
    { MP_ROM_QSTR(MP_QSTR_int8), MP_ROM_INT(NDARRAY_INT8) },
    { MP_ROM_QSTR(MP_QSTR_uint16), MP_ROM_INT(NDARRAY_UINT16) },
    { MP_ROM_QSTR(MP_QSTR_int16), MP_ROM_INT(NDARRAY_INT16) },
    { MP_ROM_QSTR(MP_QSTR_float_), MP_ROM_INT(NDARRAY_FLOAT) },
};

STATIC MP_DEFINE_CONST_DICT (
    mp_module_numpy_globals,
    numpy_globals_table
);

const mp_obj_module_t numpy_user_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&mp_module_numpy_globals,
};

MP_REGISTER_MODULE(MP_QSTR_numpy, numpy_user_cmodule, MODULE_NUMPY_ENABLED);
