#include "scalar.h"

bool
ndarray_is_scalar(mp_obj_t arg) {

    if (arg == mp_const_false || arg == mp_const_true ||
        mp_obj_is_small_int(arg) || mp_obj_is_type(arg, &mp_type_int) ||
        mp_obj_is_float(arg))
        return true;
    else
        return false;
}

ndarray_scalar_t
ndarray_get_scalar(mp_obj_t arg) {

    // follows mp_obj_get_int
    ndarray_scalar_t scalar = { .typecode=' ', .base_type.int_ = 0 };

    if (arg == mp_const_false) {
        scalar.base_type.uint8_ = 0;
        scalar.typecode = NDARRAY_BOOL;
    } else if (arg == mp_const_true) {
        scalar.base_type.uint8_ = 1;
        scalar.typecode = NDARRAY_BOOL;
    } else if (mp_obj_is_small_int(arg)) {
        scalar.base_type.int16_ = MP_OBJ_SMALL_INT_VALUE(arg);
        scalar.typecode = NDARRAY_INT16;
    } else if (mp_obj_is_type(arg, &mp_type_int)) {
        scalar.base_type.int_ =  mp_obj_int_get_checked(arg);
        scalar.typecode = NDARRAY_INT16;
    } else { // if (mp_obj_is_float(arg)) {
        scalar.base_type.float_ = mp_obj_get_float(arg);
        scalar.typecode = NDARRAY_FLOAT;
    }

    return scalar;
}
