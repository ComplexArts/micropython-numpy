#ifndef _SCALAR_H_
#define _SCALAR_H_

#include <ndarray.h>

typedef union {
    mp_uint_t  uint_;
    mp_int_t   int_;
    uint8_t    uint8_;
    int8_t     int8_;
    uint16_t   uint16_;
    int16_t    int16_;
    uint32_t   uint32_;
    int32_t    int32_;
    mp_float_t float_;
} ndarray_base_type_t;

typedef struct {
    ndarray_base_type_t base_type;
    char typecode;
} ndarray_scalar_t;

bool ndarray_is_scalar(mp_obj_t arg);
ndarray_scalar_t ndarray_get_scalar(mp_obj_t arg);

#endif
