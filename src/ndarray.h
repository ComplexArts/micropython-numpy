#ifndef _NDARRAY_
#define _NDARRAY_

#include "py/objarray.h"
#include "py/binary.h"
#include "py/objstr.h"
#include "py/objlist.h"

#define SWAP(t, a, b) { t tmp = a; a = b; b = tmp; }

#define PRINT_MAX  10

#if MICROPY_FLOAT_IMPL == MICROPY_FLOAT_IMPL_FLOAT
#define FLOAT_TYPECODE 'f'
#elif MICROPY_FLOAT_IMPL == MICROPY_FLOAT_IMPL_DOUBLE
#define FLOAT_TYPECODE 'd'
#endif

//#define NDARRAY_DEBUG

extern const mp_obj_type_t ndarray_type;

typedef enum NDARRAY_TYPE {
    NDARRAY_BOOL = 'o',
    NDARRAY_UINT8 = 'B',
    NDARRAY_INT8 = 'b',
    NDARRAY_UINT16 = 'H', 
    NDARRAY_INT16 = 'h',
    NDARRAY_FLOAT = FLOAT_TYPECODE,
    NDARRAY_UNKNOWN = 'X',
} ndarray_type_t;

typedef struct _ndarray_obj_t {
    mp_obj_base_t base;

    char typecode;
    size_t dims;
    mp_bound_slice_t *slice;
    size_t *shape;
    size_t size;

    // container
    size_t *array_shape;
    size_t array_size;
    mp_obj_array_t *array;
    size_t bytes;
} ndarray_obj_t;

mp_obj_t mp_obj_new_ndarray_iterator(mp_obj_t , size_t , mp_obj_iter_buf_t *);

void ndarray_print_row(const mp_print_t *, mp_obj_array_t *, size_t , size_t, size_t );
void ndarray_print(const mp_print_t *, mp_obj_t , mp_print_kind_t );
void ndarray_print_debug(ndarray_obj_t *self);

void ndarray_assign_elements(mp_obj_array_t *, mp_obj_t , uint8_t , size_t *);

mp_obj_t ndarray_make_new(const mp_obj_type_t *, size_t , size_t , const mp_obj_t *);
mp_obj_t ndarray_subscr(mp_obj_t , mp_obj_t , mp_obj_t );
mp_obj_t ndarray_getiter(mp_obj_t , mp_obj_iter_buf_t *);
mp_obj_t ndarray_unary_op(mp_unary_op_t , mp_obj_t );

mp_obj_t ndarray_shape(mp_obj_t );
mp_obj_t ndarray_size(mp_obj_t self_in);
mp_obj_t ndarray_ndim(mp_obj_t );

mp_obj_t ndarray_rawsize(mp_obj_t );
mp_obj_t ndarray_flatten(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t ndarray_copy(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t ndarray_asbytearray(mp_obj_t );
mp_obj_t ndarray_astype(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t ndarray_reshape(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);

mp_obj_t ndarray_all(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t ndarray_any(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t ndarray_min(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t ndarray_max(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t ndarray_sum(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t ndarray_prod(size_t , const mp_obj_t *, mp_map_t *);

mp_obj_t ndarray_maximum(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);
mp_obj_t ndarray_minimum(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);

mp_obj_t ndarray_full(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);
mp_obj_t ndarray_zeros(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);
mp_obj_t ndarray_ones(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);
mp_obj_t ndarray_empty(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);

mp_obj_t ndarray_eye(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);

mp_obj_t ndarray_concatenate(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);
mp_obj_t ndarray_vstack(mp_obj_t iterable);
mp_obj_t ndarray_hstack(mp_obj_t iterable);

mp_obj_t ndarray_fabs(mp_obj_t lhs_obj);
mp_obj_t ndarray_issubsctype(mp_obj_t lhs_obj, mp_obj_t rhs_obj);
mp_obj_t ndarray_array_equal(mp_obj_t lhs_obj, mp_obj_t rhs_obj);

mp_obj_t ndarray_flip(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);
mp_obj_t ndarray_flipud(mp_obj_t self_obj);
mp_obj_t ndarray_fliplr(mp_obj_t self_obj);

#endif
