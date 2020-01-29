#ifndef _ARRAY_H_
#define _ARRAY_H_

#include <ndarray.h>

mp_obj_array_t *array_new(char typecode, size_t n);

size_t array_get_size(char struct_type, char val_type, size_t *palign);
mp_obj_t array_get_val_array(char typecode, void *p, size_t index);
void array_set_val_array(char typecode, void *p, size_t index, mp_obj_t val_in);
void array_set_val_array_from_int(char typecode, void *p, size_t index, mp_int_t val);
void* array_get_ptr_array(char typecode, void *p, size_t index);

// ARRAY NEW

ndarray_obj_t *
array_create_new_view(ndarray_obj_t *in, size_t dims, mp_bound_slice_t *slice, bool deep_copy_slice);

ndarray_obj_t *
array_create_new(size_t dims, size_t* shape, uint8_t typecode, bool initialize);

// ARRAY COPY

void
array_copy_recursive(size_t in_level,
                     ndarray_obj_t *origin, mp_uint_t origin_offset,
                     size_t out_level,
                     ndarray_obj_t *destination, mp_uint_t destination_offset);

mp_obj_t array_copy(mp_obj_t self_in);

// ARRAY ASTYPE

void
array_astype_recursive(size_t level,
                       ndarray_obj_t *origin, mp_uint_t origin_offset,
                       ndarray_obj_t *destination, mp_uint_t destination_offset);

mp_obj_t
array_astype(mp_obj_t self_obj, char dtype, bool copy);

// ARRAY TRANSPOSE
void
array_transpose_recursive(size_t in_level,
                          ndarray_obj_t *in, size_t in_offset,
                          ndarray_obj_t *out, size_t out_offset);

mp_obj_t array_transpose(mp_obj_t self_in);

// ARRAY DOT

void
array_dot_recursive(size_t lhs_level,
                    ndarray_obj_t *lhs, size_t lhs_offset,
                    size_t rhs_level,
                    ndarray_obj_t *rhs, size_t rhs_offset,
                    ndarray_obj_t *out, size_t out_offset);

mp_obj_t
array_dot(mp_obj_t lhs_obj, mp_obj_t rhs_obj);

// ASSIGNE

void
array_assign_array(mp_obj_array_t *array,
                   mp_uint_t start, mp_int_t step, mp_uint_t count,
                   mp_obj_t value);

void
array_assign_recursive(ndarray_obj_t *self,
                       size_t level,
                       mp_uint_t offset,
                       mp_obj_t value);

void
array_assign_array_iterable(mp_obj_array_t *array,
                            mp_uint_t start, mp_int_t step, mp_uint_t count,
                            mp_obj_t iterable);

void
array_assign_recursive_iterable(ndarray_obj_t *self,
                                size_t level,
                                mp_uint_t offset,
                                mp_obj_t value);

// ITERABLES

mp_int_t
array_iterable_dims(mp_obj_t iterable);

mp_int_t
array_iterable_shape(mp_obj_t iterable,
                     size_t level,
                     size_t dims,
                     size_t *shape);

// CONCATENATE

mp_obj_t
array_concatenate(size_t axis,
                  size_t dims,
                  size_t len,
                  mp_obj_t *items);
#endif
