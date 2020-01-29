#ifndef _SLICE_H_
#define _SLICE_H_

#include <ndarray.h>

size_t ndarray_slice_length(mp_bound_slice_t *slice);
void ndarray_slice_slice(mp_bound_slice_t* sliced_slice, mp_bound_slice_t *slice0, mp_bound_slice_t *slice1);
void ndarray_slice_flatten(size_t level, size_t j, size_t dims, mp_bound_slice_t *slice, size_t *shape, size_t *array_shape);

#endif
