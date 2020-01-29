#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "py/runtime.h"
#include "py/binary.h"
#include "py/obj.h"
#include "py/objtuple.h"

#include "ndarray.h"

// SLICE FUNCTIONS

size_t
ndarray_slice_length(mp_bound_slice_t *slice) {
    if (slice->step > 0 && slice->start < slice->stop)
        return 1 + (slice->stop - 1 - slice->start) / slice->step;
    else if (slice->step < 0 && slice->start >= slice->stop)
        return 2 + (slice->start - 1 - slice->stop) / (0 - slice->step);
    else
        return 0;
}

void
ndarray_slice_slice(mp_bound_slice_t* sliced_slice, mp_bound_slice_t *slice0, mp_bound_slice_t *slice1) {
    // apply slice1 to slice0
    //
    // eg: [2:3]

#ifdef NDARRAY_DEBUG
    printf("slice0: start = %ld, stop = %ld, step = %ld\n", slice0->start, slice0->stop, slice0->step);
    printf("slice1: start = %ld, stop = %ld, step = %ld\n", slice1->start, slice1->stop, slice1->step);
#endif

    sliced_slice->start = slice0->start + slice1->start * slice0->step;
    sliced_slice->stop = slice0->start + slice1->stop * slice0->step;
    sliced_slice->step = slice0->step * slice1->step;

#ifdef NDARRAY_DEBUG
    printf("sliced slice: start = %ld, stop = %ld, step = %ld\n", sliced_slice->start, sliced_slice->stop, sliced_slice->step);
#endif
}

void
ndarray_slice_flatten(size_t level, size_t j, size_t dims,
                      mp_bound_slice_t *slice, size_t *shape, size_t *array_shape) {

    // flatten slice with index j
    //
    // level 0:
    // eg: index = j, slice = [0:m:1, 0:n:1],          shape = [m, n],         array_shape = [m,n]
    //                slice = [j*n*s:(j*n*+n)*s:1:1],  shape = [n],            array_shape = [m*n], s = 1
    //
    // eg: index = j, slice = [a:b:c, 0:n:p],          shape = [(b-a)/c, n/p], array_shape = [m,n]
    //                slice = [a+j*c:a+j*c+(n/p)*p:p], shape = [n/p],          array_shape = [m*n]
    //
    // eg: index = j, slice = [a:b:c, 0:v:p, 0:q:1],   shape = [(b-a)/c, v/p, q], array_shape = [m,n,q]
    //                slice = [(a+j*n)*s:(a+j*n+v)*s:p*s, 0:r:1], shape = [v/p, q], array_shape = [m*n,q], s = q
    //
    // level 1:
    // eg: index = j, slice = [0:m:1, 0:n:1],    shape = [m, n],     array_shape = [m,n]
    //                slice = [j:j+m*n:n],       shape = [m],        array_shape = [m*n], s = 1
    //
    // eg: index = j, slice = [0:m:1, 0:n:1, 0:q:1],    shape = [m, n, q],     array_shape = [m,n,q]
    //                slice = [j:j+m*n*s:n*s, 0:q:1],   shape = [m, q],        array_shape = [m, n*q], s = q
    //
    // eg: index = j, slice = [0:m:p, a:b:c, 0:q:1],   shape = [m/p, (b-a)/c, q],   array_shape = [m,n,q]
    //                slice = [(a+j*n):(a+j*n+v)*s:p*s, 0:r:1], shape = [m/p, q], array_shape = [m,n*q], s = q

    if (level < dims - 1) {
        // if not the last level, flatten forward

        // set level slice
        slice[level].start = slice[level+1].start + j * array_shape[level+1];
        slice[level].stop = slice[level+1].stop + j * array_shape[level+1];
        slice[level].step = slice[level+1].step;

        // set shape and array_shape
        shape[level] = shape[level + 1];
        array_shape[level] *= array_shape[level + 1];

        // flatten levels
        for (size_t i = level + 1; i < dims - 1; i++) {
            slice[i] = slice[i + 1];
            shape[i] = shape[i + 1];
            array_shape[i] = array_shape[i + 1];
        }

    } else {
        // if last level, flatten backwards

        // set level slice
        slice[level - 1].start = j + slice[level - 1].start * array_shape[level];
        slice[level - 1].stop = j + slice[level - 1].stop * array_shape[level];
        slice[level - 1].step *= array_shape[level];

        // set shape and array_shape
        array_shape[level - 1] *= array_shape[level];

        // no need to flatten levels, last level
    }

}
