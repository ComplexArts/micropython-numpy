#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>

#include "py/binary.h"
#include "py/smallint.h"
#include "py/objint.h"
#include "py/runtime.h"
#include "py/objarray.h"

#include "ndarray.h"
#include "array.h"
#include "slice.h"
#include "binary.h"

// Helpers to work with binary-encoded data

#ifndef alignof
#define alignof(type) offsetof(struct { char c; type t; }, t)
#endif

// This function is copied verbatim from objarray.c
mp_obj_array_t *array_new(char typecode, size_t n) {
    int typecode_size = array_get_size('@', typecode, NULL);
    mp_obj_array_t *o = m_new_obj(mp_obj_array_t);
    // this step could probably be skipped: we are never going to store a bytearray per se
#if MICROPY_PY_BUILTINS_BYTEARRAY && MICROPY_PY_ARRAY
    o->base.type = (typecode == BYTEARRAY_TYPECODE) ? &mp_type_bytearray : &mp_type_array;
#elif MICROPY_PY_BUILTINS_BYTEARRAY
    o->base.type = &mp_type_bytearray;
#else
    o->base.type = &mp_type_array;
#endif
    o->typecode = typecode;
    o->free = 0;
    o->len = n;
    o->items = m_new(byte, typecode_size * o->len);
    return o;
}

size_t
array_get_size(char struct_type, char val_type, size_t *palign) {
    size_t size = 0;
    int align = 1;
    switch (struct_type) {
        case '<': case '>':
            switch (val_type) {
                case 'b': case 'B': case 'o':
                    size = 1; break;
                case 'h': case 'H':
                    size = 2; break;
                case 'i': case 'I':
                    size = 4; break;
                case 'l': case 'L':
                    size = 4; break;
                case 'q': case 'Q':
                    size = 8; break;
                case 'P': case 'O': case 'S':
                    size = sizeof(void*); break;
                case 'f':
                    size = sizeof(float); break;
                case 'd':
                    size = sizeof(double); break;
            }
            break;
        case '@': {
            // TODO:
            // The simplest heuristic for alignment is to align by value
            // size, but that doesn't work for "bigger than int" types,
            // for example, long long may very well have long alignment
            // So, we introduce separate alignment handling, but having
            // formal support for that is different from actually supporting
            // particular (or any) ABI.
            switch (val_type) {
                case BYTEARRAY_TYPECODE:
                case 'b': case 'B': case 'o':
                    align = size = 1; break;
                case 'h': case 'H':
                    align = alignof(short);
                    size = sizeof(short); break;
                case 'i': case 'I':
                    align = alignof(int);
                    size = sizeof(int); break;
                case 'l': case 'L':
                    align = alignof(long);
                    size = sizeof(long); break;
                case 'q': case 'Q':
                    align = alignof(long long);
                    size = sizeof(long long); break;
                case 'P': case 'O': case 'S':
                    align = alignof(void*);
                    size = sizeof(void*); break;
                case 'f':
                    align = alignof(float);
                    size = sizeof(float); break;
                case 'd':
                    align = alignof(double);
                    size = sizeof(double); break;
            }
        }
    }

    if (size == 0) {
        mp_raise_ValueError("bad typecode");
    }

    if (palign != NULL) {
        *palign = align;
    }
    return size;
}

#define IDX_TO_PTR(type, p, index) (void*)(((type*)p) + index)

void*
array_get_ptr_array(char typecode, void *p, size_t index) {
    switch (typecode) {
        case 'b':
            return IDX_TO_PTR(signed char, p, index);
        case BYTEARRAY_TYPECODE:
        case 'B': case 'o':
            return IDX_TO_PTR(unsigned char, p, index);
        case 'h':
            return IDX_TO_PTR(short, p, index);
        case 'H':
            return IDX_TO_PTR(unsigned short, p, index);
        case 'i':
            return IDX_TO_PTR(int, p, index);
        case 'I':
            return IDX_TO_PTR(unsigned int, p, index);
        case 'l':
            return IDX_TO_PTR(long, p, index);
        case 'L':
            return IDX_TO_PTR(unsigned long, p, index);
#if MICROPY_LONGINT_IMPL != MICROPY_LONGINT_IMPL_NONE
        case 'q':
            return IDX_TO_PTR(long long, p, index);
        case 'Q':
            return IDX_TO_PTR(unsigned long long, p, index);
#endif
#if MICROPY_PY_BUILTINS_FLOAT
        case 'f':
            return IDX_TO_PTR(float, p, index);
        case 'd':
            return IDX_TO_PTR(double, p, index);
#endif
            // Extension to CPython: array of objects
        case 'O':
            return IDX_TO_PTR(mp_obj_t, p, index);
            // Extension to CPython: array of pointers
        case 'P':
            return IDX_TO_PTR(void**, p, index);
        default:
            return NULL;
    }
}

mp_obj_t
array_get_val_array(char typecode, void *p, size_t index) {
    mp_int_t val = 0;
    switch (typecode) {
        case 'b':
            val = ((signed char*)p)[index];
            break;
        case BYTEARRAY_TYPECODE:
        case 'B':
            val = ((unsigned char*)p)[index];
            break;
        case 'o':
            return ((unsigned char*)p)[index] ? mp_const_true : mp_const_false;
        case 'h':
            val = ((short*)p)[index];
            break;
        case 'H':
            val = ((unsigned short*)p)[index];
            break;
        case 'i':
            return mp_obj_new_int(((int*)p)[index]);
        case 'I':
            return mp_obj_new_int_from_uint(((unsigned int*)p)[index]);
        case 'l':
            return mp_obj_new_int(((long*)p)[index]);
        case 'L':
            return mp_obj_new_int_from_uint(((unsigned long*)p)[index]);
#if MICROPY_LONGINT_IMPL != MICROPY_LONGINT_IMPL_NONE
        case 'q':
            return mp_obj_new_int_from_ll(((long long*)p)[index]);
        case 'Q':
            return mp_obj_new_int_from_ull(((unsigned long long*)p)[index]);
#endif
#if MICROPY_PY_BUILTINS_FLOAT
        case 'f':
            return mp_obj_new_float(((float*)p)[index]);
        case 'd':
            return mp_obj_new_float(((double*)p)[index]);
#endif
            // Extension to CPython: array of objects
        case 'O':
            return ((mp_obj_t*)p)[index];
            // Extension to CPython: array of pointers
        case 'P':
            return mp_obj_new_int((mp_int_t)(uintptr_t)((void**)p)[index]);
    }
    return MP_OBJ_NEW_SMALL_INT(val);
}


void
array_set_val_array(char typecode, void *p, size_t index, mp_obj_t val_in) {
    switch (typecode) {
#if MICROPY_PY_BUILTINS_FLOAT
        case 'f':
            ((float*)p)[index] = mp_obj_get_float(val_in);
            break;
        case 'd':
            ((double*)p)[index] = mp_obj_get_float(val_in);
            break;
#endif
        // Extension to CPython: array of objects
        case 'O':
            ((mp_obj_t*)p)[index] = val_in;
            break;
        default:
#if MICROPY_LONGINT_IMPL != MICROPY_LONGINT_IMPL_NONE
            if (mp_obj_is_type(val_in, &mp_type_int)) {
                size_t size = array_get_size('@', typecode, NULL);
                mp_obj_int_to_bytes_impl(val_in, MP_ENDIANNESS_BIG,
                    size, (uint8_t*)p + index * size);
                return;
            }
#endif
            array_set_val_array_from_int(typecode, p, index, mp_obj_get_int(val_in));
    }
}

void
array_set_val_array_from_int(char typecode, void *p, size_t index, mp_int_t val) {
    switch (typecode) {
        case 'b':
            ((signed char*)p)[index] = val;
            break;
        case BYTEARRAY_TYPECODE:
        case 'B':
            ((unsigned char*)p)[index] = val;
            break;
        case 'o':
            ((unsigned char*)p)[index] = (val != 0);
            break;
        case 'h':
            ((short*)p)[index] = val;
            break;
        case 'H':
            ((unsigned short*)p)[index] = val;
            break;
        case 'i':
            ((int*)p)[index] = val;
            break;
        case 'I':
            ((unsigned int*)p)[index] = val;
            break;
        case 'l':
            ((long*)p)[index] = val;
            break;
        case 'L':
            ((unsigned long*)p)[index] = val;
            break;
#if MICROPY_LONGINT_IMPL != MICROPY_LONGINT_IMPL_NONE
        case 'q':
            ((long long*)p)[index] = val;
            break;
        case 'Q':
            ((unsigned long long*)p)[index] = val;
            break;
#endif
#if MICROPY_PY_BUILTINS_FLOAT
        case 'f':
            ((float*)p)[index] = val;
            break;
        case 'd':
            ((double*)p)[index] = val;
            break;
#endif
            // Extension to CPython: array of pointers
        case 'P':
            ((void**)p)[index] = (void*)(uintptr_t)val;
            break;
    }
}

// ARRAY CREATION

ndarray_obj_t *
array_create_new_view(ndarray_obj_t *in, size_t dims, mp_bound_slice_t *slice, bool deep_copy_slice) {

    // initialize
    ndarray_obj_t *self = m_new_obj(ndarray_obj_t);
    self->base.type = &ndarray_type;

    // initialize container
    self->bytes = in->bytes;
    self->array = in->array;
    self->array_size = in->array_size;

    // dims
    if (in->dims != dims) {
        if (in->size != in->array_size) {
            mp_raise_ValueError("view sizes do not match");
            return self;
        }
    }

    // set dims
    self->dims = dims;

    // set typecode
    self->typecode = in->typecode;

    // allocate shape
    self->shape = m_new(size_t, dims);
    self->array_shape = m_new(size_t, dims);

    // set slices
    if (deep_copy_slice) {
        self->slice = m_new(mp_bound_slice_t, dims);
        memcpy(self->slice, slice, sizeof(mp_bound_slice_t) * dims);
    } else
        self->slice = slice;

    // set size
    size_t size = 1;
    for (int i = 0; i < dims; i++) {
        size *= (self->shape[i] = ndarray_slice_length(slice++));
    }
    self->size = size;

    // set array_shape
    if (in->dims == dims) {
        // copy from in
        for (int i = 0; i < dims; i++) {
            self->array_shape[i] = in->array_shape[i];
        }
    } else {
        // copy from shape
        memcpy(self->array_shape, self->shape, sizeof(size_t) * dims);
    }

#ifdef NDARRAY_DEBUG
    // debug
    printf("NEW_VIEW");
    ndarray_print_debug(self);
#endif

    return self;
}

ndarray_obj_t *
array_create_new(size_t dims, size_t* shape, uint8_t typecode, bool initialize) {

    // creates the base ndarray
    ndarray_obj_t *self = m_new_obj(ndarray_obj_t);
    self->base.type = &ndarray_type;

    // set dims, shape, size, and slices
    self->dims = dims;
    self->typecode = typecode;
    self->shape = self->array_shape = shape;
    mp_bound_slice_t *_slice = self->slice = m_new(mp_bound_slice_t, dims);
    size_t array_size = 1;
    for (int i = 0; i < dims; i++) {
        _slice->start = 0;
        _slice->stop = *shape;
        _slice++->step = 1;
        array_size *= *shape++;
    }
    self->array_size = self->size = array_size;

    // set container
    self->array = array_new(typecode, array_size);
    self->bytes = array_size * array_get_size('@', typecode, NULL);

    // initialize container
    if (initialize) {
        memset(self->array->items, 0, self->bytes);
    }

#ifdef NDARRAY_DEBUG
    // debug
    printf("NEW_ARRAY");
    ndarray_print_debug(self);
#endif

    return self;
}

// ARRAY COPY

void
array_copy_recursive(size_t in_level,
                     ndarray_obj_t *in, size_t in_offset,
                     size_t out_level,
                     ndarray_obj_t *out, size_t out_offset) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY_COPY_RECURSIVE, in_level = %ld, in_offset = %ld, out_level = %ld, out_offset = %ld\n",
            in_level, in_offset, out_level, out_offset);
#endif

    // get current slice and shape
    mp_bound_slice_t in_slice = in->slice[in_level];
    size_t n = in->shape[in_level];

#ifdef NDARRAY_DEBUG
    printf("n = %ld\n", n);
    printf("in_slice: start = %ld, stop = %ld, step = %ld\n",
            in_slice.start, in_slice.stop, in_slice.step);
#endif

    // get current slice and shape
    mp_bound_slice_t out_slice = out->slice[out_level];
    if (n != out->shape[out_level] || (in->dims - in_level) != (out->dims - out_level)) {
        // THIS SHOULD NEVER HAPPEN
        mp_raise_ValueError("operands could not be broadcast together");
        return;
    }

#ifdef NDARRAY_DEBUG
    printf("out_slice: start = %ld, stop = %ld, step = %ld\n",
            out_slice.start, out_slice.stop, out_slice.step);
#endif

    // increment level
    in_level++;
    out_level++;

    // calculate remaining depth
    mp_uint_t in_level_depth = 1;
    for (mp_uint_t i = in_level; i < in->dims; i++) {
        in_level_depth *= in->array_shape[i];
    }

    mp_uint_t out_level_depth = 1;
    for (mp_uint_t i = out_level; i < out->dims; i++) {
        out_level_depth *= out->array_shape[i];
    }

    if (in_level < in->dims) {

        // upper level, recurse
        mp_uint_t in_index = in_slice.start, out_index = out_slice.start;
        for (mp_uint_t i = 0; i < n; i++) {
            array_copy_recursive(in_level,
                                 in, in_offset + in_index * in_level_depth,
                                 out_level,
                                 out, out_offset + out_index * out_level_depth);
            in_index += in_slice.step;
            out_index += out_slice.step;
        }

    } else {

        // last level, just copy
        uint8_t _sizeof = array_get_size('@', in->array->typecode, NULL);

        // number of elements to copy
        size_t count = ndarray_slice_length(&in_slice);

        uint8_t *in_buffer = in->array->items;
        uint8_t *out_buffer = out->array->items;
        if (in_slice.step == 1 && out_slice.step == 1) {

            // if both input and output buffers are contiguous, simply copy
            memcpy(out_buffer + out_offset * _sizeof, in_buffer + in_offset * _sizeof, count * _sizeof);

        } else {

            // is slice, need to iterate
            in_offset += in_slice.start;
            out_offset += out_slice.start;
            for (size_t i = 0; i < count; i++) {
                // copy one by one
                memcpy(out_buffer + out_offset * _sizeof, in_buffer + in_offset * _sizeof, _sizeof);
                in_offset += in_slice.step;
                out_offset += out_slice.step;
            }

        }
    }
}

mp_obj_t
array_copy(mp_obj_t self_in) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY_COPY\n");
#endif

    // get pointer to self
    ndarray_obj_t *in = MP_OBJ_TO_PTR(self_in);

    // copy in shape
    // it seems not safe to copy shape in case the input array has an inplace change of shape
    size_t dims = in->dims;
    size_t *shape = m_new(size_t, dims);
    for (size_t i = 0; i < dims; i++) {
        shape[i] = in->shape[i];
    }

    // create array, do not initialize, will need to copy anyway
    ndarray_obj_t *out = array_create_new(dims, shape, in->typecode, false);

    // copy entries
    if (out->array->len == in->array->len) {
        // straight out copy if the underlying array is the same shape
        memcpy(out->array->items, in->array->items, in->bytes);
    } else {
        // otherwise will need to copy recursively
        array_copy_recursive(0,
                             in, 0,
                             0,
                             out, 0);
    }
    return MP_OBJ_FROM_PTR(out);
}

// ASTYPE

void
array_astype_recursive(size_t level,
                       ndarray_obj_t *origin, mp_uint_t origin_offset,
                       ndarray_obj_t *destination, mp_uint_t destination_offset) {

    // get current slice and shape
    mp_bound_slice_t origin_slice = origin->slice[level];
    size_t n = origin->shape[level];

    // increment level
    level++;

    // calculate remaining depth
    mp_uint_t origin_level_depth = 1, destination_level_depth = 1;
    for (mp_uint_t i = level; i < origin->dims; i++) {
        origin_level_depth *= origin->array_shape[i];
        destination_level_depth *= destination->array_shape[i];
    }

    if (level < origin->dims) {
        // upper level
        mp_uint_t origin_index = origin_slice.start, destination_index = 0;
        for (mp_uint_t i = 0; i < n; i++) {
            array_astype_recursive(level,
                                   origin, origin_offset + origin_index * origin_level_depth,
                                   destination, destination_offset + destination_index * destination_level_depth);
            origin_index += origin_slice.step;
            destination_index ++;
        }
    } else {
        // last level, just cast
        mp_uint_t origin_index = origin_offset + origin_slice.start,
                destination_index = destination_offset;
        if (origin->typecode == NDARRAY_FLOAT && destination->typecode != origin->typecode) {
            if (destination->typecode == NDARRAY_BOOL) {
                // need to convert from float to bool first
                for (mp_uint_t i = 0; i < n; i++) {
                    // set value
                    mp_obj_t value = mp_obj_new_bool(mp_obj_get_float(
                            array_get_val_array(origin->array->typecode, origin->array->items, origin_index)) != 0.);
                    array_set_val_array(destination->array->typecode,
                                        destination->array->items, destination_index, value);
                    origin_index += origin_slice.step;
                    destination_index++;
                }
            } else {
                // need to convert from float to integer first
                for (mp_uint_t i = 0; i < n; i++) {
                    // set value
                    mp_obj_t value = MP_OBJ_NEW_SMALL_INT(mp_obj_get_float(
                            array_get_val_array(origin->array->typecode, origin->array->items, origin_index)));
                    array_set_val_array(destination->array->typecode,
                                        destination->array->items, destination_index, value);
                    origin_index += origin_slice.step;
                    destination_index++;
                }
            }
        } else {
            for (mp_uint_t i = 0; i < n; i++) {
                // set value
                mp_obj_t value = array_get_val_array(origin->array->typecode, origin->array->items, origin_index);
                array_set_val_array(destination->array->typecode,
                                    destination->array->items, destination_index, value);
                origin_index += origin_slice.step;
                destination_index++;
            }
        }
    }
}

mp_obj_t
array_astype(mp_obj_t self_obj, char dtype, bool copy) {

    // get array
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_obj);

    // copy?
    copy = copy || (self->typecode != dtype);

    // if same typecode
    if (self->typecode == dtype) {
        if (!copy)
            // return self if no copy needed
            return MP_OBJ_FROM_PTR(self);
        else
            // return copy
            return array_copy(self_obj);
    }

    // create destination without initializing
    ndarray_obj_t *destination = array_create_new(self->dims, self->shape, dtype, false);

    // cast
    array_astype_recursive(0, self, 0, destination, 0);

    return MP_OBJ_FROM_PTR(destination);

}

// TRANSPOSE

void
array_transpose_recursive(size_t in_level,
                          ndarray_obj_t *in, size_t in_offset,
                          ndarray_obj_t *out, size_t out_offset) {

    // get current slice and shape
    mp_bound_slice_t in_slice = in->slice[in_level];
    size_t n = in->shape[in_level];
    size_t out_level = out->dims - in_level - 1;

#ifdef NDARRAY_DEBUG
    printf("ARRAY_TRANSPOSE_RECURSIVE, in_level = %ld, in_offset = %ld, out_level = %ld, out_offset = %ld\n",
            in_level, in_offset, out_level, out_offset);
#endif

#ifdef NDARRAY_DEBUG
    printf("n = %ld\n", n);
    printf("in_slice: start = %ld, stop = %ld, step = %ld\n",
            in_slice.start, in_slice.stop, in_slice.step);
#endif

    // get current slice and shape
    mp_bound_slice_t out_slice = out->slice[out_level];
    if (n != out->shape[out_level]) {
        // THIS SHOULD NEVER HAPPEN
        mp_raise_ValueError("operands could not be broadcast together");
        return;
    }

#ifdef NDARRAY_DEBUG
    printf("out_slice: start = %ld, stop = %ld, step = %ld\n",
            out_slice.start, out_slice.stop, out_slice.step);
#endif

    // increment levels
    in_level++;

    // calculate remaining depth
    mp_uint_t in_level_depth = 1;
    for (mp_uint_t i = in_level; i < in->dims; i++) {
        in_level_depth *= in->array_shape[i];
    }

    mp_uint_t out_level_depth = 1;
    for (mp_uint_t i = out_level + 1; i < out->dims; i++) {
        out_level_depth *= out->array_shape[i];
    }

#ifdef NDARRAY_DEBUG
    printf("in_level_depth = %ld, out_level_depth = %ld\n",
            in_level_depth, out_level_depth);
#endif

    if (in_level < in->dims) {

        // upper level, recurse
        mp_uint_t in_index = in_slice.start, out_index = out_slice.start;
        for (mp_uint_t i = 0; i < n; i++) {
            array_transpose_recursive(in_level,
                                      in, in_offset + in_index * in_level_depth,
                                      out, out_offset + out_index * out_level_depth);
            in_index += in_slice.step;
            out_index += out_slice.step;
        }

    } else {

        // last level, just copy
        uint8_t _sizeof = array_get_size('@', in->array->typecode, NULL);

        // number of elements to copy
        size_t count = ndarray_slice_length(&in_slice);

        in_offset += in_slice.start;
        out_offset += out_slice.start * out_level_depth;
        for (size_t i = 0; i < count; i++) {
            // copy one by one
            memcpy(((uint8_t*)out->array->items) + out_offset * _sizeof,
                   ((uint8_t*)in->array->items) + in_offset * _sizeof, _sizeof);
            in_offset += in_slice.step;
            out_offset += out_slice.step * out_level_depth;
        }

    }
}

mp_obj_t
array_transpose(mp_obj_t self_in) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY_TRANSPOSE\n");
#endif

    // get pointer to self
    ndarray_obj_t *in = MP_OBJ_TO_PTR(self_in);

    // copy shape in reverse
    size_t dims = in->dims, max_shape = 1;
    size_t *shape = m_new(size_t, dims);
    for (size_t i = 0; i < dims; i++) {
        size_t _shape = in->shape[dims-i-1];
        shape[i] = _shape;
        max_shape = max_shape > _shape ? max_shape : _shape;
    }

    // create array, do not initialize, will need to copy anyway
    ndarray_obj_t *out = array_create_new(dims, shape, in->typecode, false);

    // copy entries if packed and "one dimensional"
    if (in->size == in->array_size && in->size == max_shape) {
        // straight out copy if the underlying array is the same shape
        memcpy(out->array->items, in->array->items, in->bytes);
    } else {
        // otherwise will need to transpose recursively
        array_transpose_recursive(0, in, 0, out, 0);
    }
    return MP_OBJ_FROM_PTR(out);
}

// ARRAY DOT

void
array_dot_recursive(size_t lhs_level,
                    ndarray_obj_t *lhs, size_t lhs_offset,
                    size_t rhs_level,
                    ndarray_obj_t *rhs, size_t rhs_offset,
                    ndarray_obj_t *out, size_t out_offset) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY_DOT_RECURSIVE, lhs_level = %ld, lhs_offset = %ld, rhs_level = %ld, rhs_offset = %ld, out_offset = %ld\n",
            lhs_level, lhs_offset, rhs_level, rhs_offset, out_offset);
#endif

    // get current slice and shape
    mp_bound_slice_t lhs_slice = lhs->slice[lhs_level];
    mp_bound_slice_t rhs_slice = rhs->slice[rhs_level];
    size_t n = lhs->shape[lhs_level];

    // calculate out_level
    size_t out_level;
    if (lhs_level < lhs->dims - 1) {
        out_level = lhs_level;
    } else if (lhs->dims == 1) {
        out_level = 0;
    } else {
        out_level = lhs->dims - 2 + rhs_level;
    }
    mp_bound_slice_t out_slice = out->slice[out_level];

    // calculate remaining depth
    mp_uint_t lhs_level_depth = 1;
    for (mp_uint_t i = lhs_level + 1; i < lhs->dims; i++) {
        lhs_level_depth *= lhs->array_shape[i];
    }

    mp_uint_t rhs_level_depth = 1;
    for (mp_uint_t i = rhs_level + 1; i < rhs->dims; i++) {
        rhs_level_depth *= rhs->array_shape[i];
    }

    mp_uint_t out_level_depth = 1;
    for (mp_uint_t i = out_level + 1; i < out->dims; i++) {
        out_level_depth *= out->array_shape[i];
    }

#ifdef NDARRAY_DEBUG
    printf("n = %ld, out_level = %ld\n", n, out_level);
    printf("lhs_level_depth = %ld, rhs_level_depth = %ld, out_level_depth = %ld\n",
            lhs_level_depth, rhs_level_depth, out_level_depth);
    printf("lhs_slice: start = %ld, stop = %ld, step = %ld\n",
           lhs_slice.start, lhs_slice.stop, lhs_slice.step);
    printf("rhs_slice: start = %ld, stop = %ld, step = %ld\n",
           rhs_slice.start, rhs_slice.stop, rhs_slice.step);
    printf("out_slice: start = %ld, stop = %ld, step = %ld\n",
           out_slice.start, out_slice.stop, out_slice.step);
#endif

    if (lhs_level < lhs->dims - 1) {

#ifdef NDARRAY_DEBUG
        printf("LHS UPPER LEVEL, RECURSE\n");
#endif

        // lhs upper level, recurse
        mp_uint_t lhs_index = lhs_slice.start, out_index = out_slice.start;
        for (mp_uint_t i = 0; i < n; i++) {
            array_dot_recursive(lhs_level + 1,
                                lhs, lhs_offset + lhs_index * lhs_level_depth,
                                rhs_level,
                                rhs, rhs_offset,
                                out, out_offset + out_index * out_level_depth);
            lhs_index += lhs_slice.step;
            out_index += out_slice.step;
        }

    } else {

        if (rhs->dims > 1 && rhs_level < rhs->dims - 2) {

#ifdef NDARRAY_DEBUG
            printf("RHS UPPER LEVEL, RECURSE\n");
#endif

            // upper level, recurse
            mp_uint_t rhs_index = rhs_slice.start, out_index = out_slice.start;
            for (mp_uint_t i = 0; i < n; i++) {
                array_dot_recursive(lhs_level,
                                    lhs, lhs_offset,
                                    rhs_level + 1,
                                    rhs, rhs_offset + rhs_index * rhs_level_depth,
                                    out, out_offset + out_index * out_level_depth);
                rhs_index += rhs_slice.step;
                out_index += out_slice.step;
            }

        } else {

            size_t m = rhs->dims > 1 ? rhs->shape[rhs_level + 1] : 1;

#ifdef NDARRAY_DEBUG
            printf("OPERATE LEVEL, m = %ld\n", m);
#endif

            // last level, multiply
            lhs_offset += lhs_slice.start;

            mp_uint_t rhs_index = rhs_slice.start, out_index = out_slice.start;
            for (mp_uint_t i = 0; i < m; i++) {

                // multiply
                binary_op_dot(out->typecode,
                              array_get_ptr_array(out->typecode, out->array->items,
                                                  out_offset + out_index),
                              lhs->typecode,
                              array_get_ptr_array(lhs->typecode, lhs->array->items, lhs_offset),
                              lhs_slice.step,
                              rhs->typecode,
                              array_get_ptr_array(rhs->typecode, rhs->array->items,
                                                  rhs_offset + rhs_index),
                              rhs_slice.step * rhs_level_depth,
                              n, NDARRAY_BINARY_OP_DOT);

                rhs_index += rhs_slice.step;
                out_index += out_slice.step;
            }

        }

    }

}


mp_obj_t
array_dot(mp_obj_t lhs_obj, mp_obj_t rhs_obj) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY_DOT\n");
#endif

    // are lhs and rhs ndarray?
    if (!MP_OBJ_IS_TYPE(lhs_obj, &ndarray_type) || !MP_OBJ_IS_TYPE(rhs_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }

    // get pointers
    ndarray_obj_t *lhs = MP_OBJ_TO_PTR(lhs_obj);
    ndarray_obj_t *rhs = MP_OBJ_TO_PTR(rhs_obj);
    ndarray_obj_t *out;
    bool reduce_to_scalar = false;
    char typecode = binary_upcast(lhs->typecode, rhs->typecode);

#ifdef NDARRAY_DEBUG
    printf("lhs:");
    ndarray_print_debug(lhs);
    printf("rhs:");
    ndarray_print_debug(rhs);
#endif

    // shape of the output
    if (rhs->dims == 1) {

        if (lhs->dims == 1) {

#ifdef NDARRAY_DEBUG
            printf("1Dx1D DOT\n");
#endif

            // dot product of 1D arrays
            if (lhs->size != rhs->size) {
                mp_raise_ValueError("1D arrays do not have the same length");
                return mp_const_none;
            }

            // create 1x1 return array and initialize to zero
            size_t shape = 1;
            out = array_create_new(1, &shape, typecode, true);
            reduce_to_scalar = true;

        } else {

#ifdef NDARRAY_DEBUG
            printf("%ldDx%ldD (1D) DOT\n", lhs->shape[lhs->dims - 1], rhs->size);
#endif

            // dot product with 1D arrays
            if (lhs->shape[lhs->dims - 1] != rhs->size) {
                mp_raise_ValueError("arrays do not have compatible dimensions");
                return mp_const_none;
            }

            // create return array and initialize to zero
            size_t dims = lhs->dims - 1;
            size_t *shape = m_new(size_t, dims);
            for (size_t i = 0; i < dims; i++) {
                shape[i] = lhs->shape[i];
            }
            out = array_create_new(dims, shape, typecode, true);

        }

    } else {

#ifdef NDARRAY_DEBUG
        printf("%ldDx%ldD (ND) DOT\n", lhs->shape[lhs->dims - 1], rhs->shape[rhs->dims - 2]);
#endif

        // array multiplication
        if (lhs->shape[lhs->dims - 1] != rhs->shape[rhs->dims - 2]) {
            mp_raise_ValueError("arrays do not have compatible dimensions");
            return mp_const_none;
        }

        // create return array and initialize to zero
        size_t dims = (lhs->dims - 1) + (rhs->dims - 1);
        size_t *shape = m_new(size_t, dims);
        for (size_t i = 0; i < lhs->dims - 1; i++) {
            shape[i] = lhs->shape[i];
        }
        size_t j = lhs->dims - 1;
        for (size_t i = 0; i < rhs->dims - 2; i++) {
            shape[j++] = rhs->shape[i];
        }
        shape[j++] = rhs->shape[rhs->dims - 1];

#ifdef NDARRAY_DEBUG
        printf("j = %ld, shape = ", j);
        for (int i = 0; i < dims; i++)
            printf("%ld, ", shape[i]);
        printf("\n");
#endif

        out = array_create_new(dims, shape, typecode, true);

    }

#ifdef NDARRAY_DEBUG
    printf("out:");
    ndarray_print_debug(out);
#endif

    // call array_dot_recursive
    array_dot_recursive(0, lhs, 0, 0, rhs, 0, out, 0);

    // assign result to object
    if (reduce_to_scalar)
        return array_get_val_array(typecode, out->array->items, 0);
    else
        return MP_OBJ_TO_PTR(out);

}

// assign

void
array_assign_array(mp_obj_array_t *array,
                   mp_uint_t start, mp_int_t step, mp_uint_t count,
                   mp_obj_t value) {
    mp_uint_t index = start;
    for (mp_uint_t i = 0; i < count; i++) {
        // set value
        array_set_val_array(array->typecode, array->items, index, value);
        index += step;
    }
}

void
array_assign_recursive(ndarray_obj_t *self, size_t level, mp_uint_t offset, mp_obj_t value) {

    // get current slice and shape
    mp_bound_slice_t slice = self->slice[level];
    size_t n = self->shape[level];

    // increment level
    level++;

    // calculate remaining depth
    mp_uint_t level_depth = 1;
    for (mp_uint_t i = level; i < self->dims; i++) {
        level_depth *= self->array_shape[i];
    }

    if (level < self->dims) {
        // upper level
        mp_uint_t index = slice.start;
        for (mp_uint_t i = 0; i < n; i++) {
            array_assign_recursive(self, level, offset + index * level_depth, value);
            index += slice.step;
        }
    } else {
        // last level, just assign
        array_assign_array(self->array, offset + slice.start, slice.step, n, value);
    }

}

void
array_assign_array_iterable(mp_obj_array_t *array,
                            mp_uint_t start, mp_int_t step, mp_uint_t count,
                            mp_obj_t iterable) {
#ifdef NDARRAY_DEBUG
    printf("ARRAY_ASSIGN_ARRAY_ITERABLE, start = %ld, step = %ld, count = %ld\n", start, step, count);
#endif

    mp_uint_t index = start;
    for (mp_uint_t i = 0; i < count; i++) {
        // get value
        mp_obj_t value = mp_iternext(iterable);
        if (value == MP_OBJ_STOP_ITERATION) {
            // THIS SHOULD NEVER HAPPEN, LEN HAS BEEN CHECKED A PRIORI
            mp_raise_ValueError("operands could not be broadcast together");
            return;
        }
        // set value
        array_set_val_array(array->typecode, array->items, index, value);
        index += step;
    }
}

void
array_assign_recursive_iterable(ndarray_obj_t *self, size_t level, mp_uint_t offset, mp_obj_t value) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY_ASSIGN_RECURSIVE_ITERABLE, level = %ld, offset = %ld\n", level, offset);
    ndarray_print_debug(self);
#endif

    // get current slice and shape
    mp_bound_slice_t slice = self->slice[level];
    size_t n = self->shape[level];

    // check iterable length
    // only works if iterable supports the len operator!
    mp_obj_t len_obj = mp_obj_len_maybe(value);
    if (len_obj == MP_OBJ_NULL) {
        mp_raise_ValueError("can only assign to iterables that support the operator len");
        return;
    }
    size_t iterable_len = MP_OBJ_SMALL_INT_VALUE(len_obj);
    if (iterable_len != n) {
        mp_raise_ValueError("operands could not be broadcast together");
    }

    // get iterable
    mp_obj_t iterable = mp_getiter(value, NULL);

    // increment level
    level++;

    // calculate remaining depth
    mp_uint_t level_depth = 1;
    for (mp_uint_t i = level; i < self->dims; i++) {
        level_depth *= self->array_shape[i];
    }

#ifdef NDARRAY_DEBUG
    printf("level_depth = %ld\n", level_depth);
#endif

    if (level < self->dims) {
        // upper level
        // TODO: check for broadcast length one level before the final assignment
        mp_uint_t index = slice.start;
        mp_obj_t item;
        for (mp_uint_t i = 0; i < n; i++) {
            // get iterables
            item = mp_iternext(iterable);
            if (item == MP_OBJ_STOP_ITERATION) {
                // THIS SHOULD NEVER HAPPEN, LEN HAS BEEN CHECKED A PRIORI
                mp_raise_ValueError("operands could not be broadcast together");
                return;
            }
            array_assign_recursive_iterable(self, level, offset + index * level_depth, item);
            index += slice.step;
        }
    } else {
        // last level, just assign
        array_assign_array_iterable(self->array, offset + slice.start * level_depth, slice.step, n, iterable);
    }

}


// iterables

mp_int_t
array_iterable_dims(mp_obj_t iterable) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY_ITERABLE_DIMS\n");
#endif

    size_t dims = 0;
    mp_obj_t iterable_len;
    while (true) {
        // shortcut if ndarray
        if (mp_obj_is_type(iterable, &ndarray_type)) {

            ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(iterable);
            return dims + ndarray->dims;

        } else {

            // does current item have len?
            iterable_len = mp_obj_len_maybe(iterable);
            // stop if not iterable
            if (iterable_len == MP_OBJ_NULL)
                break;
            // otherwise go down one more level
            dims++;
            iterable = mp_getiter(iterable, NULL);
            // stop if empty iterable
            if ((iterable = mp_iternext(iterable)) == MP_OBJ_STOP_ITERATION)
                break;

        }
    }

    return dims;

}

mp_int_t
array_iterable_shape(mp_obj_t iterable,
                     size_t level,
                     size_t dims,
                     size_t *shape) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY_ITERABLE_SHAPE, level = %ld\n", level);
#endif

    if (mp_obj_is_type(iterable, &ndarray_type)) {

        ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(iterable);

        // check depth
        if (ndarray->dims != dims - level) {
            mp_raise_ValueError("iterables do not have the same depth");
            return -1;
        }

        // check len and shape
        mp_int_t len = ndarray->shape[0];
        for (size_t i = 0; i < ndarray->dims; i++) {
            size_t _shape = ndarray->shape[i];
            if (shape[level + i] == 0)
                shape[level + i] = _shape;
            else if (_shape != shape[level + i]) {
                mp_raise_ValueError("iterables do not have the same length");
                return -1;
            }
        }

        return len;

    } else {

        // get iterable shape
        mp_obj_t iterable_len = mp_obj_len_maybe(iterable);
        if (iterable_len == MP_OBJ_NULL) {
            mp_raise_ValueError("argument is not iterable");
            return -1;
        }
        mp_int_t len = MP_OBJ_SMALL_INT_VALUE(iterable_len);

        // set shape
        if (shape[level] == 0)
            shape[level] = len;
        else if (len != shape[level]) {
            mp_raise_ValueError("iterables do not have the same length");
            return -1;
        }

        if (level < dims - 1) {

            // recurse
            mp_obj_t item, iter = mp_getiter(iterable, NULL);
            for (size_t i = 0; i < len; i++) {

                if ((item = mp_iternext(iter)) == MP_OBJ_STOP_ITERATION) {
                    mp_raise_ValueError("argument is not iterable");
                    return -1;
                }

                if (array_iterable_shape(item, level + 1, dims, shape) < 1) {
                    return -1;
                }

            }

        }

        return len;
    }
}

// concatenate

mp_obj_t
array_concatenate(size_t axis, size_t dims, size_t len, mp_obj_t *items) {

#ifdef NDARRAY_DEBUG
    printf("ARRAY_CONCATENATE\n");
#endif

    if (axis >= dims) {
        mp_raise_ValueError("axis cannot be larger than array dimensions");
        return mp_const_none;
    }
#ifdef NDARRAY_DEBUG
    printf("dims = %ld\n", dims);
#endif

    // get shapes
    size_t *shape = m_new(size_t, dims);
    memset(shape, 0, dims * sizeof(size_t));
    size_t axis_shape = 0;
    size_t axis_shapes[len];
    char typecode = NDARRAY_FLOAT;
    for (size_t j = 0; j < len; j++) {

        // set shape
        shape[axis] = 0;
        array_iterable_shape(items[j], 0, dims, shape);
        axis_shapes[j] = shape[axis];
        axis_shape += shape[axis];

#ifdef NDARRAY_DEBUG
        printf("shape = ");
        for (int i = 0; i < dims; i++)
            printf("%ld, ", shape[i]);
        printf("\n");
#endif

        // upcast?
        char _typecode;
        if (mp_obj_is_type(items[j], &ndarray_type)) {
            ndarray_obj_t *_ndarray = MP_OBJ_TO_PTR(items[j]);
            _typecode = _ndarray->typecode;
        } else {
            _typecode = NDARRAY_FLOAT;
        }
        if (j > 0)
            typecode = binary_upcast(typecode, _typecode);
        else
            typecode = _typecode;

    }
    shape[axis] = axis_shape;

#ifdef NDARRAY_DEBUG
    printf("shape = ");
    for (int i = 0; i < dims; i++)
        printf("%ld, ", shape[i]);
    printf("\n");
#endif

    // create array, do not initialize
    ndarray_obj_t *ndarray = array_create_new(dims, shape, typecode, false);

    // create view
    ndarray_obj_t *view = array_create_new_view(ndarray, dims, ndarray->slice, true);

    size_t index = 0;
    for (size_t j = 0; j < len; j++) {

        // modify axis slice and shape
        view->slice[axis].start = index;
        view->slice[axis].stop = index + axis_shapes[j];
        view->shape[axis] = axis_shapes[j];

#ifdef NDARRAY_DEBUG
        printf("shape = ");
        for (int i = 0; i < dims; i++)
            printf("%ld, ", view->shape[i]);
        printf("\n");
        printf("slices: \n");
        for (size_t i = 0; i < dims; i++) {
            printf("  start = %ld, stop = %ld, step = %ld\n", view->slice[i].start, view->slice[i].stop, view->slice[i].step);
        }
#endif

        // assign
        array_assign_recursive_iterable(view, 0, 0, items[j]);

        // increment index
        index += axis_shapes[j];

    }

    // return array
    return MP_OBJ_FROM_PTR(ndarray);

}