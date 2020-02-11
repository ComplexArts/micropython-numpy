#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "py/runtime.h"
#include "py/binary.h"
#include "py/obj.h"
#include "py/objtuple.h"

#include "ndarray.h"
#include "binary.h"
#include "unary.h"
#include "slice.h"
#include "array.h"
#include "scalar.h"

// ARRAY PRINT

void
ndarray_print_array_aux(const mp_print_t *print,
                        const mp_obj_array_t *array,
                        mp_uint_t index, const mp_int_t step, const mp_uint_t count) {
    for (mp_uint_t i = 0; i < count; i++) {
        if (i != 0)
            mp_print_str(print, ", ");
        mp_obj_print_helper(print, array_get_val_array(array->typecode, array->items, index), PRINT_REPR);
        index += step;
    }
}

void
ndarray_print_array(const mp_print_t *print,
                    const ndarray_obj_t *self, size_t level, const mp_uint_t offset) {

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

    // print the current level
    mp_print_str(print, "[");
    if (level < self->dims) {
        mp_uint_t index = slice.start;
        for (mp_uint_t i = 0; i < n; i++) {
            if (i != 0)
                mp_print_str(print, ",\n");
            ndarray_print_array(print, self, level, offset + index * level_depth);
            index += slice.step;
        }
    } else {
        // last level, just print
        if (n < PRINT_MAX) // small, just print all
            ndarray_print_array_aux(print, self->array,
                                    offset + slice.start * level_depth,
                                    slice.step * level_depth, n);
        else { // break in parts
            ndarray_print_array_aux(print, self->array,
                                    offset + slice.start * level_depth,
                                    slice.step * level_depth, 3);
            mp_printf(print, ", ..., ");
            ndarray_print_array_aux(print, self->array,
                                    offset + (slice.start + (n - 3) * slice.step) * level_depth,
                                    slice.step * level_depth, 3);
        }
    }
    mp_print_str(print, "]");
}

void ndarray_print(const mp_print_t *print, const mp_obj_t self_in, mp_print_kind_t kind) {
    (void)kind;
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    mp_print_str(print, "array(");
    ndarray_print_array(print, self, 0, 0);
    if(self->typecode == NDARRAY_BOOL) {
        mp_print_str(print, ", dtype=bool)");
    } else if(self->typecode == NDARRAY_UINT8) {
        mp_print_str(print, ", dtype=uint8)");
    } else if(self->typecode == NDARRAY_INT8) {
        mp_print_str(print, ", dtype=int8)");
    } else if(self->typecode == NDARRAY_UINT16) {
        mp_print_str(print, ", dtype=uint16)");
    } else if(self->typecode == NDARRAY_INT16) {
        mp_print_str(print, ", dtype=int16)");
    } else if(self->typecode == NDARRAY_FLOAT) {
        mp_print_str(print, ", dtype=float)");
    }
}

// ARRAY NEW

void ndarray_print_debug(ndarray_obj_t *self) {

    printf("\nNDARRAY with %zu dimensions", self->dims);
    printf("\n> typecode: %c", self->typecode);
    printf("\n> size: %zu", self->size);
    printf("\n> shapes: ");
    for (size_t i = 0; i < self->dims; i++) {
        printf("%zu, ", self->shape[i]);
    }
    printf("\n> slices: ");
    for (size_t i = 0; i < self->dims; i++) {
        printf("\n  start = %zu, stop = %zu, step = %zu", self->slice[i].start, self->slice[i].stop, self->slice[i].step);
    }
    printf("\n> array_size: %zu", self->array_size);
    printf("\n> array_shapes: ");
    for (size_t i = 0; i < self->dims; i++) {
        printf("%zu, ", self->array_shape[i]);
    }
    printf("\n> array_bytes: %zu", self->bytes);
    printf("\n> array->typecode: %c", self->array->typecode);
    printf("\n");

}

// ASSIGN

void
ndarray_assign(ndarray_obj_t *self,  mp_obj_t value) {

    if (mp_obj_is_int(value) || mp_obj_is_float(value)) {

#ifdef NDARRAY_DEBUG
        printf("RIGHT-HAND SIDE IS SCALAR\n");
#endif
        array_assign_recursive(self, 0, 0, value);

    } else {

        // value must be iterable
#ifdef NDARRAY_DEBUG
        printf("RIGHT-HAND SIDE IS ITERABLE\n");
#endif
        array_assign_recursive_iterable(self, 0, 0, value);

    }

}

// NEW

STATIC
uint8_t
ndarray_init_helper(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
            { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = mp_const_none} },
            { MP_QSTR_dtype, MP_ARG_INT, {.u_int = NDARRAY_FLOAT } },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    return args[1].u_int;
}

mp_obj_t
ndarray_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {

    mp_arg_check_num(n_args, n_kw, 1, 2, true);
    mp_map_t kw_args;
    mp_map_init_fixed_table(&kw_args, n_kw, args + n_args);
    uint8_t dtype = ndarray_init_helper(n_args, args, &kw_args);

    // determine dims
    mp_int_t dims = array_iterable_dims(args[0]);
    if (dims == 0) {
        mp_raise_ValueError("first argument must be an iterable");
        return mp_const_none;
    }
#ifdef NDARRAY_DEBUG
    printf("dims = %zu\n", dims);
#endif

    // initialize shape
    size_t *shape = m_new(size_t, dims);
    memset(shape, 0, dims * sizeof(size_t));

    // determine shape
    if ( array_iterable_shape(args[0], 0, dims, shape) < 0 ) {
        return mp_const_none;
    }
#ifdef NDARRAY_DEBUG
    printf("shape = ");
    for (int i = 0; i < dims; i++)
        printf("%zu, ", shape[i]);
    printf("\n");
#endif

    // create array, do not initialize
    ndarray_obj_t *self = array_create_new(dims, shape, dtype, false);

    // assign values
    ndarray_assign(self, args[0]);

    return MP_OBJ_FROM_PTR(self);
}

// SLICE

mp_obj_t
array_get_slice_integer(ndarray_obj_t *self, size_t level, size_t index, mp_obj_t value, bool create_view) {

    if (self->dims == 1)  {

        // 1D array
#ifdef NDARRAY_DEBUG
        printf("1D ARRAY\n");
#endif
        // is there a right-hand side?
        if (value != MP_OBJ_SENTINEL) {

            // set scalar
            if (mp_obj_is_int(value) || mp_obj_is_float(value)) {
                array_set_val_array(self->array->typecode, self->array->items, index, value);
            } else {
                mp_raise_ValueError("operands could not be broadcast together");
                return mp_const_none;
            }

        }

        // then return set value
        return array_get_val_array(self->array->typecode, self->array->items, index);

    } else {

        // multidimensional array, flatten
#ifdef NDARRAY_DEBUG
        printf("%zuD ARRAY, level %zu\n", self->dims, level);
#endif

        // return value
        ndarray_obj_t *view;
        if (create_view)
            // create new view, deep copy slice
            view = array_create_new_view(self, self->dims, self->slice, true);
        else
            // otherwise flatten self
            view = self;

        // flatten view
        ndarray_slice_flatten(level, index, view->dims,
                              view->slice, view->shape, view->array_shape);
        view->dims -= 1;

#ifdef NDARRAY_DEBUG
        ndarray_print_debug(view);
        ndarray_print_debug(self);
#endif

        return MP_OBJ_FROM_PTR(view);

    }

}

mp_obj_t
array_get_slice_recursive(ndarray_obj_t *self, size_t level, size_t len, mp_obj_t* items, mp_obj_t value) {

    // return value
    ndarray_obj_t *view = NULL;

    // basic slicing or advanced slicing?
    if (mp_obj_is_int(*items)) {

        // basic slicing, int
#ifdef NDARRAY_DEBUG
        printf("BASIC SLICING: INTEGER, level %zu\n", level);
#endif

        mp_int_t index = mp_obj_get_int(*items);;
        if(index < 0) {
            // flip in case it is negative
            index += *self->shape;
        }
#ifdef NDARRAY_DEBUG
        printf("index = %zu\n", index);
#endif
        if (index < 0 || index >= self->shape[level]) {
            mp_raise_msg(&mp_type_IndexError, "index is out of bounds");
            return mp_const_none;
        }

        // calculate sliced index
        index = self->slice[level].start + index * self->slice[level].step;
#ifdef NDARRAY_DEBUG
        printf("sliced index = %zu\n", index);
#endif

        // TODO: use array_get_slice_integer
        if (self->dims == 1)  {

            // 1D array
#ifdef NDARRAY_DEBUG
            printf("1D ARRAY\n");
#endif
            // is there a right-hand side?
            if (value != MP_OBJ_SENTINEL) {

                // set scalar
                if (mp_obj_is_int(value) || mp_obj_is_float(value)) {
                    array_set_val_array(self->array->typecode, self->array->items, index, value);
                } else {
                    mp_raise_ValueError("operands could not be broadcast together");
                    return mp_const_none;
                }

            }

            // then return set value
            return array_get_val_array(self->array->typecode, self->array->items, index);

        } else {

            // multidimensional array, flatten
#ifdef NDARRAY_DEBUG
            printf("%zuD ARRAY, level %zu\n", self->dims, level);
#endif
            // create new view
            if (level == 0)
                // create new view, deep copy slice
                view = array_create_new_view(self, self->dims, self->slice, true);
            else
                // otherwise flatten self
                // will be a singleton because of integer indexing
                view = self;

            // flatten view
            ndarray_slice_flatten(level, index, view->dims,
                                  view->slice, view->shape, view->array_shape);
            view->dims -= 1;

#ifdef NDARRAY_DEBUG
            ndarray_print_debug(view);
            ndarray_print_debug(self);
#endif

            // decrement level
            level--;

        }

    } else if (MP_OBJ_IS_TYPE(*items, &mp_type_slice)) {

        // basic slicing, slice
#ifdef NDARRAY_DEBUG
        printf("BASIC SLICING: SLICE, level %zu\n", level);
#endif

        // get slice
        mp_bound_slice_t slice;
        mp_seq_get_fast_slice_indexes(self->shape[level], *items, &slice);

#ifdef NDARRAY_DEBUG
        size_t len = ndarray_slice_length(&slice);
        printf("slice: start = %zu, stop = %zu, step = %zu, len = %zu\n", slice.start, slice.stop, slice.step, len);
#endif

        // create view slices
        mp_bound_slice_t *view_slice = m_new(mp_bound_slice_t, self->dims);

        // slice level slice
        ndarray_slice_slice(view_slice + level, self->slice + level, &slice);
#ifdef NDARRAY_DEBUG
        size_t view_len = ndarray_slice_length(view_slice + level);
        printf("view slice: start = %zu, stop = %zu, step = %zu, len = %zu\n", view_slice[level].start, view_slice[level].stop, view_slice[level].step, view_len);
#endif

        // copy remaining slices
        for (int i = 0; i < level; i++) {
            view_slice[i] = self->slice[i];
        }
        for (int i = level + 1; i < self->dims; i++) {
            view_slice[i] = self->slice[i];
        }

        // create view, no need to deep copy slice
        view = array_create_new_view(self, self->dims, view_slice, false);

    } else {

        mp_raise_msg(&mp_type_IndexError, "indices must be integers or slices; advanced indexing not supported yet");

    }

    // this was the last index, assign and return
    if (len == 1) {
        if (view != NULL) {

            if (value != MP_OBJ_SENTINEL) {
                // traverse view for assignment
                ndarray_assign(view, value);
            }

            // return view
            return MP_OBJ_FROM_PTR(view);

        } else
            // This should never happen!
            return mp_const_none;
    }

    // otherwise recurse
    return array_get_slice_recursive(view, level + 1, len - 1, items + 1, value);

}

mp_obj_t
ndarray_subscr(mp_obj_t self_in, mp_obj_t index, mp_obj_t value) {

    // get self
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);

    // is index single or tuple?
    if (MP_OBJ_IS_TYPE(index, &mp_type_tuple)) {

        // got a tuple
#ifdef NDARRAY_DEBUG
        printf("SLICE: GOT A TUPLE!\n");
#endif

        mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(index);
        if (tuple->len > self->dims) {
            mp_raise_msg(&mp_type_IndexError, "too many indices");
        }

        // recurse
        return array_get_slice_recursive(self, 0, tuple->len, tuple->items, value);

    } else {

        // got a singleton
#ifdef NDARRAY_DEBUG
        printf("SLICE: GOT A SINGLETON!\n");
#endif

        // got a singleton
        return array_get_slice_recursive(self, 0, 1, &index, value);

    }

}

// ITERATOR

typedef struct _mp_obj_ndarray_it_t {
    mp_obj_base_t base;
    mp_fun_1_t iternext;
    mp_obj_t ndarray;
    size_t cur;
} mp_obj_ndarray_it_t;

mp_obj_t ndarray_iternext(mp_obj_t self_in);

mp_obj_t mp_obj_new_ndarray_iterator(mp_obj_t ndarray, size_t cur, mp_obj_iter_buf_t *iter_buf) {
    assert(sizeof(mp_obj_ndarray_it_t) <= sizeof(mp_obj_iter_buf_t));
    mp_obj_ndarray_it_t *o = (mp_obj_ndarray_it_t*)iter_buf;
    o->base.type = &mp_type_polymorph_iter;
    o->iternext = ndarray_iternext;
    o->ndarray = ndarray;
    o->cur = cur;
    return MP_OBJ_FROM_PTR(o);
}

mp_obj_t ndarray_getiter(mp_obj_t o_in, mp_obj_iter_buf_t *iter_buf) {
    return mp_obj_new_ndarray_iterator(o_in, 0, iter_buf);
}

mp_obj_t ndarray_iternext(mp_obj_t self_in) {

    mp_obj_ndarray_it_t *self = MP_OBJ_TO_PTR(self_in);
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(self->ndarray);

    mp_bound_slice_t *slice = ndarray->slice;
    size_t index = slice->start + self->cur * slice->step;
#ifdef NDARRAY_DEBUG
    printf("cur = %zu, index = %zu, start = %zu, step = %zu, stop = %zu\n", self->cur, index, slice->start, slice->step, slice->stop);
#endif

    if (index < slice->stop) {

        mp_obj_t value = array_get_slice_integer(ndarray, 0, index, MP_OBJ_SENTINEL, true);
        self->cur++;
        return value;

    } else {
        return MP_OBJ_STOP_ITERATION;
    }
}

// ALL, ANY, MIN, MAX, SUM, PROD

mp_int_t
unary_op_helper(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
            { MP_QSTR_axis, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_obj = mp_const_none} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(pos_args[0], &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
    }

    // axis
    mp_int_t axis = -1;
    mp_obj_t axis_obj = args[0].u_obj;
    if (axis_obj != mp_const_none)
        axis = mp_obj_get_int(axis_obj);

#ifdef NDARRAY_DEBUG
    printf("axis = %zu\n", axis);
    mp_obj_print_helper(&mp_sys_stdout_print, pos_args[0], PRINT_REPR);
    printf("\n");
#endif

    return axis;
}

mp_obj_t
ndarray_all(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // parse args
    mp_int_t axis = unary_op_helper(n_args, pos_args, kw_args);
    // cast to bool first
    mp_obj_t self_bool = unary_op(MP_UNARY_OP_BOOL, pos_args[0]);
    // calculate
    return unary_fun(UNARY_OP_ALL, self_bool, axis);
}

mp_obj_t
ndarray_any(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // parse args
    mp_int_t axis = unary_op_helper(n_args, pos_args, kw_args);
    // cast to bool first
    mp_obj_t self_bool = unary_op(MP_UNARY_OP_BOOL, pos_args[0]);
    // calculate
    return unary_fun(UNARY_OP_ANY, self_bool, axis);
}

mp_obj_t
ndarray_min(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // parse args
    mp_int_t axis = unary_op_helper(n_args, pos_args, kw_args);
    // calculate
    return unary_fun(UNARY_OP_MIN, pos_args[0], axis);
}

mp_obj_t
ndarray_max(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // parse args
    mp_int_t axis = unary_op_helper(n_args, pos_args, kw_args);
    // calculate
    return unary_fun(UNARY_OP_MAX, pos_args[0], axis);
}

mp_obj_t
ndarray_sum(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // parse args
    mp_int_t axis = unary_op_helper(n_args, pos_args, kw_args);
    // calculate
    return unary_fun(UNARY_OP_SUM, pos_args[0], axis);
}

mp_obj_t
ndarray_prod(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // parse args
    mp_int_t axis = unary_op_helper(n_args, pos_args, kw_args);
    // calculate
    return unary_fun(UNARY_OP_PROD, pos_args[0], axis);
}

// SIZE

mp_obj_t
ndarray_size(mp_obj_t self_in) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_in, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }

    // return size
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(self->size);
}


// NDIM

mp_obj_t
ndarray_ndim(mp_obj_t self_in) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_in, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }

    // return dims
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(self->dims);
}

// SHAPE

mp_obj_t
ndarray_shape(mp_obj_t self_in) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_in, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }

    // get self
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);

    // build tuple
    mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(mp_obj_new_tuple(self->dims, NULL));
    for (int i = 0; i < self->dims; i++)
        tuple->items[i] = mp_obj_new_int(self->shape[i]);

    // return tuple
    return MP_OBJ_FROM_PTR(tuple);

}

// RESHAPE

mp_obj_t
ndarray_reshape(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
            {MP_QSTR_a, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = mp_const_none}},
            {MP_QSTR_newshape, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = mp_const_none}},
            {MP_QSTR_order, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_QSTR(MP_QSTR_C)}},
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    // get self
    mp_obj_t self_obj = args[0].u_obj;

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
        return mp_const_none;
    }
    ndarray_obj_t *self_ndarray = MP_OBJ_TO_PTR(self_obj);

    // get newshape
    mp_obj_t newshape_obj = args[1].u_obj;

    // is newshape single or tuple?
    mp_int_t newdims = 0;
    mp_obj_t *items = NULL;
    if (MP_OBJ_IS_TYPE(newshape_obj, &mp_type_tuple)) {
        mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(newshape_obj);
        newdims = tuple->len;
        items = tuple->items;
    } else if (MP_OBJ_IS_TYPE(newshape_obj, &mp_type_list)) {
        mp_obj_list_t *list = MP_OBJ_TO_PTR(newshape_obj);
        newdims = list->len;
        items = list->items;
    } else if (mp_obj_is_int(newshape_obj)) {
        newdims = 1;
        items = &newshape_obj;
    } else {
        mp_raise_ValueError("newshape must be tuple, list or integer");
        return mp_const_none;
    }

    // get order
    GET_STR_DATA_LEN(args[2].u_obj, order, len);
    if ((len != 1) || ((memcmp(order, "C", 1) != 0) && (memcmp(order, "F", 1) != 0))) {
        mp_raise_ValueError("flattening order must be either 'C', or 'F'");
        return mp_const_none;
    }

    if (memcmp(order, "F", 1) == 0) {
        mp_raise_ValueError("order 'F' not supported yet");
        return mp_const_none;
    }

#ifdef NDARRAY_DEBUG
    printf("newdims = %zu\n", newdims);
#endif

    // calculate shape and depth
    size_t newdepth = 1;
    size_t newshape[newdims];
    mp_bound_slice_t *newslice = m_new(mp_bound_slice_t, newdims);
    for (size_t i = 0; i < newdims; i++) {
        mp_uint_t _shape = mp_obj_get_int(items[i]);
        newslice[i].start = 0;
        newslice[i].stop = _shape;
        newslice[i].step = 1;
        newshape[i] = _shape;
        newdepth *= _shape;
    }

    // -1 defaults to size
    if (newdims == 1 && newdepth == -1 ) {
        newslice[0].stop = newshape[0] = newdepth = self_ndarray->size;
    }

#ifdef NDARRAY_DEBUG
    printf("newdepth = %zu\n", newdepth);
    printf("newshape = ");
    for (size_t i = 0; i < newdims; i++) {
        printf("%zu, ", newshape[i]);
    }
    printf("\n");
    printf("newslice = ");
    for (size_t i = 0; i < newdims; i++) {
        printf("  start = %zu, stop = %zu, step = %zu\n", newslice[i].start, newslice[i].stop, newslice[i].step);
    }
    printf("\n");
#endif

    // is new shape compatible?
    if (self_ndarray->size != newdepth) {
        mp_raise_ValueError("newshape is not compatible with current shape");
        return mp_const_none;
    }

    // if not contiguous, copy first
    ndarray_obj_t *reshape_ndarray;
    if (self_ndarray->size == self_ndarray->array_size) {
        reshape_ndarray = self_ndarray;
    } else {
        reshape_ndarray = array_copy(self_ndarray);
    }

    // reshape
    return MP_OBJ_FROM_PTR(array_create_new_view(reshape_ndarray, newdims, newslice, false));

}

// FLATTEN

mp_obj_t
ndarray_flatten(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_order, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_QSTR(MP_QSTR_C)} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    mp_obj_t self_copy = array_copy(pos_args[0]);
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(self_copy);
    
    GET_STR_DATA_LEN(args[0].u_obj, order, len);    
    if((len != 1) || ((memcmp(order, "C", 1) != 0) && (memcmp(order, "F", 1) != 0))) {
        mp_raise_ValueError("flattening order must be either 'C', or 'F'");        
    }

    if(memcmp(order, "F", 1) == 0) {
        // if order == 'F', transpose first
        self_copy = MP_OBJ_TO_PTR(array_transpose(MP_OBJ_FROM_PTR(self_copy)));
    }

    // flatten dims
    ndarray->dims = 1;
    ndarray->shape[0] = ndarray->array_shape[0] = ndarray->array->len;

    // flatten slice
    mp_bound_slice_t *slice = ndarray->slice;
    slice->start = 0;
    slice->stop = ndarray->array->len;
    slice++->step = 1;

    return self_copy;
}

mp_obj_t
ndarray_copy(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
            { MP_QSTR_order, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_QSTR(MP_QSTR_C)} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    mp_obj_t self_copy = array_copy(pos_args[0]);

    GET_STR_DATA_LEN(args[0].u_obj, order, len);
    if((len != 1) || ((memcmp(order, "C", 1) != 0) && (memcmp(order, "F", 1) != 0))) {
        mp_raise_ValueError("order must be either 'C', or 'F'");
    }

    // if order == 'C', we simply have to set m, and n, there is nothing else to do
    if(memcmp(order, "F", 1) == 0) {
        mp_raise_ValueError("order 'F' not supported yet");
    }
    return self_copy;
}

mp_obj_t
ndarray_asbytearray(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return MP_OBJ_FROM_PTR(self->array);
}

// ASTYPE

mp_obj_t
ndarray_astype(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
            { MP_QSTR_dtype, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = NDARRAY_FLOAT} },
            { MP_QSTR_copy, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_obj = mp_const_true} }
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    uint8_t dtype = args[0].u_int;

    return array_astype(pos_args[0], dtype, args[1].u_obj == mp_const_false ? false : true);
}

// CONCATENATE

mp_obj_t
ndarray_concatenate(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
            {MP_QSTR_,     MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = mp_const_none}},
            {MP_QSTR_axis, MP_ARG_KW_ONLY | MP_ARG_INT,  {.u_int = 0}},
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(1, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

#ifdef NDARRAY_DEBUG
    printf("NDARRAY_CONCATENATE\n");
#endif

    // first argument is tuple or list
    mp_obj_t iterable = pos_args[0];
    size_t len;
    mp_obj_t *items;
    if (mp_obj_is_type(iterable, &mp_type_tuple)) {
        mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(iterable);
        len = tuple->len;
        items = tuple->items;
    } else if (mp_obj_is_type(iterable, &mp_type_list)) {
        mp_obj_list_t *list = MP_OBJ_TO_PTR(iterable);
        len = list->len;
        items = list->items;
    } else {
        mp_raise_ValueError("first argument must be tuple or list");
        return mp_const_none;
    }

#ifdef NDARRAY_DEBUG
    printf("len = %zu\n", len);
#endif

    // get axis
    size_t axis = args[1].u_int;
    if (axis < 0) {
        mp_raise_ValueError("axis cannot be negative");
        return mp_const_none;
    }

    // number of objects to concatenate
    if ( len == 0 ) {
        mp_raise_ValueError("need at least one array to concatenate");
        return mp_const_none;
    }

    // get dims
    size_t dims = array_iterable_dims(iterable) - 1;

    // concatenate
    return array_concatenate(axis, dims, len, items);

}

// VSTACK

mp_obj_t
ndarray_vstack(mp_obj_t iterable) {

#ifdef NDARRAY_DEBUG
    printf("NDARRAY_VSTACK\n");
#endif

    // first argument is tuple or list
    size_t len;
    mp_obj_t *items;
    if (mp_obj_is_type(iterable, &mp_type_tuple)) {
        mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(iterable);
        len = tuple->len;
        items = tuple->items;
    } else if (mp_obj_is_type(iterable, &mp_type_list)) {
        mp_obj_list_t *list = MP_OBJ_TO_PTR(iterable);
        len = list->len;
        items = list->items;
    } else {
        mp_raise_ValueError("first argument must be tuple or list");
        return mp_const_none;
    }

    // number of objects to concatenate
    if ( len == 0 ) {
        mp_raise_ValueError("need at least one array to concatenate");
        return mp_const_none;
    }

    // get dims
    mp_int_t dims = array_iterable_dims(iterable) - 1;

    // raise dimensions to 2 in case it is 1
    if (dims == 1)
        dims++;

    // reshape 1D arrays to (1, N) arrays
    for (size_t j = 0; j < len; j++) {

        if (mp_obj_is_type(items[j], &ndarray_type)) {

            // get ndarray
            ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(items[j]);

            if (ndarray->dims == 1) {

                // reshape as (1, N)
                size_t newdims = 2;
                mp_bound_slice_t *newslice = m_new(mp_bound_slice_t, newdims);

                // one element at level 0
                newslice[0].start = 0;
                newslice[0].stop = 1;
                newslice[0].step = 1;

                // copy slice from level 0 to level 1
                newslice[1] = ndarray->slice[0];

                // replace current item by view, do not deep copy slices
                items[j] = array_create_new_view(ndarray, newdims, newslice, false);

            }

        }

    }

    // call concatenate on first axis
    return array_concatenate(0, dims, len, items);

}

// HSTACK

mp_obj_t
ndarray_hstack(mp_obj_t iterable) {

#ifdef NDARRAY_DEBUG
    printf("NDARRAY_HSTACK\n");
#endif

    // first argument is tuple or list
    size_t len;
    mp_obj_t *items;
    if (mp_obj_is_type(iterable, &mp_type_tuple)) {
        mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(iterable);
        len = tuple->len;
        items = tuple->items;
    } else if (mp_obj_is_type(iterable, &mp_type_list)) {
        mp_obj_list_t *list = MP_OBJ_TO_PTR(iterable);
        len = list->len;
        items = list->items;
    } else {
        mp_raise_ValueError("first argument must be tuple or list");
        return mp_const_none;
    }

    // number of objects to concatenate
    if ( len == 0 ) {
        mp_raise_ValueError("need at least one array to concatenate");
        return mp_const_none;
    }

    // reshape ints and floats as 1D arrays
    for (size_t j = 0; j < len; j++) {

        if (mp_obj_is_int(items[j]) || mp_obj_is_float(items[j])) {

            // create 1D array to hold scalar
            size_t dims = 1;
            size_t *shape = m_new(size_t, dims);
            shape[0] = 1;

            // create array, do not initialize
            ndarray_obj_t *ndarray = array_create_new(dims, shape, NDARRAY_FLOAT, false);

            // assign fill value
            array_assign_array(ndarray->array, 0, 1, 1, items[j]);

            // replace current item by array
            items[j] = MP_OBJ_FROM_PTR(ndarray);

        }

    }

    // get dims
    mp_int_t dims = array_iterable_dims(iterable) - 1;
    if (dims == 0) {
        mp_raise_ValueError("cannot concatenate entries");
        return mp_const_none;
    } else if (dims == 1) {
        // call concatenate on first axis
        return array_concatenate(0, dims, len, items);
    } else { // dims >= 1
        // call concatenate on second axis
        return array_concatenate(1, dims, len, items);
    }

}

// FILL, ONES, ZEROS

mp_obj_t
array_create_new_fill(mp_obj_t tuple, mp_obj_t fill_value, ndarray_type_t dtype) {

    // is first argument integer?
    if (mp_obj_is_int(tuple)) {
        mp_obj_t _tuple_obj = mp_obj_new_tuple(1, NULL);
        mp_obj_tuple_t *_tuple = MP_OBJ_TO_PTR(_tuple_obj);
        _tuple->items[0] = tuple;
        tuple = _tuple_obj;
    } else if (!mp_obj_is_type(tuple, &mp_type_tuple)) {
        // is first argument not tuple?
        mp_raise_TypeError("first argument must be a tuple");
        return mp_const_none;
    }

    // is fill_value a scalar?
    if (fill_value != mp_const_none && !ndarray_is_scalar(fill_value)) {
        mp_raise_TypeError("fill _value argument must be a scalar");
        return mp_const_none;
    }

    // calculate shape
    mp_obj_tuple_t *shape_tuple = MP_OBJ_TO_PTR(tuple);
    size_t dims = shape_tuple->len;
    size_t *shape = m_new(size_t, dims);
    for (size_t i = 0; i < dims; i++) {
        mp_int_t _shape = mp_obj_get_int(shape_tuple->items[i]);
        if (_shape < 0) {
            mp_raise_TypeError("shape cannot be negative");
            return mp_const_none;
        }
        shape[i] = _shape;
    }

    // create array, do not initialize
    ndarray_obj_t *ndarray = array_create_new(dims, shape, dtype, false);

    // assign fill value
    if (fill_value != mp_const_none) {
        array_assign_array(ndarray->array, 0, 1, ndarray->size, fill_value);
    }

    // return ndarray
    return  MP_OBJ_FROM_PTR(ndarray);

}

mp_obj_t
ndarray_full(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
            { MP_QSTR_shape, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL} },
            { MP_QSTR_fill_value, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL} },
            { MP_QSTR_dtype, MP_ARG_INT, {.u_int = NDARRAY_FLOAT} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    mp_obj_t shape = args[0].u_obj;
    mp_obj_t fill_value = args[1].u_obj;
    uint8_t dtype = args[2].u_int;

    return MP_OBJ_FROM_PTR(array_create_new_fill(shape, fill_value, dtype));
}

mp_obj_t
ndarray_zeros_or_ones(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args, mp_obj_t fill_value) {
    static const mp_arg_t allowed_args[] = {
            { MP_QSTR_shape, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL} },
            { MP_QSTR_dtype, MP_ARG_INT, {.u_int = NDARRAY_FLOAT} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    mp_obj_t shape = args[0].u_obj;
    uint8_t dtype = args[1].u_int;

    return MP_OBJ_FROM_PTR(array_create_new_fill(shape, fill_value, dtype));
}

mp_obj_t
ndarray_zeros(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return ndarray_zeros_or_ones(n_args, pos_args, kw_args, mp_obj_new_int(0));
}

mp_obj_t
ndarray_ones(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return ndarray_zeros_or_ones(n_args, pos_args, kw_args, mp_obj_new_int(1));
}

mp_obj_t
ndarray_empty(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return ndarray_zeros_or_ones(n_args, pos_args, kw_args, mp_const_none);
}

// EYE

mp_obj_t
ndarray_eye(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
            { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
            { MP_QSTR_M, MP_ARG_OBJ, {.u_obj = mp_const_none } },
            { MP_QSTR_k, MP_ARG_INT, {.u_int = 0} },
            { MP_QSTR_dtype, MP_ARG_INT, {.u_int = NDARRAY_FLOAT} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    size_t n = args[0].u_int, m;
    mp_int_t k = args[2].u_int;
    char dtype = args[3].u_int;
    if(args[1].u_rom_obj == mp_const_none) {
        m = n;
    } else {
        m = mp_obj_get_int(args[1].u_rom_obj);
    }

    // create array
    size_t *shape = m_new(size_t, 2);
    shape[0] = n; shape[1] = m;
    ndarray_obj_t *eye_obj = array_create_new(2, shape, dtype, true);
    mp_obj_t one_obj = mp_obj_new_int(1);
    size_t i = 0;
    if ((k >= 0) && (k < m)) {
        while(k < m) {
            array_set_val_array(dtype, eye_obj->array->items, i * m + k, one_obj);
            k++;
            i++;
        }
    } else if((k < 0) && (-k < n)) {
        k = -k;
        while(k < n) {
            array_set_val_array(dtype, eye_obj->array->items, k * m + i, one_obj);
            k++;
            i++;
        }
    }
    return MP_OBJ_FROM_PTR(eye_obj);
}

// FABS

mp_obj_t
ndarray_fabs(mp_obj_t lhs_obj) {

    return mp_unary_op(MP_UNARY_OP_ABS, lhs_obj);

}

// ISSUBSCTYPE

mp_obj_t
ndarray_issubsctype(mp_obj_t lhs_obj, mp_obj_t rhs_obj) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(lhs_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
    }

    // get lhs
    ndarray_obj_t *lhs = MP_OBJ_TO_PTR(lhs_obj);

    // check substype
    if (mp_obj_is_int(rhs_obj)) {
        char rhs = mp_obj_get_int(rhs_obj);
        char typecode = lhs->typecode;
        switch (typecode) {

            // float >> (int16 >> int8) >> uint16 >> uint8
            case NDARRAY_FLOAT:
                if (rhs == NDARRAY_FLOAT)
                    return mp_const_true;

            case NDARRAY_INT16:
                if (rhs == NDARRAY_INT16)
                    return mp_const_true;
                else if (rhs == NDARRAY_INT8)
                    return mp_const_true;

            case NDARRAY_UINT16:
                if (rhs == NDARRAY_UINT16)
                    return mp_const_true;

            case NDARRAY_UINT8:
                if (rhs == NDARRAY_UINT8)
                    return mp_const_true;
                break;

            // int8 >> uint8
            case NDARRAY_INT8:
                if (rhs == NDARRAY_INT8)
                    return mp_const_true;
                else if (rhs == NDARRAY_UINT8)
                    return mp_const_true;
                break;

            // bool
            case NDARRAY_BOOL:
                if (rhs == NDARRAY_BOOL)
                    return mp_const_true;
                break;

        }
    }

    return mp_const_false;

}

// ARRAY_EQUAL

mp_obj_t
ndarray_array_equal(mp_obj_t lhs_obj, mp_obj_t rhs_obj) {

    // are lhs and rhs ndarray?
    if (!MP_OBJ_IS_TYPE(lhs_obj, &ndarray_type) && !MP_OBJ_IS_TYPE(rhs_obj, &ndarray_type))
        return mp_obj_equal_not_equal(MP_BINARY_OP_EQUAL, lhs_obj, rhs_obj);

    // are lhs and rhs ndarray?
    if (!MP_OBJ_IS_TYPE(lhs_obj, &ndarray_type) || !MP_OBJ_IS_TYPE(rhs_obj, &ndarray_type))
        return mp_const_false;

    ndarray_obj_t *lhs = MP_OBJ_TO_PTR(lhs_obj);
    ndarray_obj_t *rhs = MP_OBJ_TO_PTR(rhs_obj);

    // same dims?
    if (lhs->dims != rhs->dims)
        return mp_const_false;

    // same shape?
    for (size_t i = 0; i < lhs->dims; i++)
        if (lhs->shape[i] != rhs->shape[i])
            return mp_const_false;

    // are equal?
    mp_obj_t equal = binary_op(MP_BINARY_OP_EQUAL, lhs_obj, rhs_obj);

    // all?
    return unary_fun(UNARY_OP_ALL, equal, -1);

}

// FLIP, FLIPUD, FLIPLR

mp_obj_t
array_flip(ndarray_obj_t *self, mp_int_t axis) {

    // create slices
    mp_obj_t minus_one = mp_obj_new_int(-1);
    size_t number_of_slices = (axis > 0 ? axis + 1 : self->dims);
    mp_obj_t slices[number_of_slices];
    if (axis > 0) {
        // flip one axis
        for (size_t i = 0; i < axis; i++) {
            slices[i] = mp_obj_new_slice(mp_const_none, mp_const_none, mp_const_none);
        }
        slices[axis] = mp_obj_new_slice(mp_const_none, mp_const_none, minus_one);
    } else {
        // flip all
        for (size_t i = 0; i < number_of_slices; i++) {
            slices[i] = mp_obj_new_slice(mp_const_none, mp_const_none, minus_one);
        }
    }

    // recurse, do not assign
    return array_get_slice_recursive(self, 0, number_of_slices, slices, MP_OBJ_SENTINEL);
}

mp_obj_t
ndarray_flip(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
            { MP_QSTR_axis, MP_ARG_OBJ, {.u_obj = mp_const_none} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(pos_args[0], &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
    }

    // axis
    mp_int_t axis = -1;
    mp_obj_t axis_obj = args[0].u_obj;
    if (axis_obj != mp_const_none)
        axis = mp_obj_get_int(axis_obj);

    // flip
    return array_flip(MP_OBJ_TO_PTR(pos_args[0]), axis);

}

mp_obj_t
ndarray_flipud(mp_obj_t self_obj) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
    }

    // flip axis 0
    return array_flip(MP_OBJ_TO_PTR(self_obj), 0);

}

mp_obj_t
ndarray_fliplr(mp_obj_t self_obj) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
    }

    // flip axis 1
    return array_flip(MP_OBJ_TO_PTR(self_obj), 1);

}

// TOLIST

mp_obj_t
ndarray_tolist(mp_obj_t self_obj) {

    // is ndarray?
    if (!MP_OBJ_IS_TYPE(self_obj, &ndarray_type)) {
        mp_raise_ValueError("expected ndarray");
    }

    return self_obj;

}