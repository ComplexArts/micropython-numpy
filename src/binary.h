#ifndef _BINARY_H_
#define _BINARY_H_

#include "py/binary.h"

#include <ndarray.h>

#define NDARRAY_BINARY_OP_MAX MP_BINARY_OP_IS_NOT + 1
#define NDARRAY_BINARY_OP_MIN MP_BINARY_OP_IS_NOT + 2
#define NDARRAY_BINARY_OP_DOT MP_BINARY_OP_IS_NOT + 3

mp_obj_t
binary_op(mp_binary_op_t op, mp_obj_t lhs, mp_obj_t rhs);

mp_obj_t
binary_maximum(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);

mp_obj_t
binary_minimum(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args);

typedef
void (*binary_op_fun_t)(char self_typecode, void *self_ptr,
                        char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
                        char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
                        mp_binary_op_t op);

void
binary_op_recursive(size_t level,
                    ndarray_obj_t *self_ndarray, size_t self_offset,
                    mp_obj_t lhs, size_t lhs_offset,
                    mp_obj_t rhs, size_t rhs_offset,
                    mp_binary_op_t op,
                    binary_op_fun_t op_fun);

mp_obj_t
binary_create_op_result(mp_obj_t lhs, mp_obj_t rhs, mp_binary_op_t op);

char
binary_upcast(char lhs_typecode, char rhs_typecode);

void
binary_op_add(char self_typecode, void *self_ptr,
              char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
              char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
              mp_binary_op_t op);

void
binary_op_subtract(char self_typecode, void *self_ptr,
                   char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
                   char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
                   mp_binary_op_t op);

void
binary_op_multiply(char self_typecode, void *self_ptr,
                   char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
                   char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
                   mp_binary_op_t op);

void
binary_op_divide(char self_typecode, void *self_ptr,
                 char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
                 char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
                 mp_binary_op_t op);

void
binary_op_equal(char self_typecode, void *self_ptr,
                char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
                char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
                mp_binary_op_t op);

void
binary_op_not_equal(char self_typecode, void *self_ptr,
                    char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
                    char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
                    mp_binary_op_t op);

void
binary_op_less(char self_typecode, void *self_ptr,
               char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
               char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
               mp_binary_op_t op);

void
binary_op_less_equal(char self_typecode, void *self_ptr,
                     char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
                     char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
                     mp_binary_op_t op);

void
binary_op_more(char self_typecode, void *self_ptr,
               char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
               char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
               mp_binary_op_t op);

void
binary_op_more_equal(char self_typecode, void *self_ptr,
                     char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
                     char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
                     mp_binary_op_t op);

void
binary_op_and(char self_typecode, void *self_ptr,
              char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
              char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
              mp_binary_op_t op);

void
binary_op_or(char self_typecode, void *self_ptr,
             char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
             char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
             mp_binary_op_t op);

void
binary_op_max(char self_typecode, void *self_ptr,
              char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
              char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
              mp_binary_op_t op);

void
binary_op_min(char self_typecode, void *self_ptr,
              char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
              char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
              mp_binary_op_t op);

void
binary_op_dot(char self_typecode, void *self_ptr,
              char lhs_typecode, void *lhs_ptr, mp_int_t lhs_step,
              char rhs_typecode, void *rhs_ptr, mp_int_t rhs_step, mp_uint_t count,
              mp_binary_op_t op);

#endif