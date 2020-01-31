#ifndef _ARRAY2D_H_
#define _ARRAY2D_H_

#include <ndarray.h>

mp_obj_t array2d_qr(mp_obj_t self_obj);
mp_obj_t array2d_qr_solve(mp_obj_t self_obj, mp_obj_t b_obj);

mp_obj_t array2d_lu(mp_obj_t self_obj);
mp_obj_t array2d_lu_det(mp_obj_t self_obj);
mp_obj_t array2d_lu_solve(mp_obj_t self_obj, mp_obj_t b_obj);
mp_obj_t array2d_lu_inv(mp_obj_t self_obj);

#endif
