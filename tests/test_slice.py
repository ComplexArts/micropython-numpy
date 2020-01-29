import unittest

import math
import numpy as np

class TestUnittestAssertions(unittest.TestCase):

    def test_slices(self):

        print('--- 2D ARRAYS ---')

        M = 7
        N = 9

        v = np.array(list(range(M * N)))
        A = v.reshape((M, N))

        A0 = v.reshape((M * N))

        print('--- SINGLETON DIM 0 ---')

        # slice = [0:M:1, 0:N:1],   shape = [M, N],  array_shape = [M, N]
        # slice = [j*N:(j+1)*N:1],  shape = [N],     array_shape = [M*N]

        for j in range(M):
            X = A0[j*N:(j+1)*N:1]
            assert np.all(X == A[j, :])
            assert np.shape(X) == (N,)

        print('--- SINGLETON DIM 0 WITH SLICE---')

        # p = ceil((b-a)/c)
        # slice = [0:M:1, a:b:c],  shape = [M, P],  array_shape = [M, N]
        # slice = [a+j*N:b+j*N:c], shape = [p],     array_shape = [M*N]

        b = M - 1
        c = 3
        for a in range(1,b):
            p = math.ceil((b-a)/c)
            for j in range(M):
                X = A0[a+j*N:b+j*N:c]
                assert np.all(X == A[:, a:b:c][j, :])
                assert np.shape(X) == (p,)

        print('--- SINGLETON DIM 1 ---')

        # slice = [0:M:1, 0:N:1],  shape = [M, N], array_shape = [M, N]
        # slice = [j:j+M*N:N],     shape = [N],    array_shape = [M*N]

        for j in range(N):
            X = A0[j:j+M*N:N]
            assert np.all(X == A[:, j])
            assert np.shape(X) == (M,)

        print('--- SINGLETON DIM 1 WITH SLICE ---')

        # P = ceil((b-a)/c)
        # slice = [a:b:c, 0:1:N],     shape = [P, N], array_shape = [M, N]
        # slice = [a*N+j:b*N+j:c*N],  shape = [p],    array_shape = [M*N]

        b = M - 1
        c = 3
        a = 1
        p = math.ceil((b-a)/c)
        for j in range(N):
            X = A0[a*N+j:b*N+j:c*N]
            assert np.all(X == A[a:b:c, :][:, j])
            assert np.shape(X) == (p,)

        print('--- 4D ARRAYS ---')

        P = 7
        Q = 11
        v = np.array(list(range(M * N * P * Q)))

        A = v.reshape((M, N, P, Q))

        print('--- SINGLETON DIM 1 WITH SLICE ---')

        # P = ceil((b-a)/c)
        # slice = [0:M:1, 0:N:1, a:b:c, 0:Q:1],   shape = [M, N, p, Q],      array_shape = [M, N, P, Q]
        # slice = [0:M:1, a+j*P:b+j*P:c, 0:Q,1],  shape = [M, p, Q],         array_shape = [M, N*P, Q]

        A0 = v.reshape((M, N*P, Q))

        p = math.ceil((b-a)/c)
        for j in range(N):
            X = A0[0:M:1, a+j*P:b+j*P:c, 0:Q:1]
            assert np.all(X == A[:, :, a:b:c, :][:, j, :, :])
            assert np.shape(X) == (M, p, Q)

        print('--- SINGLETON DIM 3 WITH SLICE ---')

        # slice = [0:M:1, 0:N:1, a:b:c, 0:Q:1],    shape = [M, N, p, Q],      array_shape = [M, N, P, Q]
        # slice = [0:M:1, 0:N:1, a*Q+j:b*Q+j:c*Q], shape = [M, N, p],         array_shape = [M, N, P*Q]

        A0 = v.reshape((M, N, P*Q))

        p = math.ceil((b-a)/c)
        for j in range(Q):
            X = A0[0:M:1, 0:N:1, a*Q+j:b*Q+j:c*Q]
            assert np.all(X == A[:, :, a:b:c, :][:, :, :, j])
            assert np.shape(X) == (M, N, p)

if __name__ == "__main__":
    unittest.main()
