import unittest

import numpy as np


class TestUnittestAssertions(unittest.TestCase):

    def assertEqual(self, a, b, dtype=np.float_):
        if dtype==np.bool_:
            return super().assertEqual(a, b != 0)
        elif dtype==np.int8:
            c = b % 256
            return super().assertEqual(a, c if c < 128 else c - 256)
        elif dtype==np.uint8:
            return super().assertEqual(a, b % 256)
        elif dtype==np.int16:
            c = b % 65536
            return super().assertEqual(a, c if c < 32768 else c - 65536)
        elif dtype==np.uint16:
            return super().assertEqual(a, b % 65536)
        else:
            return super().assertEqual(a, b)

    def run_test_basic(self, dtype):

        # creation
        data = [1, 2, 3]
        a = np.array([data], dtype=dtype)
        b = np.array([[1], [2], [3]], dtype=dtype)
        c = np.array(data, dtype=dtype)
        d = np.array([data, data], dtype=dtype)
        e = np.array([[data], [data]], dtype=dtype)

        # shape
        self.assertEqual(a.shape(), (1, 3))
        self.assertEqual(b.shape(), (3, 1))
        self.assertEqual(c.shape(), (3,))
        self.assertEqual(d.shape(), (2, 3))
        self.assertEqual(e.shape(), (2, 1, 3))

        # shape method
        self.assertEqual(np.shape(a), (1, 3))
        self.assertEqual(np.shape(b), (3, 1))
        self.assertEqual(np.shape(c), (3,))
        self.assertEqual(np.shape(d), (2, 3))
        self.assertEqual(np.shape(e), (2, 1, 3))

        # size
        self.assertEqual(a.size(), 3)
        self.assertEqual(b.size(), 3)
        self.assertEqual(c.size(), 3)
        self.assertEqual(d.size(), 6)
        self.assertEqual(e.size(), 6)

        # size method
        self.assertEqual(np.size(a), 3)
        self.assertEqual(np.size(b), 3)
        self.assertEqual(np.size(c), 3)
        self.assertEqual(np.size(d), 6)
        self.assertEqual(np.size(e), 6)

        # ndim
        self.assertEqual(a.ndim(), 2)
        self.assertEqual(b.ndim(), 2)
        self.assertEqual(c.ndim(), 1)
        self.assertEqual(d.ndim(), 2)
        self.assertEqual(e.ndim(), 3)

        # ndim method
        self.assertEqual(np.ndim(a), 2)
        self.assertEqual(np.ndim(b), 2)
        self.assertEqual(np.ndim(c), 1)
        self.assertEqual(np.ndim(d), 2)
        self.assertEqual(np.ndim(e), 3)

        # len
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 3)
        self.assertEqual(len(c), 3)
        self.assertEqual(len(d), 2)
        self.assertEqual(len(e), 2)

        # copy
        f = a.copy()
        self.assertFalse(f is a)
        self.assertEqual(f.shape(), (1, 3))


        f = b.copy()
        self.assertFalse(f is b)
        self.assertEqual(f.shape(), (3, 1))

        f = c.copy()
        self.assertFalse(f is c)
        self.assertEqual(f.shape(), (3,))

        f = d.copy()
        self.assertFalse(f is d)
        self.assertEqual(f.shape(), (2, 3))

        f = e.copy()
        self.assertFalse(f is d)
        self.assertEqual(f.shape(), (2, 1, 3))

        # flatten
        f = a.flatten()
        self.assertEqual(f.shape(), (3,))
        f = b.flatten()
        self.assertEqual(f.shape(), (3,))
        f = c.flatten()
        self.assertEqual(f.shape(), (3,))
        f = d.flatten()
        self.assertEqual(f.shape(), (6,))
        f = e.flatten()
        self.assertEqual(f.shape(), (6,))

        # reshape
        f = a.reshape(-1)
        self.assertEqual(f.shape(), (3,))
        f = b.reshape(-1)
        self.assertEqual(f.shape(), (3,))
        f = c.reshape(-1)
        self.assertEqual(f.shape(), (3,))
        f = d.reshape(-1)
        self.assertEqual(f.shape(), (6,))
        f = e.reshape(-1)
        self.assertEqual(f.shape(), (6,))

        f = a.reshape((3,1))
        self.assertEqual(f.shape(), (3,1))
        f = b.reshape((1,3))
        self.assertEqual(f.shape(), (1,3))
        f = c.reshape((3,1))
        self.assertEqual(f.shape(), (3,1))
        f = d.reshape((3,2))
        self.assertEqual(f.shape(), (3,2))
        f = d.reshape((3,1,2))
        self.assertEqual(f.shape(), (3,1,2))
        f = e.reshape((3,2,1))
        self.assertEqual(f.shape(), (3,2,1))
        f = e.reshape((3,2))
        self.assertEqual(f.shape(), (3,2))

        with self.assertRaises(ValueError):
            f = e.reshape((3,3))

        # transpose
        f = a.transpose()
        self.assertEqual(f.shape(), (3,1))
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, a[j,i])

        f = b.transpose()
        self.assertEqual(f.shape(), (1,3))
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, b[j,i])

        f = c.transpose()
        self.assertEqual(f.shape(), (3,))
        for i, ei in enumerate(f):
            self.assertEqual(ei, c[i])

        f = d.transpose()
        self.assertEqual(f.shape(), (3,2))
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, d[j,i])

        f = e.transpose()
        self.assertEqual(f.shape(), (3,1,2))
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, e[k,j,i])

        # astype
        f = a.astype(np.float_)
        # self.assertSame(e, a)

        # 1d iterator
        m = data
        for i, ei in enumerate(c):
            self.assertEqual(ei, m[i])

        # 2d iterator
        m = [data, data]
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        # 3d iterator
        m = [[data], [data]]
        for i, ei in enumerate(e):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, m[i][j][k])

        # 1d_slice

        # creation
        data = list(range(5))
        c = np.array(data, dtype=dtype)

        # 1D indexing
        for i in range(5):
            self.assertEqual(c[i], data[i])

        # negative slice
        c[-1]

        # slices
        d = c[:3]
        dd = data[:3]
        for i, e in enumerate(d):
            self.assertEqual(e, dd[i])

        d = c[1:4]
        dd = data[1:4]
        for i, e in enumerate(d):
            self.assertEqual(e, dd[i])

        d = c[::2]
        dd = data[::2]
        for i, e in enumerate(d):
            self.assertEqual(e, dd[i])

        d = c[::-1]
        dd = data[::-1]
        for i, e in enumerate(d):
            self.assertEqual(e, dd[i])

        d = c[:0:-1]
        dd = data[:0:-1]
        for i, e in enumerate(d):
            self.assertEqual(e, dd[i])

        d = c[3:2:-1]
        dd = data[3:2:-1]
        for i, e in enumerate(d):
            self.assertEqual(e, dd[i])

        # 2d_slice

        # creation
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        c = np.array(data, dtype=dtype)

        # indexing
        for i in range(2):
            for j in range(3):
                self.assertEqual(c[i, j], data[i][j])

        # 2D slices

        # slices
        d = c[0]
        dd = data[0]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i])

        d = c[1]
        dd = data[1]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i])

        d = c[:, 2]
        dd = [3, 6, 9, 12]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i])

        d = c[2:3]
        dd = data[2:3]
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, dd[i][j])

        d = c[::2]
        dd = data[::2]
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, dd[i][j])

        d = c[::-1]
        dd = data[::-1]
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, dd[i][j])

        d = c[:, ::2]
        dd = [[1, 3], [4, 6], [7, 9], [10, 12]]
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, dd[i][j])

        # zeros
        m = np.zeros(3, dtype=dtype)
        self.assertEqual(m.shape(), (3,))
        for i, ei in enumerate(m):
            self.assertEqual(ei, 0)

        m = np.zeros((3,), dtype=dtype)
        self.assertEqual(m.shape(), (3,))
        for i, ei in enumerate(m):
            self.assertEqual(ei, 0)

        m = np.zeros((3, 4), dtype=dtype)
        self.assertEqual(m.shape(), (3, 4))
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, 0)

        m = np.zeros((3, 4, 5), dtype=dtype)
        self.assertEqual(m.shape(), (3, 4, 5))
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, 0)

        # ones
        m = np.ones((3,), dtype=dtype)
        self.assertEqual(m.shape(), (3,))
        for i, ei in enumerate(m):
            self.assertEqual(ei, 1)

        m = np.ones((3, 4), dtype=dtype)
        self.assertEqual(m.shape(), (3, 4))
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, 1)

        m = np.ones((3, 4, 5), dtype=dtype)
        self.assertEqual(m.shape(), (3, 4, 5))
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, 1)

        # full
        m = np.full((3,), 3, dtype=dtype)
        self.assertEqual(m.shape(), (3,))
        for i, ei in enumerate(m):
            self.assertEqual(ei, 3)

        m = np.full((3, 4), 3, dtype=dtype)
        self.assertEqual(m.shape(), (3, 4))
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, 3)

        m = np.full((3, 4, 5), 3, dtype=dtype)
        self.assertEqual(m.shape(), (3, 4, 5))
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, 3)

        # eye
        mm = [[1,0,0],[0,1,0],[0,0,1]]

        m = np.eye(3, dtype=dtype)
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j])

        m = np.eye(3, 3, dtype=dtype)
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j])

        mm = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]]

        m = np.eye(3, 5, dtype=dtype)
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j])

        mm = [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]

        m = np.eye(3, 5, 1, dtype=dtype)
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j])

        mm = [[0,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0]]

        m = np.eye(3, 5, -1, dtype=dtype)
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j])

        # slice assignment
        m = np.zeros((3,), dtype=dtype)

        for i, ei in enumerate(m):
            m[i] = i
        for i, ei in enumerate(m):
            self.assertEqual(ei, i)

        mm = m.copy()
        mm[::-1] = m
        for i, ei in enumerate(mm):
            self.assertEqual(ei, 2 - i)

        m = np.zeros((3, 4), dtype=dtype)
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                m[i, j] = i + j
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, i + j)

        mm = m.copy()
        mm[::-1] = m
        for i, ei in enumerate(mm):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, 2 - i + j)

        mm = m.copy()
        mm[:, ::-1] = m
        for i, ei in enumerate(mm):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, i + 3 - j)

        m = np.zeros((3, 4, 5), dtype=dtype)
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    m[i, j, k] = i + j - k
        for i, ei in enumerate(m):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, i + j - k, dtype)

        mm = m.copy()
        mm[::-1] = m
        for i, ei in enumerate(mm):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, 2 - i + j - k, dtype)

        mm = m.copy()
        mm[:, ::-1] = m
        for i, ei in enumerate(mm):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, i + 3 - j - k, dtype)

        mm = m.copy()
        mm[:, :, ::-1] = m
        for i, ei in enumerate(mm):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, i + j - 4 + k, dtype)

        # test_concatenate

        data = [1, 2, 3]
        c = np.array(data, dtype=dtype)
        d = np.array([data, data], dtype=dtype)
        e = np.array([[data], [data]], dtype=dtype)

        f = np.concatenate((c, c + 1))
        m = data + [i+1 for i in data]
        for i, ei in enumerate(f):
            self.assertEqual(ei, m[i])

        f = np.concatenate((d, d + 1))
        m = [data, data, [i+1 for i in data], [i+1 for i in data]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        f = np.concatenate((d, d+1), axis=1)
        m = [data + [i+1 for i in data], data + [i+1 for i in data]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        f = np.concatenate((e, e + 1))
        m = [[data], [data], [[i+1 for i in data]], [[i+1 for i in data]]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, m[i][j][k])

        f = np.concatenate((e, e+1), axis=1)
        m = [[data] + [[i+1 for i in data]], [data] + [[i+1 for i in data]]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, m[i][j][k])

        f = np.concatenate((e, e+1), axis=2)
        m = [[data + [i+1 for i in data]], [data + [i+1 for i in data]]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, m[i][j][k])

        # hstack
        f = np.hstack((c, c))
        m = data + data
        for i, ei in enumerate(f):
            self.assertEqual(ei, m[i])

        f = np.hstack((c, c + 1))
        m = data + [i+1 for i in data]
        for i, ei in enumerate(f):
            self.assertEqual(ei, m[i])

        f = np.hstack((d, d+1))
        m = [data + [i+1 for i in data], data + [i+1 for i in data]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        # vstack
        f = np.vstack((c, c))
        m = [data, data]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        f = np.vstack((c, c+1))
        m = [data, [i+1 for i in data]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        f = np.vstack((c, d, c+1))
        m = [data, data, data, [i+1 for i in data]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

    def test_basic(self):
        self.run_test_basic(np.float_)
        self.run_test_basic(np.int8)
        self.run_test_basic(np.uint8)
        self.run_test_basic(np.int16)
        self.run_test_basic(np.uint16)
        # self.run_test_basic(np.bool_)

    def run_test_scalar_ops_1(self, dtype):

        # creation
        data = [0, 2, 3]
        c = np.array(data, dtype=dtype)

        # all
        self.assertFalse(c.all())

        d = c + 1
        self.assertTrue(d.all())

        # method all
        self.assertFalse(np.all(c))

        d = c + 1
        self.assertTrue(np.all(d))

        # any
        self.assertTrue(c.any())

        d = 0 * c
        self.assertFalse(d.any())

        # method any
        self.assertTrue(np.any(c))

        d = 0 * c
        self.assertFalse(np.any(d))

        # axis
        self.assertFalse(c.all(axis=0))
        d = c + 1
        self.assertTrue(d.all(axis=0))
        self.assertTrue(c.any(axis=0))
        d = 0 * c
        self.assertFalse(d.any(axis=0))

        # axis with methods
        self.assertFalse(np.all(c, axis=0))
        d = c + 1
        self.assertTrue(np.all(d, axis=0))
        self.assertTrue(np.any(c, axis=0))
        d = 0 * c
        self.assertFalse(np.any(d, axis=0))

        # slices
        data = data + data
        c = np.array(data, dtype=dtype)

        # all
        self.assertFalse(c[::2].all())
        self.assertTrue(c[1::3].all())

        d = c + 1
        self.assertTrue(d[::2].all())

        # any
        self.assertTrue(c[::2].any())
        self.assertFalse(c[::3].any())

        d = 0 * c
        self.assertFalse(d[::2].any())

        # axis
        self.assertFalse(c[::2].all(axis=0))
        self.assertTrue(c[1::3].all(axis=0))
        d = c + 1
        self.assertTrue(d[::2].all(axis=0))
        self.assertTrue(c[::2].any(axis=0))
        self.assertFalse(c[::3].any(axis=0))
        d = 0 * c
        self.assertFalse(d[::2].any(axis=0))

    def test_scalar_ops_1(self):
        self.run_test_scalar_ops_1(np.float_)
        self.run_test_scalar_ops_1(np.int8)
        self.run_test_scalar_ops_1(np.uint8)
        self.run_test_scalar_ops_1(np.int16)
        self.run_test_scalar_ops_1(np.uint16)
        self.run_test_scalar_ops_1(np.bool_)

    def run_test_scalar_ops_2(self, dtype):

        # creation
        data = [0, 2, 3]
        c = np.array([data, [i+1 for i in data]], dtype=dtype)

        # max, min
        self.assertEqual(c.max(), 4, dtype)
        self.assertEqual(c.min(), 0, dtype)

        # sum, prod
        self.assertEqual(c.sum(), 13, dtype)
        self.assertEqual(c.prod(), 0, dtype)

        d = (c + 1).astype(dtype)
        self.assertEqual(d.max(), 5, dtype)
        self.assertEqual(d.min(), 1, dtype)
        self.assertEqual(d.sum(), 19, dtype)
        self.assertEqual(d.prod(), 12*2*4*5, dtype)

        # axis
        d = c.max(axis=0)
        dd = [1, 3, 4]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i], dtype)

        d = c.max(axis=1)
        dd = [3, 4]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i], dtype)

        d = c.min(axis=0)
        dd = [0, 2, 3]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i], dtype)

        d = c.min(axis=1)
        dd = [0, 1]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i], dtype)

        d = c.sum(axis=0)
        dd = [1, 5, 7]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i], dtype)

        d = c.sum(axis=1)
        dd = [5, 8]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i], dtype)

        d = c.prod(axis=0)
        dd = [0, 6, 12]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i], dtype)

        d = c.prod(axis=1)
        dd = [0, 12]
        for i, ei in enumerate(d):
            self.assertEqual(ei, dd[i], dtype)

        # slices
        data = data + data
        c = np.array([data, [i+1 for i in data]], dtype=dtype)

        # max, min
        self.assertEqual(c[:, ::2].max(), 4, dtype)
        self.assertEqual(c[:, ::3].max(), 1, dtype)
        self.assertEqual(c[:, 1::3].max(), 3, dtype)

        self.assertEqual(c[:, ::2].min(), 0, dtype)
        self.assertEqual(c[:, ::3].min(), 0, dtype)
        self.assertEqual(c[:, 1::3].min(), 2, dtype)

        # sum, prod
        self.assertEqual(c[:, ::2].sum(), 5+8, dtype)
        self.assertEqual(c[:, 1::2].sum(), 5+8, dtype)
        self.assertEqual(c[:, ::2].prod(), 0, dtype)
        self.assertEqual(c[:, 1::2].prod(), 0, dtype)

        # axis

        d = (c + 1).astype(dtype)

        self.assertEqual(d[:, ::2].max(), 5, dtype)
        self.assertEqual(d[:, ::3].max(), 2, dtype)
        self.assertEqual(d[:, 1::3].max(), 4, dtype)

        self.assertEqual(d[:, ::2].min(), 1, dtype)
        self.assertEqual(d[:, ::3].min(), 1, dtype)
        self.assertEqual(d[:, 1::3].min(), 3, dtype)

        # sum, prod
        self.assertEqual(d[:, ::2].sum(), 8+6+5, dtype)
        self.assertEqual(d[:, 1::2].sum(), 8+6+5, dtype)
        self.assertEqual(d[:, ::2].prod(), 4*3*2*5*4, dtype)
        self.assertEqual(d[:, 1::2].prod(), 4*3*2*5*4, dtype)


    def test_scalar_ops_2(self):
        self.run_test_scalar_ops_2(np.float_)
        self.run_test_scalar_ops_2(np.int8)
        self.run_test_scalar_ops_2(np.uint8)
        self.run_test_scalar_ops_2(np.int16)
        self.run_test_scalar_ops_2(np.uint16)
        self.run_test_scalar_ops_2(np.bool_)

    def test_minimum_max(self):

        # creation
        data1 = [1, -2, 3]
        c = np.array(data1)

        data2 = [0, -2, 5]
        d = np.array(data2)

        e = np.maximum(c, d)
        mm = [i if i > j else j for i, j in zip(data1, data2)]
        for i, ei in enumerate(e):
            self.assertEqual(ei, mm[i])

        e = np.minimum(c, d)
        mm = [i if i < j else j for i, j in zip(data1, data2)]
        for i, ei in enumerate(e):
            self.assertEqual(ei, mm[i])

        # creation
        data1 = [1, -2, 3]
        c = np.array([data1])

        data2 = [0, -2, 5]
        d = np.array([data2])

        e = np.maximum(c, d)
        mm = [[i if i > j else j for i, j in zip(data1, data2)]]
        for i, ei in enumerate(e):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j])

        e = np.minimum(c, d)
        mm = [[i if i < j else j for i, j in zip(data1, data2)]]
        for i, ei in enumerate(e):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j])

    def test_1d_unary(self):

        # creation
        data = [1, -2, 3]
        c = np.array(data)

        # unary -
        d = -c
        for i, e in enumerate(d):
            self.assertEqual(e, -data[i])

        for i, e in enumerate(c):
            self.assertEqual(e, data[i])

        # unary +
        d = +c
        for i, e in enumerate(d):
            self.assertEqual(e, data[i])

        # unary abs
        d = abs(c)
        for i, e in enumerate(d):
            self.assertEqual(e, abs(data[i]))

        # fabs
        d = np.fabs(c)
        for i, e in enumerate(d):
            self.assertEqual(e, abs(data[i]))

        # unary len
        self.assertEqual(len(c), len(data))

        # op with slices

        # creation
        data += data
        c = np.array(data)

        # unary -
        d = -c[::2]
        for i, e in enumerate(d):
            self.assertEqual(e, -data[2 * i])

        for i, e in enumerate(c):
            self.assertEqual(e, data[i])

        # unary +
        d = +c[::2]
        for i, e in enumerate(d):
            self.assertEqual(e, data[2 * i])

        # unary abs
        d = abs(c[::2])
        for i, d in enumerate(d):
            self.assertEqual(d, abs(data[2 * i]))

        # unary len
        self.assertEqual(len(c[::2]), len(data[::2]))

    def test_2d_unary(self):

        # creation
        data = [1, -2, 3]
        data = [data, data]
        c = np.array(data)

        # unary -
        d = -c
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, -data[i][j])

        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, data[i][j])

        # unary +
        d = +c
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, data[i][j])

        # unary abs
        d = abs(c)
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, abs(data[i][j]))

        # unary len
        self.assertEqual(len(c), len(data))

        # op with slices

        # creation
        data = [1, -2, 3]
        data += data
        data = [data, data]
        c = np.array(data)

        # unary -
        d = -c[:, ::2]
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, -data[i][2 * j])

        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, data[i][j])

        # unary +
        d = +c
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, data[i][j])

        # unary abs
        d = abs(c)
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, abs(data[i][j]))

        # unary len
        self.assertEqual(len(c[:1]), len(data[:1]))

    def test_1d_binary_arithmetic(self):

        # creation
        data1 = [1, 2, 3, 4, 5]
        data2 = [-1, 6, -7, 0, 5]

        a = np.array(data1)
        b = np.array(data2)

        c = a * b
        d = [i * j for i, j in zip(data1, data2)]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a + b
        d = [i + j for i, j in zip(data1, data2)]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a - b
        d = [i - j for i, j in zip(data1, data2)]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = b / a
        d = [j / i for i, j in zip(data1, data2)]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a * 3
        d = [3 * i for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = 3 * a
        d = [3 * i for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a + 3
        d = [3 + i for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = 3 + a
        d = [3 + i for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a - 3
        d = [i - 3 for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = 3 - a
        d = [3 - i for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = 3 / a
        d = [3 / i for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a / 3
        d = [i / 3 for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        # op with slices

        c = a[::2] * b[::2]
        d = [i * j for i, j in zip(data1[::2], data2[::2])]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

    def test_1d_binary_logic(self):

        # creation
        data1 = [1, 2, 3, 4, 5]
        data2 = [-1, 6, -7, 0, 5]

        a = np.array(data1)
        b = np.array(data2)

        c = a == b
        d = [i == j for i, j in zip(data1, data2)]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a != b
        d = [i != j for i, j in zip(data1, data2)]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a < b
        d = [i < j for i, j in zip(data1, data2)]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a > b
        d = [i > j for i, j in zip(data1, data2)]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a <= b
        d = [i <= j for i, j in zip(data1, data2)]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a >= b
        d = [i >= j for i, j in zip(data1, data2)]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a == 3
        d = [i == 3 for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a != 3
        d = [i != 3 for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a < 3
        d = [i < 3 for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a > 3
        d = [i > 3 for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a <= 3
        d = [i <= 3 for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        c = a >= 3
        d = [i >= 3 for i in data1]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        # op with slices

        c = a[::2] == b[::2]
        d = [i == j for i, j in zip(data1[::2], data2[::2])]
        for i, ei in enumerate(c):
            self.assertEqual(ei, d[i])

        # array equal
        self.assertTrue(np.array_equal(a, a))
        self.assertTrue(np.array_equal(b, b))
        self.assertFalse(np.array_equal(a, b))
        self.assertFalse(np.array_equal(b, a))

        c = np.array(data2 + data2)
        self.assertFalse(np.array_equal(c, a))
        self.assertFalse(np.array_equal(c, 1))
        self.assertFalse(np.array_equal(np.array([1]), 1))
        self.assertFalse(np.array_equal(np.array([1]), 1))
        self.assertTrue(np.array_equal(1, 1))
        self.assertFalse(np.array_equal(1, 2))

    def test_2d_binary_arithmetic(self):

        # creation
        data1 = [1, 2, 3, 4, 5]
        data2 = [-1, 6, -7, 0, 5]

        m1 = [data1, [i + 2 for i in data2]]
        m2 = [[i + 2 for i in data2], data1]

        a = np.array(m1)
        b = np.array(m2)

        c = a * b
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i * j for i, j in zip(*rows)])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a + b
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i + j for i, j in zip(*rows)])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a - b
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i - j for i, j in zip(*rows)])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a / b
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i / j for i, j in zip(*rows)])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a * 3
        m = []
        for i, row in enumerate(m1):
            m.append([i * 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = 3 * a
        m = []
        for i, row in enumerate(m1):
            m.append([i * 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a + 3
        m = []
        for i, row in enumerate(m1):
            m.append([i + 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = 3 + a
        m = []
        for i, row in enumerate(m1):
            m.append([i + 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a - 3
        m = []
        for i, row in enumerate(m1):
            m.append([i - 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = 3 - a
        m = []
        for i, row in enumerate(m1):
            m.append([3 - i for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a / 3
        m = []
        for i, row in enumerate(m1):
            m.append([i / 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = 3 / a
        m = []
        for i, row in enumerate(m1):
            m.append([3 / i for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        # op with slices

        c = a[:, ::2] * b[:, ::2]
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i * j for i, j in zip(*rows)])
        mm = []
        for row in m:
            mm.append(row[::2])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j])

    def test_2d_binary_logic(self):

        # creation
        data1 = [1, 2, 3, 4, 5]
        data2 = [-1, 6, -7, 0, 5]

        m1 = [data1, [i + 2 for i in data2]]
        m2 = [[i + 2 for i in data2], data1]

        a = np.array(m1)
        b = np.array(m2)

        c = a == b
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i == j for i, j in zip(*rows)])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a != b
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i != j for i, j in zip(*rows)])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a < b
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i < j for i, j in zip(*rows)])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a > b
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i > j for i, j in zip(*rows)])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a <= b
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i <= j for i, j in zip(*rows)])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a >= b
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i >= j for i, j in zip(*rows)])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a == 3
        m = []
        for i, row in enumerate(m1):
            m.append([i == 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a != 3
        m = []
        for i, row in enumerate(m1):
            m.append([i != 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a < 3
        m = []
        for i, row in enumerate(m1):
            m.append([i < 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a > 3
        m = []
        for i, row in enumerate(m1):
            m.append([i > 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a <= 3
        m = []
        for i, row in enumerate(m1):
            m.append([i <= 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        c = a >= 3
        m = []
        for i, row in enumerate(m1):
            m.append([i >= 3 for i in row])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        # op with slices

        c = a[:, ::2] < b[:, ::2]
        m = []
        for i, rows in enumerate(zip(m1, m2)):
            m.append([i < j for i, j in zip(*rows)])
        mm = []
        for row in m:
            mm.append(row[::2])
        for i, ei in enumerate(c):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j])

        # array equal
        self.assertTrue(np.array_equal(a, a))
        self.assertTrue(np.array_equal(b, b))
        self.assertFalse(np.array_equal(a, b))
        self.assertFalse(np.array_equal(b, a))

        c = np.array(m2 + m2)
        self.assertFalse(np.array_equal(c, a))

    def test_unary_fun(self):

        import math

        data = [1., 2., 3.]
        c = np.array(data)

        d = np.acos(c/3)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.acos(data[i]/3))

        d = np.asin(c/3)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.asin(data[i]/3))

        d = np.atan(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.atan(data[i]))

        d = np.sin(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.sin(data[i]))

        d = np.cos(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.cos(data[i]))

        d = np.tan(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.tan(data[i]))

        d = np.acosh(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.acosh(data[i]))

        d = np.asinh(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.asinh(data[i]))

        d = np.atanh(c/3.1)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.atanh(data[i]/3.1))

        d = np.sinh(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.sinh(data[i]))

        d = np.cosh(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.cosh(data[i]))

        d = np.tanh(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.tanh(data[i]))

        d = np.ceil(c*2.7)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.ceil(data[i]*2.7))

        d = np.floor(c*2.7)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.floor(data[i]*2.7))

        d = np.erf(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.erf(data[i]))

        d = np.erfc(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.erfc(data[i]))

        d = np.exp(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.exp(data[i]))

        d = np.expm1(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.expm1(data[i]))

        d = np.gamma(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.gamma(data[i]))

        d = np.lgamma(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.lgamma(data[i]))

        d = np.log(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.log(data[i]))

        d = np.log10(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.log10(data[i]))

        d = np.log2(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.log2(data[i]))

        d = np.sqrt(c)
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.sqrt(data[i]))

        # slices
        data = [1., 2., 3.]
        c = np.array(data + data)

        d = np.cos(c[::2])
        mm = data + data
        for i, ei in enumerate(d):
            self.assertEqual(ei, math.cos(mm[2*i]))

        # 2d array
        data = [1., 2., 3.]
        c = np.array([data, data])

        d = np.cos(c)
        mm = [data, data]
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, math.cos(mm[i][j]))

        # 2d array slices
        data = [1., 2., 3.]
        c = np.array([data + data, data + data])

        d = np.cos(c[:,::2])
        mm = [data + data, data + data]
        for i, ei in enumerate(d):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, math.cos(mm[i][2*j]))

    def run_test_dot(self, dtype=np.float_):

        # 1D
        data = [1, 2, 3]
        c = np.array(data, dtype=dtype)

        data = [1, 2, 3, 4]
        d = np.array(data, dtype=dtype)

        f = np.dot(c, c)
        self.assertEqual(f, 1 + 4 + 9, dtype)

        f = c.dot(c)
        self.assertEqual(f, 1 + 4 + 9, dtype)

        with self.assertRaises(ValueError):
            np.dot(c, d)

        with self.assertRaises(ValueError):
            c.dot(d)

        # 2D x 1D
        data = [1, 2, 3]
        c = np.array([data, [i+1 for i in data]], dtype=dtype)

        data = [1, 2, 3]
        d = np.array(data, dtype=dtype)

        f = np.dot(c, d)
        mm = [1+4+9, 2+6+12]
        for i, ei in enumerate(f):
            self.assertEqual(ei, mm[i], dtype)

        e = c.transpose()
        data = [1, -1]
        d = np.array(data, dtype=dtype)

        f = np.dot(e, d)
        mm = [-1, -1, -1]
        for i, ei in enumerate(f):
            self.assertEqual(ei, mm[i], dtype)

        # 2D x 2D
        f = np.dot(e, c)
        mm = [[1+4,2+6,3+8],[2+6,4+9,6+12],[3+8,6+12,9+16]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j], dtype)

        f = np.dot(e, (c + 1).astype(dtype))
        mm = [[2+6,3+8,4+10],[4+9,6+12,8+15],[6+12,9+16,12+20]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j], dtype)

        f = np.dot(e, c[:,:2])
        mm = [[1+4,2+6],[2+6,4+9],[3+8,6+12]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j], dtype)

        f = np.dot(e[:2,:], c)
        mm = [[1+4,2+6,3+8],[2+6,4+9,6+12]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j], dtype)

        # 3D x 1D
        data = [1, 2, 3]
        c = np.array([[data, [i+1 for i in data]]], dtype=dtype)

        data = [1, 2, 3]
        d = np.array(data, dtype=dtype)

        f = np.dot(c, d)
        mm = [[1+4+9, 2+6+12]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j], dtype)

        data = [1, 2]
        c = np.array([[data, [i+1 for i in data], [i-1 for i in data]]], dtype=dtype)

        data = [1, -1]
        d = np.array(data, dtype=dtype)

        f = np.dot(c, d)
        mm = [[-1, -1, -1]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, mm[i][j], dtype)

        # TODO: 3D x 3D

    def test_dot(self):
        self.run_test_dot(np.float_)
        self.run_test_dot(np.int8)
        self.run_test_dot(np.uint8)
        self.run_test_dot(np.int16)
        self.run_test_dot(np.uint16)
        self.run_test_dot(np.bool_)

    def test_flip(self, dtype=np.float_):

        # creation
        data = [1, 2, 3]
        a = np.array([data], dtype=dtype)
        b = np.array([[1], [2], [3]], dtype=dtype)
        c = np.array(data, dtype=dtype)
        d = np.array([data, [i+1 for i in data]], dtype=dtype)
        e = np.array([[data], [[i-1 for i in data]]], dtype=dtype)

        f = np.flip(a)
        m = [i for i in data[::-1]]
        for i, ei in enumerate(f):
            self.assertEqual(ei, m[i])

        f= np.flip(d)
        m = data[::-1][::-1]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        f= np.flip(e)
        m = data[::-1][::-1][::-1]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, m[i][j][k])

        f= np.flip(d, axis=0)
        m = [[i+1 for i in data], data]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        f= np.fliplr(d)
        m = [[i+1 for i in data], data]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        f= np.flip(d, axis=1)
        m = [data[::-1], [i+1 for i in data][::-1]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        f= np.flipud(d)
        m = [data[::-1], [i+1 for i in data][::-1]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, m[i][j])

        f= np.flip(e, axis=0)
        m = [[[i-1 for i in data]], [data]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, m[i][j][k])

        f= np.fliplr(e)
        m = [[[i-1 for i in data]], [data]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, m[i][j][k])

        f= np.flip(e, axis=1)
        m = [[data][::-1], [[i-1 for i in data]][::-1]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, m[i][j][k])

        f= np.flipud(e)
        m = [[data][::-1], [[i-1 for i in data]][::-1]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, m[i][j][k])

        f= np.flip(e, axis=2)
        m = [[data[::-1]], [[i-1 for i in data[::-1]]]]
        for i, ei in enumerate(f):
            for j, eij in enumerate(ei):
                for k, eijk in enumerate(eij):
                    self.assertEqual(eijk, m[i][j][k])

    def test_issubsctype(self):

        data = [1,2,3]
        types = [np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.float_]

        c = np.array(data, np.bool_)
        answer = [True, False, False, False, False, False]
        for i, t in enumerate(types):
            self.assertEqual(np.issubsctype(c, t), answer[i])

        c = np.array(data, np.uint8)
        answer = [False, True, False, False, False, False]
        for i, t in enumerate(types):
            self.assertEqual(np.issubsctype(c, t), answer[i])

        c = np.array(data, np.int8)
        answer = [False, True, True, False, False, False]
        for i, t in enumerate(types):
            self.assertEqual(np.issubsctype(c, t), answer[i])

        c = np.array(data, np.uint16)
        answer = [False, True, False, True, False, False]
        for i, t in enumerate(types):
            self.assertEqual(np.issubsctype(c, t), answer[i])

        c = np.array(data, np.int16)
        answer = [False, True, True, True, True, False]
        for i, t in enumerate(types):
            self.assertEqual(np.issubsctype(c, t), answer[i])

        c = np.array(data, np.float_)
        answer = [False, True, True, True, True, True]
        for i, t in enumerate(types):
            self.assertEqual(np.issubsctype(c, t), answer[i])

if __name__ == "__main__":
    unittest.main()
