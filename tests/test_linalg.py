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

    def _test_lu(self):

        import math

        A = np.array([[3, 17, 10], [2, 4, -2], [6, 18, -12]])
        LU = np.array([[6, 18, -12],[1/2,8,16],[1/3,-1/4,6]])
        P = np.array([2,0,1], dtype=np.uint16)

        # lu decomposition
        lu, p = np.lu(A)
        for i, ei in enumerate(lu):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, LU[i][j])

        for i, ei in enumerate(p):
            self.assertEqual(ei, P[i])

        # determinant
        self.assertEqual(np.det(A), 6*8*6)

        # zero determinant, singular
        AA = np.array([[3, 17, 10], [2, 4, -2], [3, 17, 10]])
        self.assertEqual(np.det(AA), 0)

        # vector solve
        b = np.array([1,-1,0])
        x = np.solve(A, b)
        X = [-1.375,  0.375, -0.125]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        res = np.dot(A, x) - b
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        # vector solve. singular
        with self.assertRaises(ValueError):
            x = np.solve(AA, b)

        # matrix solve
        b = np.array([[1, 0], [-1,1], [0, 0]])
        x = np.solve(A, b)
        X = [[-1.375, 4/3], [0.375, -1/3], [-0.125, 1/6]]
        for i, ei in enumerate(x):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, X[i][j])

        res = np.dot(A, x) - b
        for i, ei in enumerate(res):
            for j, eij in enumerate(ei):
                self.assertTrue(math.fabs(eij) < 1e-6)

        # matrix inverse
        Ai = np.inv(A)
        I = np.eye(3)
        res = np.dot(A, Ai) - I
        for i, ei in enumerate(res):
            for j, eij in enumerate(ei):
                self.assertTrue(math.fabs(eij) < 1e-6)

        res = np.dot(Ai, A) - I
        for i, ei in enumerate(res):
            for j, eij in enumerate(ei):
                self.assertTrue(math.fabs(eij) < 1e-6)

        Ais = np.solve(A, I)
        res = Ai - Ais
        for i, ei in enumerate(res):
            for j, eij in enumerate(ei):
                self.assertTrue(math.fabs(eij) < 1e-6)

        # inverse. singular
        with self.assertRaises(ValueError):
            Ai = np.inv(AA)

    def test_qr(self):

        import math

        A = np.array([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
        QR = np.array([[2, 3, 2],[-1, 5, -2], [-1, 0, 4], [-1, 1, 0]])

        # qr decomposition
        qr = np.qr(A)
        print(qr)
        for i, ei in enumerate(qr):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, QR[i][j])

if __name__ == "__main__":
    unittest.main()
