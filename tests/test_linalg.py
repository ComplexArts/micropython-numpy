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

    def test_lu(self):

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

        # qr decomposition
        A = np.array([[1, -1],[0, 1]])
        QR = np.array([[1, -1], [0, 1]])
        qr = np.qr(A)
        for i, ei in enumerate(qr):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, QR[i][j])

        # vector solve
        b = np.array([1, 0])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)
        X = [1, 0]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([0, 1])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [1, 1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([1, 1])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [2, 1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # qr decomposition
        A = np.array([[-1, -1],[0, 1]])
        QR = np.array([[1, 1], [0, 1]])
        qr = np.qr(A)
        for i, ei in enumerate(qr):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, QR[i][j])

        # vector solve
        b = np.array([1, 0])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)
        X = [-1, 0]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([0, 1])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [-1, 1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([1, 1])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [-2, 1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # qr decomposition
        A = np.array([[1, -1],[0, -1]])
        QR = np.array([[1, -1], [0, 1]])
        qr = np.qr(A)
        for i, ei in enumerate(qr):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, QR[i][j])

        # vector solve
        b = np.array([1, 0])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)
        X = [1, 0]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([0, 1])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [-1, -1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([1, 1])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [0, -1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # qr decomposition
        A = np.array([[1, -1],[1, 1]])
        QR = np.array([[math.sqrt(2), 0], [-1-math.sqrt(2), math.sqrt(2)]])
        qr = np.qr(A)
        for i, ei in enumerate(qr):
            for j, eij in enumerate(ei):
                self.assertTrue(math.fabs(eij - QR[i][j]) < 1e-6)

        # vector solve
        b = np.array([2, 1])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertTrue(math.fabs(ei) < 1e-6)
        X = [3/2, -1/2]
        for i, ei in enumerate(x):
            self.assertTrue(math.fabs(ei - X[i]) < 1e-6)

        # lu example
        A = np.array([[3, 17, 10], [2, 4, -2], [6, 18, -12]])
        qr = np.qr(A)

        # vector solve
        b = np.array([1,-1,0])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertTrue(math.fabs(ei) < 1e-6)
        X = [-1.375,  0.375, -0.125]
        for i, ei in enumerate(x):
            self.assertTrue(math.fabs(ei - X[i]) < 1e-6)

        # qr decomposition
        A = np.array([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
        QR = np.array([[2, 3, 2],[-1, 5, -2], [-1, 0, 4], [-1, 1, 0]])
        qr = np.qr(A)
        for i, ei in enumerate(qr):
            for j, eij in enumerate(ei):
                self.assertEqual(eij, QR[i][j])

        # vector solve
        b = np.array([1,-1,0,0])
        x, res = np.qr_solve(A, b)
        for i, ei in enumerate(res):
            self.assertTrue(math.fabs(ei) < 1e-6)

        X = [-1/10,-1/10,1/4]
        for i, ei in enumerate(x):
            self.assertTrue(math.fabs(ei - X[i]) < 1e-6)

        # vector solve
        b = np.array([1,-1,1,0])
        x, res = np.qr_solve(A, b)
        RES = [1/4,1/4,-1/4,-1/4]
        for i, ei in enumerate(res):
            self.assertTrue(math.fabs(ei - RES[i]) < 1e-6)

        X = [-1/5,1/20,3/8]
        for i, ei in enumerate(x):
            self.assertTrue(math.fabs(ei - X[i]) < 1e-6)

    def test_lstsq(self):

        import math

        # qr decomposition
        A = np.array([[1, -1],[0, 1]])
        b = np.array([1, 0])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)
        X = [1, 0]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([0, 1])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [1, 1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([1, 1])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [2, 1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # qr decomposition
        A = np.array([[-1, -1],[0, 1]])
        b = np.array([1, 0])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)
        X = [-1, 0]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([0, 1])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [-1, 1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([1, 1])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [-2, 1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # qr decomposition
        A = np.array([[1, -1],[0, -1]])
        b = np.array([1, 0])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)
        X = [1, 0]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([0, 1])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [-1, -1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # vector solve
        b = np.array([1, 1])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertEqual(ei, 0)

        X = [0, -1]
        for i, ei in enumerate(x):
            self.assertEqual(ei, X[i])

        # qr decomposition
        A = np.array([[1, -1],[1, 1]])
        b = np.array([2, 1])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertTrue(math.fabs(ei) < 1e-6)
        X = [3/2, -1/2]
        for i, ei in enumerate(x):
            self.assertTrue(math.fabs(ei - X[i]) < 1e-6)

        # lu example
        A = np.array([[3, 17, 10], [2, 4, -2], [6, 18, -12]])
        b = np.array([1,-1,0])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertTrue(math.fabs(ei) < 1e-6)
        X = [-1.375,  0.375, -0.125]
        for i, ei in enumerate(x):
            self.assertTrue(math.fabs(ei - X[i]) < 1e-6)

        # qr decomposition
        A = np.array([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
        b = np.array([1,-1,0,0])
        x, res = np.lstsq(A, b)
        for i, ei in enumerate(res):
            self.assertTrue(math.fabs(ei) < 1e-6)

        X = [-1/10,-1/10,1/4]
        for i, ei in enumerate(x):
            self.assertTrue(math.fabs(ei - X[i]) < 1e-6)

        # vector solve
        b = np.array([1,-1,1,0])
        x, res = np.lstsq(A, b)
        RES = [1/4,1/4,-1/4,-1/4]
        for i, ei in enumerate(res):
            self.assertTrue(math.fabs(ei - RES[i]) < 1e-6)

        X = [-1/5,1/20,3/8]
        for i, ei in enumerate(x):
            self.assertTrue(math.fabs(ei - X[i]) < 1e-6)

if __name__ == "__main__":
    unittest.main()
