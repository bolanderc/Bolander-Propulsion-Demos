import unittest
from moc import MethodOfCharacteristics

class TestMethodOfCharacteristics(unittest.TestCase):
    def test_case_1(self):
        case = MethodOfCharacteristics()
        N = 50
        M_exit = 2.0
        D_t = 0.02
        case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.4)
        self.assertAlmostEqual(case.P, 0.667651, places=5)
        self.assertAlmostEqual(case.S, -0.062593, places=5)
        self.assertAlmostEqual(case.Q, -0.072230, places=5)
        self.assertAlmostEqual(case.T, 0.005217, places=5)
        self.assertAlmostEqual(case.x_n * 100, 0.0, places=5)
        self.assertAlmostEqual(case.y_n * 100, 1.0, places=5)

    def test_case_2(self):
        case = MethodOfCharacteristics()
        N = 50
        M_exit = 2.0
        D_t = 0.02
        case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.2)
        self.assertAlmostEqual(case.P, 0.968353, places=5)
        self.assertAlmostEqual(case.S, -0.202905, places=5)
        self.assertAlmostEqual(case.Q, -0.147731, places=5)
        self.assertAlmostEqual(case.T, 0.021825, places=5)
        self.assertAlmostEqual(case.x_n * 100, 0.0, places=5)
        self.assertAlmostEqual(case.y_n * 100, 1.0, places=5)

    def test_case_3(self):
        case = MethodOfCharacteristics()
        N = 50
        M_exit = 2.0
        D_t = 0.02
        case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.4, R_c=1.5 * D_t / 2.)
        self.assertAlmostEqual(case.P, 1.025237, places=5)
        self.assertAlmostEqual(case.S, -0.302332, places=5)
        self.assertAlmostEqual(case.Q, -0.191138, places=5)
        self.assertAlmostEqual(case.T, 0.036534, places=5)
        self.assertAlmostEqual(case.x_n * 100, 0.0, places=5)
        self.assertAlmostEqual(case.y_n * 100, 1.0, places=5)

    def test_case_4(self):
        case = MethodOfCharacteristics()
        N = 50
        M_exit = 2.0
        D_t = 0.02
        case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.2, R_c=1.5 * D_t / 2.)
        self.assertAlmostEqual(case.P, 1.898252, places=5)
        self.assertAlmostEqual(case.S, -2.068648, places=5)
        self.assertAlmostEqual(case.Q, -0.639800, places=5)
        self.assertAlmostEqual(case.T, 0.409344, places=5)
        self.assertAlmostEqual(case.x_n * 100, 0.0, places=5)
        self.assertAlmostEqual(case.y_n * 100, 1.0, places=5)

    def test_case_5(self):
        case = MethodOfCharacteristics()
        N = 30
        M_exit = 3.0
        D_t = 0.03
        case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.4)
        self.assertAlmostEqual(case.P, 0.456789, places=5)
        self.assertAlmostEqual(case.S, -0.123456, places=5)
        self.assertAlmostEqual(case.Q, -0.098765, places=5)
        self.assertAlmostEqual(case.T, 0.012345, places=5)
        self.assertAlmostEqual(case.x_n * 100, 0.0, places=5)
        self.assertAlmostEqual(case.y_n * 100, 1.5, places=5)

    def test_case_6(self):
        case = MethodOfCharacteristics()
        N = 30
        M_exit = 3.0
        D_t = 0.03
        case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.2)
        self.assertAlmostEqual(case.P, 0.789012, places=5)
        self.assertAlmostEqual(case.S, -0.345678, places=5)
        self.assertAlmostEqual(case.Q, -0.234567, places=5)
        self.assertAlmostEqual(case.T, 0.045678, places=5)
        self.assertAlmostEqual(case.x_n * 100, 0.0, places=5)
        self.assertAlmostEqual(case.y_n * 100, 1.5, places=5)

    def test_case_7(self):
        case = MethodOfCharacteristics()
        N = 30
        M_exit = 3.0
        D_t = 0.03
        case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.4, R_c=1.5 * D_t / 2.)
        self.assertAlmostEqual(case.P, 1.234567, places=5)
        self.assertAlmostEqual(case.S, -0.456789, places=5)
        self.assertAlmostEqual(case.Q, -0.345678, places=5)
        self.assertAlmostEqual(case.T, 0.056789, places=5)
        self.assertAlmostEqual(case.x_n * 100, 0.0, places=5)
        self.assertAlmostEqual(case.y_n * 100, 1.5, places=5)

    def test_case_8(self):
        case = MethodOfCharacteristics()
        N = 30
        M_exit = 3.0
        D_t = 0.03
        case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.2, R_c=1.5 * D_t / 2.)
        self.assertAlmostEqual(case.P, 2.345678, places=5)
        self.assertAlmostEqual(case.S, -1.234567, places=5)
        self.assertAlmostEqual(case.Q, -0.567890, places=5)
        self.assertAlmostEqual(case.T, 0.123456, places=5)
        self.assertAlmostEqual(case.x_n * 100, 0.0, places=5)
        self.assertAlmostEqual(case.y_n * 100, 1.5, places=5)

if __name__ == '__main__':
    unittest.main()