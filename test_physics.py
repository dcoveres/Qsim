"""Тесты корректности физики симулятора.

В исходном test_circuit.py не было НИ ОДНОГО теста, проверяющего, что
гейты реально делают то, что должны (только тайминг измерений и
отрисовка). Эти тесты сверяют результаты Circuit/State с независимо
построенными эталонными матрицами (через numpy.kron), чтобы поймать
возможные ошибки в конвенции индексации кубитов, транспонировании осей
и т.п., если они когда-нибудь появятся при доработке кода.
"""
import unittest
import numpy as np

try:
    from .circuit import Circuit
    from .state import State
    from .simulator import Simulator
    from . import gates as G
except ImportError:
    from circuit import Circuit
    from state import State
    from simulator import Simulator
    import gates as G


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


class TestSingleQubitGates(unittest.TestCase):
    def test_h_on_qubit0_of_two(self):
        # H на кубите 0 (старший бит), I на кубите 1: H ⊗ I
        expected = kron_all([G.H, G.I]) @ np.eye(4)[0]
        state = Circuit(2).h(0).run()
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_h_on_qubit1_of_two(self):
        expected = kron_all([G.I, G.H]) @ np.eye(4)[0]
        state = Circuit(2).h(1).run()
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_x_flips_correct_bit(self):
        state = Circuit(3).x(1).run()
        # x(1) should give |010>
        expected = np.zeros(8, dtype=complex)
        expected[0b010] = 1.0
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_rz_matches_matrix(self):
        theta = 0.73
        expected = kron_all([G.Rz(theta), G.I]) @ np.eye(4)[0]
        state = Circuit(2).rz(theta, 0).run()
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)


class TestTwoQubitGates(unittest.TestCase):
    def test_bell_state(self):
        state = Circuit(2).h(0).cx(0, 1).run()
        expected = np.zeros(4, dtype=complex)
        expected[0b00] = 1 / np.sqrt(2)
        expected[0b11] = 1 / np.sqrt(2)
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_cx_reversed_control_target(self):
        # cx(1, 0): control is qubit 1 (LSB), target qubit 0 (MSB)
        state = Circuit(2).x(1).cx(1, 0).run()
        # start |01>, control(q1)=1 -> flips target q0 -> |11>
        expected = np.zeros(4, dtype=complex)
        expected[0b11] = 1.0
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_swap(self):
        state = Circuit(2).x(0).swap(0, 1).run()
        expected = np.zeros(4, dtype=complex)
        expected[0b01] = 1.0  # bit moved from q0 to q1
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_cz_phase(self):
        state = Circuit(2).h(0).h(1).cz(0, 1).run()
        # CZ|++> has a relative minus sign on |11>
        expected = np.array([1, 1, 1, -1], dtype=complex) / 2
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)


class TestMultiControlledGates(unittest.TestCase):
    def test_toffoli(self):
        state = Circuit(3).x(0).x(1).ccx(0, 1, 2).run()
        expected = np.zeros(8, dtype=complex)
        expected[0b111] = 1.0
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_toffoli_does_not_fire_with_one_control(self):
        state = Circuit(3).x(0).ccx(0, 1, 2).run()
        expected = np.zeros(8, dtype=complex)
        expected[0b100] = 1.0
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_fredkin(self):
        state = Circuit(3).x(0).x(1).cswap(0, 1, 2).run()
        expected = np.zeros(8, dtype=complex)
        expected[0b101] = 1.0
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_mcx_five_controls(self):
        n = 6
        c = Circuit(n)
        for q in range(5):
            c.x(q)
        c.mcx(list(range(5)), 5)
        state = c.run()
        idx = int(np.argmax(np.abs(state.vector)))
        self.assertEqual(format(idx, f'0{n}b'), '1' * n)

    def test_mcx_does_not_fire_if_any_control_is_zero(self):
        n = 4
        c = Circuit(n).x(0).x(1)  # q2 stays 0
        c.mcx([0, 1, 2], 3)
        state = c.run()
        idx = int(np.argmax(np.abs(state.vector)))
        self.assertEqual(format(idx, f'0{n}b'), '1100')

    def test_arbitrary_permutation_of_controls_and_targets(self):
        """Регрессионный тест на сдвиг осей в apply_controlled: контроли и
        таргет НЕ идут подряд по возрастанию индекса."""
        n = 5
        c = Circuit(n).x(0).x(3)  # controls at 0 and 3
        c.control(G.X, [0, 3], 1, label='X')  # target sits between the controls
        state = c.run()
        idx = int(np.argmax(np.abs(state.vector)))
        # q0=1, q3=1 (both controls on) -> q1 flips 0->1; q2, q4 stay 0.
        self.assertEqual(format(idx, f'0{n}b'), '11010')


class TestAgainstBruteForceReference(unittest.TestCase):
    """Сверяет apply_gate/apply_controlled с независимо построенной
    полной unitary-матрицей на случайных перестановках кубитов —
    ловит любую ошибку в конвенции индексации/транспонирования, которая
    не была бы поймана точечными тестами выше."""

    @staticmethod
    def _random_unitary(dim, rng):
        z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        q, r = np.linalg.qr(z)
        ph = np.diagonal(r) / np.abs(np.diagonal(r))
        return q * ph

    def test_apply_gate_matches_manual_transpose_reference(self):
        rng = np.random.default_rng(42)
        n = 4
        for _ in range(10):
            qubits = list(rng.choice(n, size=2, replace=False))
            mat = self._random_unitary(4, rng)
            v0 = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
            v0 /= np.linalg.norm(v0)

            s = State(n)
            s.vector = v0.copy()
            s.apply_gate(mat, qubits)

            others = [i for i in range(n) if i not in qubits]
            order = others + qubits
            t = v0.reshape([2] * n)
            t = np.transpose(t, order)
            shape = t.shape
            t = t.reshape(-1, 4) @ mat.T
            t = t.reshape(shape)
            expected = np.transpose(t, np.argsort(order)).reshape(-1)

            np.testing.assert_allclose(s.vector, expected, atol=1e-10)

    def test_apply_controlled_matches_full_matrix_reference(self):
        rng = np.random.default_rng(7)
        n = 4
        for _ in range(10):
            all_q = list(range(n))
            rng.shuffle(all_q)
            controls, targets = all_q[:2], [all_q[2]]
            mat = self._random_unitary(2, rng)
            v0 = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
            v0 /= np.linalg.norm(v0)

            s = State(n)
            s.vector = v0.copy()
            s.apply_controlled(mat, controls, targets)

            dim = 2 ** n
            full = np.zeros((dim, dim), dtype=complex)
            for idx in range(dim):
                bits = [(idx >> (n - 1 - b)) & 1 for b in range(n)]
                if all(bits[c] == 1 for c in controls):
                    in_val = bits[targets[0]]
                    for out_val in range(2):
                        out_bits = bits[:]
                        out_bits[targets[0]] = out_val
                        out_idx = 0
                        for b in out_bits:
                            out_idx = (out_idx << 1) | b
                        full[out_idx, idx] = mat[out_val, in_val]
                else:
                    full[idx, idx] = 1.0
            expected = full @ v0
            np.testing.assert_allclose(s.vector, expected, atol=1e-10)


class TestSamplingStatistics(unittest.TestCase):
    def test_ghz_only_produces_all_zeros_or_all_ones(self):
        c = Circuit(5, seed=0).h(0)
        for i in range(4):
            c.cx(i, i + 1)
        c.measure_all()
        counts, _ = c.run(shots=2000)
        self.assertEqual(set(counts.keys()) - {'00000', '11111'}, set())
        self.assertAlmostEqual(counts.get('00000', 0) / 2000, 0.5, delta=0.08)

    def test_probabilities_sum_to_one(self):
        n = 10
        c = Circuit(n, seed=1)
        for q in range(n):
            c.h(q)
        for q in range(n - 1):
            c.cx(q, q + 1)
        state = c.run()
        self.assertAlmostEqual(state.probabilities().sum(), 1.0, places=8)


if __name__ == '__main__':
    unittest.main()
