"""Регрессионные тесты на баги, найденные при аудите исходной версии.

Каждый тест воспроизводит конкретный баг (были проверены на старой
версии кода перед исправлением) и фиксирует ожидаемое поведение после
фикса, чтобы он не мог тихо вернуться при будущих изменениях.
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


class TestQubitIndexValidation(unittest.TestCase):
    """БАГ: Circuit._check() пропускал дробные и булевы индексы кубитов
    (0 <= 1.5 < n — истина), что приводило к падению не в месте ошибки
    пользователя, а глубоко внутри numpy при run(), с невнятным
    TypeError."""

    def test_fractional_qubit_index_rejected_immediately(self):
        with self.assertRaises(TypeError):
            Circuit(2).h(1.5)

    def test_bool_qubit_index_rejected(self):
        with self.assertRaises(TypeError):
            Circuit(2).h(True)

    def test_out_of_range_qubit_index_rejected(self):
        with self.assertRaises(ValueError):
            Circuit(2).h(5)

    def test_string_qubit_index_rejected(self):
        with self.assertRaises(TypeError):
            Circuit(2).measure("0")


class TestCircuitQubitCountValidation(unittest.TestCase):
    """БАГ: Circuit(n_qubits) не проверял тип n_qubits (в отличие от
    State), поэтому Circuit(2.5) успешно строился и падал только при
    вызове .run(), с сообщением, не указывающим на настоящую причину."""

    def test_fractional_n_qubits_rejected_at_construction(self):
        with self.assertRaises(TypeError):
            Circuit(2.5)

    def test_bool_n_qubits_rejected(self):
        with self.assertRaises(TypeError):
            Circuit(True)

    def test_zero_or_negative_n_qubits_rejected(self):
        with self.assertRaises(ValueError):
            Circuit(0)
        with self.assertRaises(ValueError):
            Circuit(-3)

    def test_state_and_circuit_agree_on_max_qubits(self):
        # Раньше MAX_RECOMMENDED_QUBITS дублировался в двух файлах и мог
        # разойтись при правке только одного из них.
        from .circuit import MAX_RECOMMENDED_QUBITS as circuit_max
        from ._common import MAX_RECOMMENDED_QUBITS as common_max
        self.assertEqual(circuit_max, common_max)
        with self.assertRaises(ValueError):
            Circuit(common_max + 1)
        with self.assertRaises(ValueError):
            State(common_max + 1)


class TestControlOverlapValidation(unittest.TestCase):
    """БАГ: control() (в отличие от controlled_gate()) не проверял
    повторяющиеся controls / control==target на этапе построения схемы —
    ошибка обнаруживалась только при run(), внутри State.apply_controlled."""

    def test_duplicate_controls_rejected_at_build_time(self):
        c = Circuit(2)
        with self.assertRaises(ValueError):
            c.control(G.X, [0, 0], 1)
        self.assertEqual(c.instructions, [])  # ничего не должно было добавиться

    def test_control_equals_target_rejected_at_build_time(self):
        c = Circuit(2)
        with self.assertRaises(ValueError):
            c.control(G.X, [0], 0)

    def test_mcx_duplicate_controls_rejected(self):
        with self.assertRaises(ValueError):
            Circuit(3).mcx([0, 0], 1)


class TestShotsValidation(unittest.TestCase):
    """БАГ: на "медленном" пути run() (reset или mid-circuit measure без
    reset) отрицательные shots не проверялись вовсе: `for _ in
    range(shots)` с отрицательным shots просто ничего не делал, и функция
    молча возвращала ({}, None) — притом что None нарушает контракт
    run(shots=...) -> (counts, State)."""

    def test_negative_shots_raises_on_fast_path(self):
        c = Circuit(1).h(0)
        with self.assertRaises(ValueError):
            c.run(shots=-5)

    def test_negative_shots_raises_on_reset_path(self):
        c = Circuit(1, seed=1).h(0).measure(0).reset(0)
        with self.assertRaises(ValueError):
            c.run(shots=-5)

    def test_negative_shots_raises_on_mid_circuit_measure_path(self):
        c = Circuit(1, seed=1).h(0).measure(0).h(0).measure(0)
        with self.assertRaises(ValueError):
            c.run(shots=-3)

    def test_float_shots_rejected(self):
        with self.assertRaises(TypeError):
            Circuit(1).h(0).run(shots=2.5)

    def test_zero_shots_returns_real_state_not_none_on_reset_path(self):
        # Раньше: final_state оставался None, потому что range(0) не
        # выполнял ни одной итерации цикла.
        c = Circuit(1, seed=1).h(0).measure(0).reset(0)
        counts, state = c.run(shots=0)
        self.assertEqual(counts, {})
        self.assertIsInstance(state, State)

    def test_zero_shots_returns_real_state_on_fast_path(self):
        c = Circuit(1).h(0)
        counts, state = c.run(shots=0)
        self.assertEqual(counts, {})
        self.assertIsInstance(state, State)


class TestAmplitudeValidation(unittest.TestCase):
    """БАГ: State.amplitude(bitstring) не проверял длину строки:
    amplitude('1') на 3-кубитном состоянии молча трактовалась как
    int('1', 2) == 1, тихо возвращая амплитуду не того базисного
    состояния, какое, скорее всего, имел в виду вызывающий (без
    исключения!). Слишком длинная строка падала с низкоуровневым
    numpy IndexError."""

    def test_short_bitstring_rejected(self):
        s = State(3)
        with self.assertRaises(ValueError):
            s.amplitude('1')

    def test_long_bitstring_rejected(self):
        s = State(3)
        with self.assertRaises(ValueError):
            s.amplitude('11111')

    def test_invalid_characters_rejected(self):
        s = State(3)
        with self.assertRaises(ValueError):
            s.amplitude('01x')

    def test_correct_length_bitstring_still_works(self):
        s = State(2)
        s.apply_single(G.H, 0)
        amp = s.amplitude('00')
        self.assertAlmostEqual(amp, 1 / np.sqrt(2))

    def test_integer_index_also_accepted(self):
        s = State(2)
        s.apply_single(G.X, 1)
        self.assertAlmostEqual(s.amplitude('01'), s.amplitude(1))


class TestNonUnitaryGateRejected(unittest.TestCase):
    """БАГ: Circuit.control()/controlled_gate() с произвольной матрицей
    принимали любую матрицу без проверки унитарности; неунитарная
    матрица молча портила нормировку состояния при run() (было
    воспроизведено на State.apply_single(2*I, 0), которая удваивала
    норму вектора состояния без единого предупреждения)."""

    def test_non_unitary_control_matrix_rejected_at_build_time(self):
        garbage = np.array([[2, 0], [0, 2]], dtype=complex)
        c = Circuit(2)
        with self.assertRaises(ValueError):
            c.control(garbage, [0], 1)

    def test_non_unitary_controlled_gate_matrix_rejected(self):
        garbage = np.eye(4, dtype=complex) * 2
        c = Circuit(3)
        with self.assertRaises(ValueError):
            c.controlled_gate(garbage, [0], [1, 2])

    def test_state_low_level_api_still_allows_raw_matrices(self):
        # Circuit — валидирующий, высокоуровневый API. State остаётся
        # низкоуровневым и намеренно не проверяет унитарность (нужно,
        # например, для моделирования шума / non-CPTP операций).
        s = State(1)
        s.apply_single(np.array([[2, 0], [0, 2]], dtype=complex), 0)
        self.assertAlmostEqual(np.linalg.norm(s.vector), 2.0)


class TestBuiltinGatesAreUnitary(unittest.TestCase):
    """БАГ (гипотетический, но проверяемый): любая опечатка в матрице
    встроенного гейта осталась бы незамеченной до первого случая, когда
    кто-то заметил физически неверный результат. gates.py теперь
    проверяет унитарность всех встроенных гейтов при импорте."""

    def test_all_named_gates_are_unitary(self):
        for name, mat in G.NAMED.items():
            product = mat.conj().T @ mat
            np.testing.assert_allclose(
                product, np.eye(mat.shape[0]), atol=1e-10,
                err_msg=f"Гейт {name} не унитарен")

    def test_all_two_qubit_matrices_are_unitary(self):
        for name, mat in [('CNOT', G.CNOT_MATRIX), ('CY', G.CY_MATRIX),
                           ('CZ', G.CZ_MATRIX), ('SWAP', G.SWAP_MATRIX)]:
            product = mat.conj().T @ mat
            np.testing.assert_allclose(
                product, np.eye(4), atol=1e-10, err_msg=f"Гейт {name} не унитарен")


class TestMeasureCbitValidation(unittest.TestCase):
    """БАГ: Circuit.measure(q, cbit=...) не проверял тип cbit — cbit
    документирован (measured_labels()) как строка или None, но
    measure(0, cbit=42) молча принимался, и несоответствие типа
    обнаружилось бы не здесь, а где-то произвольно далеко ниже по
    потоку использования измеренных подписей как строк."""

    def test_non_string_cbit_rejected(self):
        with self.assertRaises(TypeError):
            Circuit(1).measure(0, cbit=42)

    def test_none_cbit_still_allowed(self):
        c = Circuit(1).measure(0, cbit=None)
        self.assertEqual(c.measured_labels(), ['q0'])

    def test_string_cbit_still_allowed(self):
        c = Circuit(1).measure(0, cbit='a')
        self.assertEqual(c.measured_labels(), ['a'])


class TestSampleQubitDuplicateValidation(unittest.TestCase):
    """БАГ: Simulator.sample(qubits=...) не проверял повторы в qubits —
    sample(qubits=[0, 0]) молча дублировал соответствующий бит в каждом
    ключе counts (например, '00' вместо '0' для однокубитного среза),
    без единого предупреждения о вероятной ошибке вызывающего кода."""

    def test_duplicate_qubit_in_sample_rejected(self):
        state = Circuit(2).h(0).cx(0, 1).run()
        sim = Simulator(state)
        with self.assertRaises(ValueError):
            sim.sample(shots=10, qubits=[0, 0])

    def test_non_duplicate_qubits_still_work(self):
        state = Circuit(2).h(0).cx(0, 1).run()
        sim = Simulator(state, seed=0)
        counts = sim.sample(shots=10, qubits=[1, 0])
        self.assertTrue(all(len(k) == 2 for k in counts))


class TestPerShotCopyPerformance(unittest.TestCase):
    """БАГ: на "медленном" пути run() состояние копировалось на КАЖДОЙ
    итерации цикла (``final_state = state.copy()`` внутри цикла), хотя
    нужна только последняя копия — O(shots) лишних копий statevector.
    Здесь не измеряем время напрямую (нестабильно в CI), а проверяем
    через monkeypatch, что State.copy() вызывается ровно один раз."""

    def test_state_copy_called_once_regardless_of_shots(self):
        copy_calls = {'n': 0}
        original_copy = State.copy

        def counting_copy(self):
            copy_calls['n'] += 1
            return original_copy(self)

        State.copy = counting_copy
        try:
            c = Circuit(1, seed=3).h(0).measure(0).reset(0)
            c.run(shots=500)
        finally:
            State.copy = original_copy

        self.assertEqual(copy_calls['n'], 1)


if __name__ == '__main__':
    unittest.main()
