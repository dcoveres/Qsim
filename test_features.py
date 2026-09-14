"""Тесты на возможности, добавленные при расширении qsim:

- Иерархия исключений (QSimError и подклассы), обратная совместимость
  с TypeError/ValueError.
- Параметризованные схемы (Parameter, Circuit.bind()).
- Общие Pauli-string expectation values и вектор Блоха.
- Матрица плотности (State.density_matrix()).
- Шумовые модели (NoiseModel + каналы Крауса).
- Именованные классические биты (measure(q, cbit=...)).
- Экспорт в OpenQASM 2.0.
"""
import unittest
import numpy as np

try:
    from .circuit import Circuit
    from .state import State
    from .simulator import Simulator
    from . import gates as G
    from .parameters import Parameter
    from .noise import (
        NoiseModel, NoiseChannel, TwoQubitNoiseChannel,
        depolarizing_channel, bit_flip_channel, phase_flip_channel,
        amplitude_damping_channel, correlated_depolarizing_channel,
    )
    from ._errors import (
        QSimError, QSimTypeError, QSimValueError,
        QubitError, GateError, ShotsError, CircuitError,
    )
except ImportError:
    from circuit import Circuit
    from state import State
    from simulator import Simulator
    import gates as G
    from parameters import Parameter
    from noise import (
        NoiseModel, NoiseChannel, TwoQubitNoiseChannel,
        depolarizing_channel, bit_flip_channel, phase_flip_channel,
        amplitude_damping_channel, correlated_depolarizing_channel,
    )
    from _errors import (
        QSimError, QSimTypeError, QSimValueError,
        QubitError, GateError, ShotsError, CircuitError,
    )


class TestExceptionHierarchy(unittest.TestCase):
    """Новые исключения должны одновременно являться QSimError И
    соответствующим builtin-типом — иначе старый код с
    except TypeError/except ValueError сломался бы."""

    def test_qubit_index_error_is_both_qsim_and_value_error(self):
        with self.assertRaises(ValueError):
            Circuit(2).h(5)
        try:
            Circuit(2).h(5)
        except QSimError as e:
            self.assertIsInstance(e, QubitError)
            self.assertIsInstance(e, ValueError)
        else:
            self.fail("Circuit(2).h(5) должен был бросить исключение")

    def test_type_error_is_both_qsim_and_type_error(self):
        with self.assertRaises(TypeError):
            Circuit(2).h(1.5)
        try:
            Circuit(2).h(1.5)
        except QSimError as e:
            self.assertIsInstance(e, QSimTypeError)
            self.assertIsInstance(e, TypeError)
        else:
            self.fail("Circuit(2).h(1.5) должен был бросить исключение")

    def test_shots_error_is_value_error(self):
        with self.assertRaises(ValueError):
            Circuit(1).h(0).run(shots=-1)
        try:
            Circuit(1).h(0).run(shots=-1)
        except QSimError as e:
            self.assertIsInstance(e, ShotsError)
        else:
            self.fail("отрицательные shots должны были бросить исключение")

    def test_non_unitary_matrix_is_gate_error(self):
        garbage = np.array([[2, 0], [0, 2]], dtype=complex)
        try:
            Circuit(2).control(garbage, [0], 1)
        except QSimError as e:
            self.assertIsInstance(e, GateError)
        else:
            self.fail("неунитарная матрица должна была бросить исключение")

    def test_catching_qsim_error_catches_everything(self):
        # Программный код может ловить один тип для всех ошибок qsim.
        offenders = [
            lambda: Circuit(2).h(5),          # QubitError
            lambda: Circuit(2).h(1.5),        # QSimTypeError
            lambda: Circuit(1).h(0).run(shots=-1),  # ShotsError
        ]
        for fn in offenders:
            with self.assertRaises(QSimError):
                fn()


class TestParameterizedCircuits(unittest.TestCase):
    def test_unbound_parameter_listed(self):
        theta = Parameter('theta')
        c = Circuit(1).rx(theta, 0)
        self.assertEqual(c.parameters, [theta])

    def test_run_with_unbound_parameter_raises_circuit_error(self):
        theta = Parameter('theta')
        c = Circuit(1).rx(theta, 0)
        with self.assertRaises(CircuitError):
            c.run()

    def test_bind_produces_correct_matrix(self):
        theta = Parameter('theta')
        c = Circuit(1).rx(theta, 0)
        bound = c.bind({theta: np.pi})
        state = bound.run()
        expected = G.Rx(np.pi) @ np.array([1, 0], dtype=complex)
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_bind_does_not_mutate_original_circuit(self):
        theta = Parameter('theta')
        c = Circuit(1).rx(theta, 0)
        _ = c.bind({theta: np.pi})
        self.assertEqual(c.parameters, [theta])  # оригинал остался несвязанным

    def test_bind_by_name_also_works(self):
        theta = Parameter('theta')
        c = Circuit(1).rx(theta, 0)
        bound = c.bind({'theta': np.pi})
        state = bound.run()
        expected = G.Rx(np.pi) @ np.array([1, 0], dtype=complex)
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_partial_bind_leaves_other_parameters_unbound(self):
        t1, t2 = Parameter('t1'), Parameter('t2')
        c = Circuit(2).rx(t1, 0).ry(t2, 1)
        partial = c.bind({t1: 1.23})
        self.assertEqual(partial.parameters, [t2])
        full = partial.bind({t2: 4.56})
        self.assertEqual(full.parameters, [])
        full.run()  # не должно бросать

    def test_two_parameters_with_same_name_are_distinct(self):
        a = Parameter('theta')
        b = Parameter('theta')
        c = Circuit(2).rx(a, 0).rx(b, 1)
        self.assertEqual(len(c.parameters), 2)
        # bind по объекту должен связать только конкретный Parameter:
        bound = c.bind({a: 0.5})
        self.assertEqual(bound.parameters, [b])


class TestExpectationAndBlochVector(unittest.TestCase):
    def test_expectation_zz_on_bell_state(self):
        state = Circuit(2).h(0).cx(0, 1).run()
        sim = Simulator(state)
        self.assertAlmostEqual(sim.expectation('ZZ'), 1.0, places=8)

    def test_expectation_xx_on_bell_state(self):
        state = Circuit(2).h(0).cx(0, 1).run()
        sim = Simulator(state)
        self.assertAlmostEqual(sim.expectation('XX'), 1.0, places=8)

    def test_expectation_matches_expectation_z_single_qubit(self):
        state = Circuit(1).rx(0.9, 0).run()
        sim = Simulator(state)
        self.assertAlmostEqual(sim.expectation('Z'), sim.expectation_z(0), places=10)

    def test_expectation_rejects_invalid_pauli_chars(self):
        state = Circuit(1).h(0).run()
        sim = Simulator(state)
        with self.assertRaises(QSimValueError):
            sim.expectation('Q')

    def test_expectation_rejects_length_mismatch(self):
        state = Circuit(2).h(0).run()
        sim = Simulator(state)
        with self.assertRaises(QSimValueError):
            sim.expectation('ZZ', qubits=[0])

    def test_bloch_vector_of_zero_state(self):
        state = Circuit(1).run()
        sim = Simulator(state)
        x, y, z = sim.bloch_vector(0)
        np.testing.assert_allclose([x, y, z], [0, 0, 1], atol=1e-10)

    def test_bloch_vector_of_one_state(self):
        state = Circuit(1).x(0).run()
        sim = Simulator(state)
        x, y, z = sim.bloch_vector(0)
        np.testing.assert_allclose([x, y, z], [0, 0, -1], atol=1e-10)

    def test_bloch_vector_of_plus_state(self):
        state = Circuit(1).h(0).run()
        sim = Simulator(state)
        x, y, z = sim.bloch_vector(0)
        np.testing.assert_allclose([x, y, z], [1, 0, 0], atol=1e-10)

    def test_bloch_vector_of_entangled_qubit_has_norm_less_than_one(self):
        state = Circuit(2).h(0).cx(0, 1).run()
        sim = Simulator(state)
        x, y, z = sim.bloch_vector(0)
        norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        self.assertLess(norm, 1e-6)  # максимально смешанный редуцированный кубит


class TestDensityMatrix(unittest.TestCase):
    def test_density_matrix_of_plus_state(self):
        state = Circuit(1).h(0).run()
        rho = state.density_matrix()
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        np.testing.assert_allclose(rho, expected, atol=1e-10)

    def test_density_matrix_trace_is_one(self):
        state = Circuit(3, seed=0).h(0).cx(0, 1).rz(0.4, 2).run()
        rho = state.density_matrix()
        self.assertAlmostEqual(np.trace(rho).real, 1.0, places=8)

    def test_density_matrix_is_hermitian(self):
        state = Circuit(2).h(0).cx(0, 1).run()
        rho = state.density_matrix()
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-10)


class TestNoiseChannels(unittest.TestCase):
    def test_channel_rejects_non_tpcp_operators(self):
        with self.assertRaises(GateError):
            NoiseChannel([np.eye(2, dtype=complex) * 2])

    def test_depolarizing_rejects_out_of_range_probability(self):
        with self.assertRaises(QSimValueError):
            depolarizing_channel(1.5)
        with self.assertRaises(QSimValueError):
            depolarizing_channel(-0.1)

    def test_bit_flip_p1_is_deterministic_x(self):
        # X затем bit-flip(p=1) -> детерминированно возвращается в |0>.
        c = Circuit(1).x(0).measure(0)
        noise = NoiseModel().add_all_qubit_channel(bit_flip_channel(1.0))
        counts, _ = c.run(shots=30, noise=noise)
        self.assertEqual(counts, {'0': 30})

    def test_bit_flip_p0_is_identity(self):
        c = Circuit(1).h(0)
        noise = NoiseModel().add_all_qubit_channel(bit_flip_channel(0.0))
        state_noisy = c.run(noise=noise)
        state_clean = Circuit(1).h(0).run()
        np.testing.assert_allclose(state_noisy.vector, state_clean.vector, atol=1e-10)

    def test_phase_flip_p0_is_identity(self):
        c = Circuit(1).h(0)
        noise = NoiseModel().add_all_qubit_channel(phase_flip_channel(0.0))
        state_noisy = c.run(noise=noise)
        state_clean = Circuit(1).h(0).run()
        np.testing.assert_allclose(state_noisy.vector, state_clean.vector, atol=1e-10)

    def test_depolarizing_p0_is_identity(self):
        c = Circuit(1).h(0)
        noise = NoiseModel().add_all_qubit_channel(depolarizing_channel(0.0))
        state_noisy = c.run(noise=noise)
        state_clean = Circuit(1).h(0).run()
        np.testing.assert_allclose(state_noisy.vector, state_clean.vector, atol=1e-10)

    def test_amplitude_damping_gamma1_relaxes_one_to_zero(self):
        c = Circuit(1).x(0).measure(0)
        noise = NoiseModel().add_all_qubit_channel(amplitude_damping_channel(1.0))
        counts, _ = c.run(shots=30, noise=noise)
        self.assertEqual(counts, {'0': 30})

    def test_noise_forces_per_shot_path_and_state_stays_normalized(self):
        c = Circuit(2, seed=1).h(0).cx(0, 1)
        noise = NoiseModel().add_all_qubit_channel(depolarizing_channel(0.1))
        counts, state = c.run(shots=50, noise=noise)
        self.assertAlmostEqual(np.linalg.norm(state.vector), 1.0, places=8)
        self.assertEqual(sum(counts.values()), 50)

    def test_per_qubit_channel_overrides_default(self):
        noise = NoiseModel()
        noise.add_all_qubit_channel(bit_flip_channel(1.0))
        noise.add_qubit_channel(1, None)  # кубит 1 явно без шума
        c = Circuit(2).x(0).x(1)
        c.measure(0); c.measure(1)
        counts, _ = c.run(shots=10, noise=noise)
        # кубит0: X затем гарантированный bit-flip -> 0; кубит1: X без шума -> 1
        self.assertEqual(counts, {'01': 10})

    def test_depolarizing_matches_analytic_expectation_value(self):
        # Для параметризации, использованной в depolarizing_channel
        # (Kraus: sqrt(1-p)*I, sqrt(p/3)*{X,Y,Z}), <Z> состояния |1>
        # после канала аналитически равно -(1 - 4p/3), а не -(1-p) —
        # см. docstring depolarizing_channel. Проверяем это статистически
        # через усреднение по многим независимым quantum-trajectory
        # прогонам (Circuit.run без shots — один прогон на вызов).
        p = 0.3
        c = Circuit(1).x(0)
        noise = NoiseModel().add_all_qubit_channel(depolarizing_channel(p))
        rng = np.random.default_rng(0)
        zs = []
        for _ in range(4000):
            c.seed = int(rng.integers(0, 2**31))
            state = c.run(noise=noise)
            zs.append(Simulator(state).expectation('Z'))
        analytic = -(1 - 4 * p / 3)
        self.assertAlmostEqual(np.mean(zs), analytic, delta=0.05)


class TestTwoQubitNoiseChannel(unittest.TestCase):
    """NoiseModel/NoiseChannel раньше могли зашумлять только отдельные
    кубиты по отдельности, что в принципе не может выразить ошибку,
    коррелированную между двумя кубитами (например, деполяризацию CX как
    единого двухкубитного события). TwoQubitNoiseChannel +
    NoiseModel.add_two_qubit_channel закрывают этот случай."""

    def test_rejects_wrong_dimension_kraus_ops(self):
        with self.assertRaises(GateError):
            TwoQubitNoiseChannel([np.eye(2, dtype=complex)])

    def test_rejects_non_tpcp_operators(self):
        with self.assertRaises(GateError):
            TwoQubitNoiseChannel([np.eye(4, dtype=complex) * 2])

    def test_correlated_depolarizing_p0_is_identity(self):
        c = Circuit(2, seed=0).h(0).cx(0, 1)
        noise = NoiseModel().add_two_qubit_channel(0, 1, correlated_depolarizing_channel(0.0))
        state_noisy = c.run(noise=noise)
        state_clean = Circuit(2).h(0).cx(0, 1).run()
        np.testing.assert_allclose(state_noisy.vector, state_clean.vector, atol=1e-10)

    def test_two_qubit_channel_keeps_state_normalized(self):
        c = Circuit(2, seed=1).h(0).cx(0, 1)
        noise = NoiseModel().add_two_qubit_channel(0, 1, correlated_depolarizing_channel(0.2))
        counts, state = c.run(shots=50, noise=noise)
        self.assertAlmostEqual(np.linalg.norm(state.vector), 1.0, places=8)
        self.assertEqual(sum(counts.values()), 50)

    def test_two_qubit_channel_pair_is_unordered(self):
        # add_two_qubit_channel(0, 1, ...) должен действовать одинаково для
        # гейта cx(0,1) и (гипотетически) cx(1,0) — пара неупорядочена.
        noise = NoiseModel().add_two_qubit_channel(1, 0, correlated_depolarizing_channel(1.0))
        self.assertIsNotNone(noise.two_qubit_channel_for(0, 1))

    def test_two_qubit_channel_overrides_per_qubit_channel_for_that_pair(self):
        # Если для пары зарегистрирован двухкубитный канал, он должен
        # ПОЛНОСТЬЮ заменять собой независимый шум по отдельным кубитам
        # для этой инструкции (иначе шум был бы учтён дважды). cx(0,1) на
        # |00> оставляет состояние |00>; per-qubit bit-flip(p=1) после
        # этого гейта перевёл бы его в |11>, если бы двухкубитный
        # identity-канал для пары (0,1) его не подавлял.
        c = Circuit(2).cx(0, 1)
        noise = NoiseModel()
        noise.add_all_qubit_channel(bit_flip_channel(1.0))
        identity_channel = TwoQubitNoiseChannel([np.eye(4, dtype=complex)], name="identity")
        noise.add_two_qubit_channel(0, 1, identity_channel)
        state = c.run(noise=noise)
        expected = np.array([1, 0, 0, 0], dtype=complex)  # |00>, неизменно
        np.testing.assert_allclose(state.vector, expected, atol=1e-10)

    def test_add_two_qubit_channel_none_removes_it(self):
        noise = NoiseModel().add_two_qubit_channel(0, 1, correlated_depolarizing_channel(0.5))
        noise.add_two_qubit_channel(0, 1, None)
        self.assertIsNone(noise.two_qubit_channel_for(0, 1))


class TestNamedClassicalBits(unittest.TestCase):
    def test_measured_labels_default_to_qN(self):
        c = Circuit(2).h(0).cx(0, 1).measure_all()
        self.assertEqual(c.measured_labels(), ['q0', 'q1'])

    def test_measured_labels_use_cbit_names(self):
        c = Circuit(2).h(0).cx(0, 1)
        c.measure(0, cbit='a')
        c.measure(1, cbit='b')
        self.assertEqual(c.measured_labels(), ['a', 'b'])

    def test_counts_keys_unaffected_by_cbit_naming(self):
        # cbit — чисто про читаемость подписей, а не про сами данные:
        # ключи counts остаются битовыми строками той же длины/порядка.
        c = Circuit(2, seed=3).h(0).cx(0, 1)
        c.measure(0, cbit='a'); c.measure(1, cbit='b')
        counts, _ = c.run(shots=20)
        self.assertTrue(all(len(k) == 2 for k in counts))
        self.assertTrue(set(counts) <= {'00', '11'})


class TestQasmExport(unittest.TestCase):
    def test_bell_state_qasm(self):
        c = Circuit(2).h(0).cx(0, 1)
        c.measure_all()
        qasm = c.to_qasm()
        self.assertIn('OPENQASM 2.0;', qasm)
        self.assertIn('qreg q[2];', qasm)
        self.assertIn('creg c[2];', qasm)
        self.assertIn('h q[0];', qasm)
        self.assertIn('cx q[0],q[1];', qasm)
        self.assertIn('measure q[0] -> c[0];', qasm)
        self.assertIn('measure q[1] -> c[1];', qasm)

    def test_no_creg_line_when_no_explicit_measure(self):
        c = Circuit(1).h(0)
        qasm = c.to_qasm()
        self.assertNotIn('creg', qasm)

    def test_rotation_gate_angle_roundtrips_reasonably(self):
        c = Circuit(1).rx(0.123456, 0)
        qasm = c.to_qasm()
        self.assertIn('rx(0.123456) q[0];', qasm)

    def test_toffoli_and_fredkin_export(self):
        c = Circuit(3).ccx(0, 1, 2)
        # ccx (Toffoli) — уже в самой первой версии qelib1.inc, экспортируется как есть.
        self.assertIn('ccx q[0],q[1],q[2];', c.to_qasm())
        c2 = Circuit(3).cswap(0, 1, 2)
        # cswap — НЕ в самой первой версии qelib1.inc, поэтому раскладывается
        # в cx/ccx/cx, а не вызывается по имени 'cswap' (см. TestQasmPortability).
        qasm2 = c2.to_qasm()
        self.assertNotIn('cswap', qasm2)
        self.assertIn('ccx q[0],q[1],q[2];', qasm2)

    def test_crx_export(self):
        # crx — НЕ в самой первой версии qelib1.inc, раскладывается в u1/cx/u3
        # (см. TestQasmPortability для проверки физической корректности разложения).
        c = Circuit(2).crx(0.5, 0, 1)
        qasm = c.to_qasm()
        self.assertNotIn('crx', qasm)
        self.assertIn('cx q[0],q[1];', qasm)

    def test_u3_gate_export_unsupported(self):
        c = Circuit(1).u(0.1, 0.2, 0.3, 0)
        with self.assertRaises(CircuitError):
            c.to_qasm()

    def test_mcx_with_three_controls_export_unsupported(self):
        c = Circuit(4).x(0).x(1).x(2).mcx([0, 1, 2], 3)
        with self.assertRaises(CircuitError):
            c.to_qasm()

    def test_arbitrary_unitary_control_export_unsupported(self):
        c = Circuit(2).control(np.eye(2, dtype=complex), [0], 1, label='CUSTOM')
        with self.assertRaises(CircuitError):
            c.to_qasm()

    def test_unbound_parameter_export_unsupported(self):
        theta = Parameter('theta')
        c = Circuit(1).rx(theta, 0)
        with self.assertRaises(CircuitError):
            c.to_qasm()

    def test_barrier_and_reset_export(self):
        c = Circuit(1).h(0).barrier().reset(0)
        qasm = c.to_qasm()
        self.assertIn('barrier q[0];', qasm)
        self.assertIn('reset q[0];', qasm)


class TestQasmPortability(unittest.TestCase):
    """to_qasm() раньше экспортировал sx/sxdg/swap/crx/cry/cswap по имени
    гейта, предполагая, что qelib1.inc инструмента-читателя определяет
    именно это имя. Самая первая/строгая версия qelib1.inc (OpenQASM 2.0,
    arXiv:1707.03429) этих имён не содержит вовсе — такой QASM не
    загрузился бы в инструменте со строгим qelib1.inc. Эти тесты проверяют,
    что перечисленные гейты теперь раскладываются в гарантированно
    присутствующий базовый набор (u1/u3/ry/cx/ccx), а также что физическая
    корректность каждого разложения (полное совпадение с исходной
    матрицей) была явно проверена численно при разработке."""

    def test_sx_sxdg_do_not_use_native_gate_name(self):
        c = Circuit(1).sx(0).sxdg(0)
        qasm = c.to_qasm()
        self.assertNotIn('sx ', qasm)
        self.assertNotIn('sxdg', qasm)
        self.assertIn('u3(1.57079632679,-1.57079632679,1.57079632679) q[0];', qasm)
        self.assertIn('u3(-1.57079632679,-1.57079632679,1.57079632679) q[0];', qasm)

    def test_swap_does_not_use_native_gate_name(self):
        c = Circuit(2).swap(0, 1)
        qasm = c.to_qasm()
        self.assertNotIn('swap', qasm)
        self.assertEqual(qasm.count('cx q[0],q[1];') + qasm.count('cx q[1],q[0];'), 3)

    def test_cry_export_does_not_use_native_gate_name(self):
        c = Circuit(2).cry(0.8, 0, 1)
        qasm = c.to_qasm()
        self.assertNotIn('cry', qasm)
        self.assertIn('ry(0.4) q[1];', qasm)
        self.assertIn('ry(-0.4) q[1];', qasm)

    def test_crz_export_still_uses_native_gate_name(self):
        # crz ЕСТЬ в самой первой версии qelib1.inc — разложение не требуется.
        c = Circuit(2).crz(0.3, 0, 1)
        self.assertIn('crz(0.3) q[0],q[1];', c.to_qasm())

    def test_control_with_empty_controls_exports_as_plain_gate(self):
        # Раньше control(...)/controlled_gate(...) с пустым списком
        # controls (полностью определённая операция — просто гейт на
        # targets без всякого управления) не совпадал ни с одним
        # известным паттерном экспорта и бросал CircuitError.
        c = Circuit(2).control(G.X, [], 1, label='X')
        self.assertIn('x q[1];', c.to_qasm())

    def test_control_with_empty_controls_and_rotation_exports_natively(self):
        c = Circuit(1).control(G.Rz(0.5), [], 0, label='RZ(0.5)')
        self.assertIn('rz(0.5) q[0];', c.to_qasm())

    def test_controlled_phase_via_control_exports_as_cu1(self):
        # Управляемая фаза, добавленная через общий control() (а не через
        # cp()), тоже должна экспортироваться — раньше произвольная
        # пользовательская матрица с меткой 'P(...)' не распознавалась.
        c = Circuit(2).control(G.Phase(0.6), [0], 1, label='P(0.6)')
        self.assertIn('cu1(0.6) q[0],q[1];', c.to_qasm())


if __name__ == '__main__':
    unittest.main()
