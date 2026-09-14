import unittest
import numpy as np

try:
    from .circuit import Circuit
except ImportError:
    from circuit import Circuit


class TestMeasurements(unittest.TestCase):
    def test_mid_circuit_measurement_is_not_deferred(self):
        c = Circuit(1, seed=123).h(0).measure(0).h(0).measure(0)
        counts, _ = c.run(shots=4000)
        self.assertGreater(counts.get("0", 0), 1500)
        self.assertGreater(counts.get("1", 0), 1500)

    def test_measure_before_reset_is_executed(self):
        c = Circuit(1, seed=7).h(0).measure(0).reset(0)
        counts, _ = c.run(shots=1000)
        self.assertAlmostEqual(counts.get("0", 0) / 1000, 0.5, delta=0.08)

    def test_measurement_after_reset(self):
        c = Circuit(1, seed=7).x(0).reset(0).measure(0)
        counts, _ = c.run(shots=1000)
        self.assertEqual(counts, {"0": 1000})


class TestDraw(unittest.TestCase):
    def test_columns_are_aligned(self):
        c = Circuit(3).h(0).rx(0.123456, 1).cx(0, 2).measure(2)
        lines = c.draw().splitlines()
        self.assertEqual(len({len(line) for line in lines}), 1)

    def test_cp_has_meaningful_target_symbol(self):
        c = Circuit(2).cp(0.5, 0, 1)
        drawing = c.draw()
        self.assertNotIn("[C]", drawing)
        self.assertIn("P", drawing)

    def test_controlled_long_label_has_connectors(self):
        c = Circuit(4).control(np.eye(2), [0, 2], 3, label="LONG_GATE")
        drawing = c.draw().splitlines()
        self.assertIn("│", drawing[1])
        # q2 is itself the second control; q3 is the target.
        self.assertIn("●", drawing[2])
        self.assertIn("LONG_GATE", drawing[3])


class TestRunContract(unittest.TestCase):
    def test_single_run_returns_state(self):
        state = Circuit(1).h(0).run()
        self.assertTrue(hasattr(state, "vector"))

    def test_shots_run_returns_counts_and_state(self):
        result = Circuit(1).h(0).run(shots=10)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
