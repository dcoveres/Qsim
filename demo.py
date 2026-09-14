"""Демонстрация возможностей эмулятора. Запуск: python -m qsim.demo"""
import time
import numpy as np
from .circuit import Circuit
from .simulator import Simulator



def demo_bell():
    print("\n=== Bell state ===")
    c = Circuit(2)
    c.h(0).cx(0, 1)
    sim = Simulator(c.run())
    print(c.draw())
    print("Точное распределение:", sim.probabilities_dict())


def demo_ghz():
    print("\n=== GHZ на 5 кубитах ===")
    c = Circuit(5, seed=0)
    c.h(0)
    for i in range(4):
        c.cx(i, i + 1)
    c.measure_all()
    counts, _ = c.run(shots=1000)
    print("Сэмплы (должны быть только 00000 и 11111):", counts)


def demo_toffoli_fredkin():
    print("\n=== Toffoli (CCX) и Fredkin (CSWAP) ===")
    c = Circuit(3).x(0).x(1).ccx(0, 1, 2)
    state = c.run()
    idx = int(np.argmax(np.abs(state.vector)))
    print(f"x(0).x(1).ccx(0,1,2) -> |{format(idx, '03b')}> (ожидаем 111)")

    c = Circuit(3).x(0).x(1).cswap(0, 1, 2)
    state = c.run()
    idx = int(np.argmax(np.abs(state.vector)))
    print(f"x(0).x(1).cswap(0,1,2) -> |{format(idx, '03b')}> (ожидаем 101)")


def demo_mcx():
    print("\n=== Многократно управляемый X (mcx) на 6 кубитах ===")
    n = 6
    c = Circuit(n).x(0).x(1).x(2).x(3).x(4)  # controls 0..4 = 1
    c.mcx([0, 1, 2, 3, 4], 5)
    state = c.run()
    idx = int(np.argmax(np.abs(state.vector)))
    print(f"все 5 контролей=1 -> |{format(idx, f'0{n}b')}> (ожидаем 111111)")


def demo_benchmark_20q():
    print("\n=== Бенчмарк на 20 кубитах ===")
    n = 20
    c = Circuit(n, seed=42)
    for q in range(n):
        c.h(q)
    for q in range(n - 1):
        c.cx(q, q + 1)
    c.mcx([0, 1, 2, 3, 4], 19)
    c.cswap(0, 5, 10)
    c.rz(0.37, 3)

    t0 = time.time()
    state = c.run()
    t1 = time.time()
    counts, _ = c.run(shots=2000)
    t2 = time.time()

    print(f"n_qubits={n}, dim={state.dim}, память={state.num_bytes()/1e6:.1f}MB")
    print(f"время прогона схемы: {t1-t0:.3f}s")
    print(f"время прогона + 2000 shots сэмплирования: {t2-t1:.3f}s")
    print(f"сумма вероятностей (должна быть ~1): {state.probabilities().sum():.6f}")


if __name__ == '__main__':
    demo_bell()
    demo_ghz()
    demo_toffoli_fredkin()
    demo_mcx()
    demo_benchmark_20q()
