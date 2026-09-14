# qsim

Лёгкий statevector-эмулятор квантовых схем на чистом NumPy. Без внешних
квантовых SDK, без C-расширений — один пакет на Python + NumPy, который
можно читать целиком и понимать, что происходит на каждом шаге.

```python
from qsim import Circuit

c = Circuit(2).h(0).cx(0, 1)
state = c.run()
print(c.draw())
# ┌───┐
# ┤ H ├──●──
# └───┘┌─┴─┐
# ─────┤ X ├
#      └───┘
```

## Возможности

- **Statevector-симуляция** до ~24–25 кубитов на обычной машине (вектор
  состояния занимает `16 байт × 2^n`; лимит и порог предупреждения
  настраиваются).
- **Набор гейтов**: все стандартные фиксированные однокубитные гейты
  (`H, X, Y, Z, S, SDG, T, TDG, SX, SXDG`), параметрические
  (`RX, RY, RZ, P/U1, U3`), двухкубитные (`CX/CNOT, CY, CZ, SWAP, CP`),
  многоуправляемые гейты произвольной формы (`control`, `controlled_gate`,
  `mcx`, `ccx`/Toffoli, `cswap`/Fredkin) — с проверкой унитарности
  пользовательских матриц на этапе построения схемы, а не во время `run()`.
- **Измерения**: mid-circuit measurement с реальным коллапсом,
  `reset()`, отложенное (deferred) измерение там, где это физически
  эквивалентно — для быстрого пути `run(shots=...)` без лишних прогонов
  схемы.
- **Параметризованные схемы** (`Parameter`, `Circuit.bind(...)`) — удобно
  для VQE/QAOA и любых сценариев, где схема строится один раз, а
  запускается многократно с разными значениями углов.
- **Модели шума** (`NoiseModel`, `NoiseChannel`, `TwoQubitNoiseChannel`) —
  реализация через quantum trajectories (Monte Carlo wave function), точная
  техника, не приближение. Есть готовые каналы (`depolarizing_channel`,
  `bit_flip_channel`, `phase_flip_channel`, `amplitude_damping_channel`,
  `correlated_depolarizing_channel`) и возможность задать свой канал через
  произвольные операторы Крауса.
- **Инспекция состояния**: точные вероятности, амплитуды, вектор Блоха
  одного кубита (через partial trace, без построения полной матрицы
  плотности), ожидания `⟨Z⟩` и произвольной строки Паули `⟨P⟩`, полная
  матрица плотности для малых `n`.
- **Экспорт в OpenQASM 2.0** (`Circuit.to_qasm()`), совместимый с самой
  строгой версией `qelib1.inc` — гейты, которых там нет (`sx`, `swap`,
  `crx`, `cry`, `cswap`), автоматически раскладываются в базовый набор,
  а не экспортируются по имени вслепую.
- **Единая иерархия исключений** (`QSimError` → `QubitError` / `GateError`
  / `ShotsError` / `CircuitError`), совместимая с обычными
  `TypeError`/`ValueError`, так что существующий код с
  `except (TypeError, ValueError)` продолжает работать без изменений.
- **Опциональная визуализация** (`qsim.visualization`, требует
  matplotlib, не импортируется автоматически) — гистограмма измерений и
  сфера Блоха.
- ASCII-отрисовка схемы (`Circuit.draw()`) с выровненными колонками.

## Установка

Пакет не публиковался в PyPI — используйте как локальный модуль:

```bash
git clone <url-репозитория>
cd <repo>
pip install numpy
# опционально, для plot_histogram/plot_bloch:
pip install matplotlib
```

Требования: Python ≥ 3.10 (используются `from __future__ import annotations`
и синтаксис `X | None`), NumPy.

## Быстрый старт

### Bell state и GHZ

```python
from qsim import Circuit, Simulator

# Bell state — точное распределение
c = Circuit(2).h(0).cx(0, 1)
state = c.run()
print(Simulator(state).probabilities_dict())
# {'00': 0.5, '11': 0.5}

# GHZ на 5 кубитах — сэмплирование
c = Circuit(5, seed=0).h(0)
for i in range(4):
    c.cx(i, i + 1)
c.measure_all()
counts, _ = c.run(shots=1000)
print(counts)  # только '00000' и '11111'
```

### Toffoli, Fredkin, многоуправляемый X

```python
Circuit(3).x(0).x(1).ccx(0, 1, 2).run()      # Toffoli
Circuit(3).x(0).x(1).cswap(0, 1, 2).run()    # Fredkin
Circuit(6).x(0).x(1).x(2).x(3).x(4).mcx([0, 1, 2, 3, 4], 5).run()
```

### Параметризованные схемы

```python
from qsim import Circuit, Parameter

theta = Parameter("θ")
c = Circuit(1).rx(theta, 0)

bound = c.bind({theta: 0.73})
state = bound.run()
```

### Шум

```python
from qsim import Circuit
from qsim.noise import NoiseModel, depolarizing_channel

noise = NoiseModel().add_all_qubit_channel(depolarizing_channel(0.02))
counts, _ = Circuit(2).h(0).cx(0, 1).run(shots=2000, noise=noise)
```

### Экспорт в QASM

```python
c = Circuit(2).h(0).cx(0, 1).measure_all()
print(c.to_qasm())
```

Полную демонстрацию можно посмотреть и запустить через `python -m qsim.demo`.

## Архитектура пакета

| Модуль             | Назначение                                                        |
|---------------------|--------------------------------------------------------------------|
| `state.py`           | `State` — вектор состояния и низкоуровневые векторизованные NumPy-операции (`apply_single/apply_gate/apply_controlled`), без валидации семантики выше уровня формы/индексов. |
| `circuit.py`          | `Circuit` — список инструкций, валидирующий высокоуровневый API построения схемы, `run()`, `draw()`, `to_qasm()`. |
| `simulator.py`        | `Simulator` — измерения, сэмплирование, `probabilities_dict`, `expectation`, `bloch_vector`. |
| `gates.py`            | Матрицы встроенных гейтов + самопроверка унитарности при импорте. |
| `parameters.py`       | `Parameter` и отложенное построение матриц для параметризованных схем. |
| `noise.py`             | `NoiseChannel`, `TwoQubitNoiseChannel`, `NoiseModel`, готовые каналы шума. |
| `_common.py`          | Общие функции валидации (индексы кубитов, shots, унитарность и т.д.), единый источник констант. |
| `_errors.py`           | Иерархия исключений `QSimError`. |
| `visualization.py`     | Опциональные `plot_histogram`/`plot_bloch` (matplotlib). |

**Конвенция индексации кубитов**: кубит 0 — самый старший (левый) бит в
двоичной записи индекса базисного состояния. Это противоположно
little-endian конвенции Qiskit (там кубит 0 — младший бит); при обмене
данными с Qiskit потребуется перестановка кубитов.

`State` остаётся низкоуровневым и намеренно не проверяет унитарность
передаваемых матриц (это нужно, например, для шумовых операторов); вся
валидация, важная для «чистого» квантового API, сосредоточена в `Circuit`.

## Тестирование

```bash
python -m pytest
# или без pytest:
python -m unittest discover
```

Тестовый набор (120+ тестов) разбит по назначению:

- `test_physics.py` — сверка каждого гейта/схемы с независимо построенными
  эталонными unitary-матрицами (`numpy.kron`), включая тесты на случайных
  перестановках control/target кубитов.
- `test_circuit.py` — mid-circuit measurement, отрисовка, контракт `run()`.
- `test_bugfixes.py` — регрессионные тесты на баги, найденные и
  исправленные при аудите (см. CHANGELOG).
- `test_features.py` — тесты новых возможностей (параметры, шум, QASM,
  Pauli-expectation, вектор Блоха и т.д.), включая переносимость
  QASM-экспорта на разложенных гейтах.

## Ограничения (осознанные)

- Statevector-бэкенд: память растёт как `O(2^n)`, полная матрица плотности
  — как `O(4^n)`; для больших схем с сильным шумом это становится
  дорого — по умолчанию лимит `MAX_RECOMMENDED_QUBITS = 25`.
- `u()`/U3 не параметризуется через `Parameter`.
- Транспиляция произвольной унитарной матрицы в базисные гейты для
  QASM-экспорта не реализована (`control()`/`controlled_gate()` с
  произвольной матрицей и `mcx` с 3+ controls экспортировать нельзя —
  явная ошибка вместо тихой потери части схемы).
- Коррелированный шум поддержан только для пар кубитов
  (`TwoQubitNoiseChannel`); многокубитные управляемые гейты зашумляются
  независимо по каждому кубиту.

## Changelog

История аудита, найденных багов и добавленных возможностей по версиям —
в [CHANGELOG.md](./CHANGELOG.md).

## Лицензия

Apache 2.0
