"""Схема: список инструкций + запуск на State.

Поддерживает до ~20+ кубит (statevector-симуляция, память ~ 16 байт * 2^n).
"""
from __future__ import annotations

import re

import numpy as np

from .state import State
from .simulator import Simulator
from . import gates as G
from ._common import (
    MAX_RECOMMENDED_QUBITS,
    check_qubit_count,
    check_qubit_index,
    check_qubit_indices,
    check_real,
    check_shots,
    check_unitary,
)
from ._errors import CircuitError, GateError, QSimTypeError, QubitError
from .parameters import Parameter, _ParamGate

# Регистрируем фабрики матриц по значению параметра здесь (а не в
# parameters.py), чтобы избежать циклического импорта parameters<->gates:
# gates.py ничего не знает про параметры, а circuit.py уже импортирует и то,
# и другое.
_ParamGate._MATRIX_FACTORIES = {'RX': G.Rx, 'RY': G.Ry, 'RZ': G.Rz, 'P': G.Phase}

_ANGLE_RE = re.compile(r'\(([^)]+)\)')


def _extract_angle(label: str) -> float:
    m = _ANGLE_RE.search(label)
    if not m:
        raise CircuitError(f"Не удалось извлечь числовой параметр из подписи гейта {label!r}")
    return float(m.group(1))


class Circuit:
    def __init__(self, n_qubits: int, seed: int | None = None):
        self.n = check_qubit_count(n_qubits, MAX_RECOMMENDED_QUBITS)
        # Инструкции хранятся как кортежи вида:
        #   ('gate', label, matrix_or_name, q)
        #   ('two', label, matrix4x4, q0, q1)          # q0=control, q1=target (кроме SWAP)
        #   ('controlled', label, matrix, controls, targets)
        #   ('reset', q)
        #   ('measure', q, cbit_name_or_None)
        #   ('barrier',)
        # label — это то, что видит draw()/repr, matrix_or_name — то, что
        # реально исполняется в _apply_instructions. Раньше label не
        # хранился отдельно, и draw() не мог показать, например, что за
        # параметрический гейт стоит на кубите (всё схлопывалось в 'R'),
        # либо какой гейт стоит под control (всё рисовалось как ● у target).
        self.instructions = []
        self.seed = seed

    def _check(self, *qs: int) -> list[int]:
        return check_qubit_indices(qs, self.n)

    @staticmethod
    def _check_matrix_shape(matrix, k, what="target-кубит(а/ов)"):
        """Проверяет форму матрицы гейта СРАЗУ при добавлении инструкции
        в схему, а не когда-то потом внутри apply_controlled — так
        несовпадение размерности всплывает с понятным сообщением в
        момент ошибки пользователя, а не глубоко в numpy."""
        matrix = np.asarray(matrix)
        expected = 2 ** k
        if matrix.ndim != 2 or matrix.shape != (expected, expected):
            raise GateError(
                f"Матрица гейта имеет форму {matrix.shape}, а для "
                f"{k} {what} ожидается {expected}x{expected}"
            )
        return matrix

    @staticmethod
    def _check_controls_targets(controls, targets):
        """Проверяет отсутствие повторов и пересечений controls/targets.

        Раньше эта проверка была только в controlled_gate() — метод
        control() (используемый, среди прочего, crx/cry/crz/ccx/mcx)
        никак не проверял повторяющиеся controls или control==target на
        этапе построения схемы: ошибка всплывала только при run(), из
        State.apply_controlled, что усложняет отладку (сообщение об
        ошибке появляется далеко от места, где реально была допущена
        ошибка).
        """
        controls = list(controls)
        targets = list(targets)
        if len(set(controls)) != len(controls):
            raise QubitError("controls не должны содержать повторов")
        if len(set(targets)) != len(targets):
            raise QubitError("targets не должны содержать повторов")
        if set(controls) & set(targets):
            raise QubitError("controls и targets должны быть непересекающимися множествами")
        return controls, targets

    # --- однокубитные фиксированные гейты ---
    def _add1(self, name: str, q: int) -> "Circuit":
        q = check_qubit_index(q, self.n)
        self.instructions.append(('gate', name, name, q))
        return self

    def h(self, q: int) -> "Circuit":    return self._add1('H', q)
    def x(self, q: int) -> "Circuit":    return self._add1('X', q)
    def y(self, q: int) -> "Circuit":    return self._add1('Y', q)
    def z(self, q: int) -> "Circuit":    return self._add1('Z', q)
    def s(self, q: int) -> "Circuit":    return self._add1('S', q)
    def sdg(self, q: int) -> "Circuit":  return self._add1('SDG', q)
    def t(self, q: int) -> "Circuit":    return self._add1('T', q)
    def tdg(self, q: int) -> "Circuit":  return self._add1('TDG', q)
    def sx(self, q: int) -> "Circuit":   return self._add1('SX', q)
    def sxdg(self, q: int) -> "Circuit": return self._add1('SXDG', q)

    # --- параметрические однокубитные ---
    # Каждая из четырёх принимает либо число (тогда матрица строится сразу
    # же, как и раньше), либо объект Parameter (тогда матрица откладывается
    # до Circuit.bind() — см. parameters.py).
    def _add_angle_gate(self, kind: str, factory, theta: float | Parameter, q: int) -> "Circuit":
        q = check_qubit_index(q, self.n)
        if isinstance(theta, Parameter):
            label = f"{kind}({theta.name})"
            self.instructions.append(('gate', label, _ParamGate(kind, theta), q))
        else:
            theta = check_real(theta, "theta" if kind != 'P' else "phi")
            label = f"{kind}({theta:.6g})"
            self.instructions.append(('gate', label, factory(theta), q))
        return self

    def rx(self, theta: float | Parameter, q: int) -> "Circuit":
        return self._add_angle_gate('RX', G.Rx, theta, q)

    def ry(self, theta: float | Parameter, q: int) -> "Circuit":
        return self._add_angle_gate('RY', G.Ry, theta, q)

    def rz(self, theta: float | Parameter, q: int) -> "Circuit":
        return self._add_angle_gate('RZ', G.Rz, theta, q)

    def p(self, phi: float | Parameter, q: int) -> "Circuit":
        """Фазовый гейт (он же U1)."""
        return self._add_angle_gate('P', G.Phase, phi, q)

    def u(self, theta: float, phi: float, lam: float, q: int) -> "Circuit":
        """Общий U3(theta, phi, lambda)."""
        q = check_qubit_index(q, self.n)
        theta = check_real(theta, "theta")
        phi = check_real(phi, "phi")
        lam = check_real(lam, "lam")
        label = "U3"
        self.instructions.append(('gate', label, G.U(theta, phi, lam), q)); return self

    # --- двухкубитные ---
    def cnot(self, c: int, t: int) -> "Circuit": return self.cx(c, t)
    def cx(self, c: int, t: int) -> "Circuit":
        c, t = self._check(c, t)
        self.instructions.append(('two', 'CX', G.CNOT_MATRIX, c, t)); return self

    def cy(self, c: int, t: int) -> "Circuit":
        c, t = self._check(c, t)
        self.instructions.append(('two', 'CY', G.CY_MATRIX, c, t)); return self

    def cz(self, c: int, t: int) -> "Circuit":
        c, t = self._check(c, t)
        self.instructions.append(('two', 'CZ', G.CZ_MATRIX, c, t)); return self

    def swap(self, a: int, b: int) -> "Circuit":
        a, b = self._check(a, b)
        self.instructions.append(('two', 'SWAP', G.SWAP_MATRIX, a, b)); return self

    def cp(self, phi: float, c: int, t: int) -> "Circuit":
        """Управляемая фаза (controlled-phase)."""
        c, t = self._check(c, t)
        phi = check_real(phi, "phi")
        label = f"CP({phi:.6g})"
        self.instructions.append(('two', label, G.CPhase(phi), c, t)); return self

    def crx(self, theta: float, c: int, t: int) -> "Circuit":
        theta = check_real(theta, "theta")
        return self.control(G.Rx(theta), [c], t, label=f"RX({theta:.6g})")

    def cry(self, theta: float, c: int, t: int) -> "Circuit":
        theta = check_real(theta, "theta")
        return self.control(G.Ry(theta), [c], t, label=f"RY({theta:.6g})")

    def crz(self, theta: float, c: int, t: int) -> "Circuit":
        theta = check_real(theta, "theta")
        return self.control(G.Rz(theta), [c], t, label=f"RZ({theta:.6g})")

    # --- многокубитные управляемые ---
    def ccx(self, c1: int, c2: int, t: int) -> "Circuit":
        """Toffoli / CCX."""
        return self.control(G.X, [c1, c2], t, label='X')

    def toffoli(self, c1: int, c2: int, t: int) -> "Circuit":
        return self.ccx(c1, c2, t)

    def cswap(self, c: int, a: int, b: int) -> "Circuit":
        """Fredkin."""
        return self.controlled_gate(G.SWAP_MATRIX, [c], [a, b], label='SWAP')

    def fredkin(self, c: int, a: int, b: int) -> "Circuit":
        return self.cswap(c, a, b)

    def mcx(self, controls, target: int) -> "Circuit":
        """Multi-controlled X: любое число control-кубитов."""
        return self.control(G.X, list(controls), target, label='X')

    def control(self, gate_or_name, controls, target: int, label: str | None = None) -> "Circuit":
        """Общий способ добавить управляемый однокубитный гейт (по имени или матрице)
        с произвольным числом controls."""
        is_named = isinstance(gate_or_name, str)
        mat = G.get(gate_or_name) if is_named else np.asarray(gate_or_name, dtype=complex)
        controls = check_qubit_indices(controls, self.n, "Control-кубит")
        target = check_qubit_index(target, self.n, "Target-кубит")
        controls, targets = self._check_controls_targets(controls, [target])
        mat = self._check_matrix_shape(mat, 1, "target-кубит")
        if not is_named:
            # Гейты из встроенного реестра (по имени) уже проверены на
            # унитарность один раз при импорте gates.py; пользовательскую
            # матрицу нужно проверить здесь и сейчас — иначе
            # неунитарная матрица молча испортит нормировку состояния
            # при run(), без единого предупреждения (было
            # воспроизведено: apply_single(2*I, 0) тихо удваивал норму
            # вектора состояния).
            mat = check_unitary(mat, what="Пользовательский гейт в control()")
        if label is None:
            label = gate_or_name if is_named else 'U'
        self.instructions.append(('controlled', label, mat, controls, targets))
        return self

    def controlled_gate(self, matrix, controls, targets, label: str | None = None) -> "Circuit":
        """Максимально общий вариант: произвольная k-кубитная матрица на targets,
        под управлением controls."""
        controls = check_qubit_indices(controls, self.n, "Control-кубит")
        targets = check_qubit_indices(targets, self.n, "Target-кубит")
        controls, targets = self._check_controls_targets(controls, targets)
        matrix = self._check_matrix_shape(matrix, len(targets))
        matrix = check_unitary(matrix, what="Пользовательский гейт в controlled_gate()")
        if label is None:
            label = 'U'
        self.instructions.append(('controlled', label, matrix, controls, targets))
        return self

    # --- служебное ---
    def barrier(self) -> "Circuit":
        self.instructions.append(('barrier',)); return self

    def reset(self, q: int) -> "Circuit":
        q = check_qubit_index(q, self.n)
        self.instructions.append(('reset', q)); return self

    def measure(self, q: int, cbit: str | None = None) -> "Circuit":
        """Измеряет кубит q. cbit — необязательное имя классического бита
        (например, из ClassicalRegister) для более читаемых результатов —
        см. Circuit.measured_labels(). Если не задано, результат
        подписывается как 'qN'.

        Раньше тип cbit никак не проверялся: measure(0, cbit=42) молча
        принимался и портил контракт measured_labels() (документированный
        как список строк) для любого кода, который потом форматирует или
        сравнивает эти подписи как str — ошибка обнаружилась бы не здесь,
        а в произвольном месте ниже по потоку использования результата.
        """
        q = check_qubit_index(q, self.n)
        if cbit is not None and not isinstance(cbit, str):
            raise QSimTypeError(
                f"cbit должен быть строкой или None, получено "
                f"{type(cbit).__name__}: {cbit!r}"
            )
        self.instructions.append(('measure', q, cbit)); return self

    def measure_all(self) -> "Circuit":
        for q in range(self.n):
            self.measure(q)
        return self

    # --- параметризованные схемы ---
    @property
    def parameters(self):
        """Список несвязанных Parameter, использованных в схеме (порядок —
        первое появление, без повторов одного и того же объекта)."""
        params, seen = [], set()
        for ins in self.instructions:
            if ins[0] == 'gate' and isinstance(ins[2], _ParamGate):
                p = ins[2].param
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    def bind(self, values: dict) -> "Circuit":
        """Возвращает НОВУЮ схему с указанными Parameter, заменёнными на
        конкретные вещественные значения; исходная схема не меняется.

        values: dict, ключи — объекты Parameter (сопоставляются по
        identity — надёжно даже при одинаковых именах разных Parameter,
        см. Parameter.__doc__) или их имена ``str`` (сопоставляются по
        имени — удобно, когда объект Parameter уже не под рукой, но
        может задеть не тот Parameter, если несколько из них называются
        одинаково; при коллизии приоритет у привязки по объекту).
        Параметры схемы, не упомянутые в values, остаются несвязанными в
        возвращаемой схеме (можно связывать по частям).
        """
        by_id = {}
        by_name = {}
        for key, val in values.items():
            if isinstance(key, Parameter):
                by_id[id(key)] = check_real(val, f"значение параметра {key.name!r}")
            else:
                by_name[key] = check_real(val, f"значение параметра {key!r}")

        new = Circuit(self.n, seed=self.seed)
        for ins in self.instructions:
            if ins[0] == 'gate' and isinstance(ins[2], _ParamGate):
                _, _label, pg, q = ins
                if id(pg.param) in by_id:
                    val = by_id[id(pg.param)]
                elif pg.param.name in by_name:
                    val = by_name[pg.param.name]
                else:
                    new.instructions.append(ins)  # остаётся несвязанным
                    continue
                label = f"{pg.kind}({val:.6g})"
                new.instructions.append(('gate', label, pg.build_matrix(val), q))
            else:
                new.instructions.append(ins)
        return new

    # --- запуск ---
    def _has_reset(self):
        return any(ins[0] == 'reset' for ins in self.instructions)

    def _measured_qubits(self):
        """Кубиты, для которых в схеме реально стоит measure(...), по
        возрастанию индекса. Если measure() ни разу не вызывался (но
        были запрошены shots), по умолчанию считаем, что интересны все
        кубиты — это сохраняет прежнее поведение для схем без явных
        измерений."""
        qs = sorted({ins[1] for ins in self.instructions if ins[0] == 'measure'})
        return qs if qs else list(range(self.n))

    def measured_labels(self) -> list[str]:
        """Метки (имя cbit, если оно было передано в measure(q, cbit=...),
        иначе 'qN') для битов ИМЕННО в том порядке, в котором они
        появляются в ключах counts, возвращаемых run(shots=...) —
        то есть в порядке _measured_qubits() (по возрастанию индекса
        кубита). При повторном measure() одного кубита с разными cbit
        используется последнее переданное имя."""
        names = {}
        for ins in self.instructions:
            if ins[0] == 'measure':
                _, q, cbit = ins
                if cbit is not None:
                    names[q] = cbit
        return [names.get(q, f"q{q}") for q in self._measured_qubits()]

    def _measurements_are_terminal(self):
        """True, если все explicit measure находятся в терминальной части.

        Тогда без reset можно применить отложенное измерение: один прогон
        схемы + сэмплирование конечного состояния shots раз.
        """
        seen_measure = False
        for ins in self.instructions:
            kind = ins[0]
            if kind == 'measure':
                seen_measure = True
            elif seen_measure and kind not in ('measure', 'barrier'):
                return False
        return seen_measure

    def _supports_deferred_measurement(self):
        """True, если измерение(-я) схемы, если они вообще есть, можно
        безопасно отложить до одного финального сэмплирования, вместо
        того чтобы прогонять каждый shot по отдельности.

        Это верно в двух случаях: (1) в схеме вообще нет explicit
        measure() — тогда просто сэмплируем конечное распределение;
        (2) все measure() стоят в самом конце схемы (после них нет
        других гейтов, только другие measure/barrier) — тогда по
        принципу отложенного измерения результат совпадает с
        измерением сразу после его "настоящего" места в схеме.
        В обоих случаях необходимо отсутствие reset() в схеме, потому
        что reset — это самая настоящая коллапсирующая операция с
        обратной связью (применяется X, если исход измерения был 1),
        которая обязана быть выполнена per-shot.

        Раньше в run() эти два случая были оформлены как два отдельных,
        почти дословно одинаковых блока кода (различавшихся только тем,
        как выведено это условие) — унификация не меняет поведения, но
        убирает дублирование и потенциальный источник рассинхронизации
        при будущих правках.
        """
        if self._has_reset():
            return False
        return self._measurements_are_terminal() or not any(
            ins[0] == 'measure' for ins in self.instructions
        )

    def _apply_instructions(self, state, sim, collapse_measurements,
                            measurement_results=None, noise=None):
        """Прогоняет все инструкции схемы на переданных state/sim.

        collapse_measurements=True -> ``measure`` реально коллапсирует.
        При переданном measurement_results туда записывается последний
        результат измерения каждого кубита.

        collapse_measurements=False -> ``measure`` откладывается до конца.
        Это безопасно только для схем без reset, где все measure находятся
        в терминальной части схемы.

        noise — необязательный NoiseModel (см. noise.py). Если задан, после
        каждого гейта на каждом затронутом им кубите разыгрывается и
        применяется канал шума этого кубита (если он задан в модели).
        """
        def _apply_noise(qubits):
            if noise is None:
                return
            if len(qubits) == 2:
                # Двухкубитный коррелированный канал для этой конкретной
                # пары (если зарегистрирован) полностью заменяет
                # независимый шум по отдельным кубитам для данной
                # инструкции — иначе шум был бы учтён дважды. См.
                # NoiseModel.add_two_qubit_channel().
                two_q_channel = noise.two_qubit_channel_for(qubits[0], qubits[1])
                if two_q_channel is not None:
                    two_q_channel.sample_and_apply(state, qubits[0], qubits[1], sim.rng)
                    return
            for q in qubits:
                channel = noise.channel_for(q)
                if channel is not None:
                    channel.sample_and_apply(state, q, sim.rng)

        for ins in self.instructions:
            kind = ins[0]
            if kind == 'gate':
                _, label, g, q = ins
                if isinstance(g, _ParamGate):
                    raise CircuitError(
                        f"Инструкция {label} на кубите {q} содержит несвязанный "
                        f"параметр {g.param.name!r} — вызовите circuit.bind({{...}}) "
                        f"перед run()."
                    )
                mat = G.get(g) if isinstance(g, str) else g
                state.apply_single(mat, q)
                _apply_noise([q])
            elif kind == 'two':
                _, _label, mat, q0, q1 = ins
                state.apply_two_qubit(mat, q0, q1)
                _apply_noise([q0, q1])
            elif kind == 'controlled':
                _, _label, mat, controls, targets = ins
                state.apply_controlled(mat, controls, targets)
                _apply_noise(list(controls) + list(targets))
            elif kind == 'reset':
                _, q = ins
                # reset — настоящий коллапс (измерение + коррекция), поэтому
                # он всегда исполняется независимо от shots/collapse_measurements.
                outcome = sim.measure(q, collapse=True)
                if outcome == 1:
                    state.apply_single(G.X, q)
            elif kind == 'measure':
                _, q, _cbit = ins
                if collapse_measurements:
                    outcome = sim.measure(q, collapse=True)
                    if measurement_results is not None:
                        # Для повторного measure(q) сохраняем последний результат.
                        measurement_results[q] = outcome
            elif kind == 'barrier':
                pass
            else:
                raise CircuitError(f"Неизвестная инструкция: {ins}")

    def run(self, shots: int | None = None, noise=None):
        """Прогоняет схему.

        noise — необязательный NoiseModel (см. noise.py). Присутствие шума
        требует свежего случайного выбора канала Крауса на каждом
        зашумлённом месте для каждого shot независимо, поэтому наличие
        noise, как и mid-circuit measurement/reset, переводит run() на
        по-shot-ный путь (см. _supports_deferred_measurement) — даже если
        сама схема формально допускала бы отложенное измерение.
        """
        if self.parameters:
            unbound = ', '.join(p.name for p in self.parameters)
            raise CircuitError(
                f"Схема содержит несвязанные параметры ({unbound}) — "
                f"вызовите circuit.bind({{...}}) перед run()."
            )

        if shots is None:
            state = State(self.n)
            sim = Simulator(state, seed=self.seed)
            self._apply_instructions(state, sim, collapse_measurements=True, noise=noise)
            return state

        # Валидируется здесь и сразу для ОБОИХ путей ниже. Раньше
        # отрицательные/нецелые shots проверялись только внутри
        # Simulator.sample(), которая вызывается лишь на "быстром" пути
        # (без reset и без mid-circuit измерений). На "медленном" пути
        # (см. ветку ниже) проверки не было вовсе: ``for _ in
        # range(shots)`` с отрицательным shots просто не делает ни одной
        # итерации, и функция молча возвращала ``({}, None)`` вместо
        # ошибки — притом что None нарушает контракт «run(shots=...)
        # возвращает (counts, state)», на который полагаются вызывающие.
        shots = check_shots(shots)

        # Только реально измеренные кубиты попадают в ключи counts.
        measured = self._measured_qubits()

        if noise is None and self._supports_deferred_measurement():
            state = State(self.n)
            sim = Simulator(state, seed=self.seed)
            self._apply_instructions(state, sim, collapse_measurements=False)
            return sim.sample(shots=shots, qubits=measured), state

        # Mid-circuit measure и/или reset требуют отдельного прогона каждого shot:
        # случайный исход должен попасть в последующие инструкции.
        # Переиспользуем один State и один Simulator между shots. Это сохраняет
        # независимые случайные исходы через один RNG, но не создаёт объект State
        # и Simulator заново для каждого выстрела.
        state = State(self.n)
        sim = Simulator(state, seed=self.seed)
        counts = {}
        has_measure = any(ins[0] == 'measure' for ins in self.instructions)

        # Схему нужно прогнать хотя бы один раз, даже если shots == 0 —
        # иначе некому вернуть осмысленный final_state (раньше при
        # shots=0 на этом пути final_state оставался None: цикл
        # ``for _ in range(0)`` не выполнялся вовсе, что нарушало
        # контракт метода). "Лишний" технический прогон при shots == 0
        # не попадает в counts.
        for shot_idx in range(max(shots, 1)):
            state.reset()
            measurement_results = {} if has_measure else None

            self._apply_instructions(
                state, sim,
                collapse_measurements=has_measure,
                measurement_results=measurement_results,
                noise=noise,
            )

            if shot_idx >= shots:
                break  # технический прогон для shots == 0 — не считаем

            if has_measure:
                # counts отражает последние explicit measure для каждого
                # измеренного кубита, а не состояние после последующих gate/reset.
                # В медленном пути каждый measured-кубит действительно
                # проходит через explicit measure, поэтому fallback больше
                # не нужен.
                bits = ''.join(str(measurement_results[q]) for q in measured)
            else:
                probs = state.probabilities()
                probs = probs / probs.sum()
                outcome_idx = int(sim.rng.choice(state.dim, p=probs))
                full_bits = format(outcome_idx, f'0{self.n}b')
                bits = ''.join(full_bits[q] for q in measured)

            counts[bits] = counts.get(bits, 0) + 1

        # Копируем состояние ОДИН раз, после цикла, а не на каждой
        # итерации — раньше ``final_state = state.copy()`` выполнялось
        # внутри цикла на каждом shot, то есть делало O(shots) полных
        # копий statevector, хотя нужна была только последняя.
        final_state = state.copy()

        return counts, final_state

    # --- визуализация ---
    def draw(self) -> str:
        """Возвращает выровненную ASCII-схему.

        Каждая инструкция занимает одну колонку фиксированной ширины,
        рассчитанной по самой длинной подписи этой колонки. Для
        controlled/multi-qubit инструкций соединительные линии проходят
        через всю ширину колонки.
        """
        def column_width(ins):
            kind = ins[0]
            if kind in ('gate', 'two', 'controlled'):
                return max(5, len(ins[1]) + 2)
            if kind in ('reset', 'measure', 'barrier'):
                return 5
            raise CircuitError(f"Неизвестная инструкция: {ins}")

        widths = [column_width(ins) for ins in self.instructions]
        lines = [f"q{i}: " for i in range(self.n)]

        def box(label, width):
            inner = max(1, width - 2)
            return f"[{label:^{inner}}]"

        def centered(symbol, width):
            return symbol.center(width)

        for ins, width in zip(self.instructions, widths):
            kind = ins[0]

            if kind == 'gate':
                _, label, _g, q = ins
                for i in range(self.n):
                    lines[i] += box(label, width) if i == q else ('─' * width)

            elif kind == 'two':
                _, label, _mat, q0, q1 = ins
                lo, hi = min(q0, q1), max(q0, q1)

                if label == 'SWAP':
                    symbols = {q0: '×', q1: '×'}
                elif label == 'CX':
                    symbols = {q0: '●', q1: '⊕'}
                elif label == 'CY':
                    symbols = {q0: '●', q1: 'Y'}
                elif label == 'CZ':
                    symbols = {q0: '●', q1: '●'}
                elif label.startswith('CP('):
                    # Показываем, что это именно controlled-phase, и не теряем phi.
                    symbols = {q0: '●', q1: box(label, width)}
                else:
                    symbols = {q0: '●', q1: '■'}

                for i in range(self.n):
                    if i in symbols:
                        lines[i] += centered(symbols[i], width)
                    elif lo < i < hi:
                        lines[i] += centered('│', width)
                    else:
                        lines[i] += '─' * width

            elif kind == 'controlled':
                _, label, _mat, controls, targets = ins

                if not controls:
                    # controlled_gate([], ...) эквивалентен обычному gate на targets:
                    # никаких control-точек и вертикальных соединителей рисовать нельзя.
                    target_symbols = {
                        q: ('×' if label == 'SWAP' else box(label, width))
                        for q in targets
                    }
                    for i in range(self.n):
                        lines[i] += (
                            target_symbols[i] if i in target_symbols else '─' * width
                        )
                    continue

                involved = list(controls) + list(targets)
                lo, hi = min(involved), max(involved)

                if label == 'X':
                    target_symbols = {q: '⊕' for q in targets}
                elif label == 'SWAP':
                    target_symbols = {q: '×' for q in targets}
                else:
                    target_symbols = {q: box(label, width) for q in targets}

                for i in range(self.n):
                    if i in controls:
                        lines[i] += centered('●', width)
                    elif i in target_symbols:
                        lines[i] += target_symbols[i]
                    elif lo < i < hi:
                        lines[i] += centered('│', width)
                    else:
                        lines[i] += '─' * width

            elif kind == 'reset':
                _, q = ins
                for i in range(self.n):
                    lines[i] += centered('[R]', width) if i == q else ('─' * width)

            elif kind == 'measure':
                _, q, _cbit = ins
                for i in range(self.n):
                    lines[i] += centered('[M]', width) if i == q else ('─' * width)

            elif kind == 'barrier':
                for i in range(self.n):
                    lines[i] += centered('║', width)

        return '\n'.join(lines)

    # --- экспорт ---
    # Гейты, которые есть по имени в САМОЙ ПЕРВОЙ версии qelib1.inc
    # (OpenQASM 2.0, arXiv:1707.03429): u3, u2, u1, cx, id, x, y, z, h,
    # s, sdg, t, tdg, rx, ry, rz, cz, cy, ch, ccx, crz, cu1, cu3. Всё,
    # чего в этом списке нет (sx, sxdg, swap, crx, cry, cswap — они
    # появились в более поздних/расширенных дистрибутивах qelib1.inc,
    # например поставляемом с Qiskit), экспортируется НЕ как вызов
    # гейта с этим именем, а раскладывается ("транспилируется") в
    # эквивалентную последовательность гейтов из гарантированного
    # набора — тогда экспортированный QASM корректно грузится любым
    # инструментом, использующим строгий/оригинальный qelib1.inc, а не
    # только тем, чей qelib1.inc это имя случайно определяет.
    # Все разложения ниже проверены на точное (не только с точностью до
    # глобальной фазы, там где это важно, т.е. для управляемых гейтов)
    # совпадение с исходной матрицей — см. test_features.py /
    # TestQasmExport.
    _QASM_NAMED = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'SDG': 'sdg', 'T': 't', 'TDG': 'tdg',
    }

    @staticmethod
    def _qasm_sx(q, dagger=False):
        # SX = e^{i*pi/4} * U3(pi/2,-pi/2,pi/2); SXDG = SX^† =
        # e^{-i*pi/4} * U3(-pi/2,-pi/2,pi/2). Глобальная фаза физически
        # ненаблюдаема на однокубитном гейте, поэтому её можно опустить.
        theta = -np.pi / 2 if dagger else np.pi / 2
        return [f'u3({theta:.12g},{-np.pi/2:.12g},{np.pi/2:.12g}) q[{q}];']

    @staticmethod
    def _qasm_crx(theta, c, t):
        # Стандартное разложение controlled-Rx через cx/u1/u3 (точное,
        # не только с точностью до глобальной фазы — проверено численно).
        return [
            f'u1({np.pi/2:.12g}) q[{t}];',
            f'cx q[{c}],q[{t}];',
            f'u3({-theta/2:.12g},0,0) q[{t}];',
            f'cx q[{c}],q[{t}];',
            f'u3({theta/2:.12g},{-np.pi/2:.12g},0) q[{t}];',
        ]

    @staticmethod
    def _qasm_cry(theta, c, t):
        # controlled-Ry через ry/cx (точное разложение).
        return [
            f'ry({theta/2:.12g}) q[{t}];',
            f'cx q[{c}],q[{t}];',
            f'ry({-theta/2:.12g}) q[{t}];',
            f'cx q[{c}],q[{t}];',
        ]

    @staticmethod
    def _qasm_swap(a, b):
        # SWAP(a,b) = CX(a,b) CX(b,a) CX(a,b) — стандартная точная
        # декомпозиция через три cx.
        return [f'cx q[{a}],q[{b}];', f'cx q[{b}],q[{a}];', f'cx q[{a}],q[{b}];']

    @staticmethod
    def _qasm_cswap(c, a, b):
        # Fredkin(c,a,b) = CX(b,a) CCX(c,a,b) CX(b,a) — точная
        # декомпозиция через ccx (Toffoli, гарантированно в базовом
        # наборе) и cx.
        return [
            f'cx q[{b}],q[{a}];',
            f'ccx q[{c}],q[{a}],q[{b}];',
            f'cx q[{b}],q[{a}];',
        ]

    def to_qasm(self) -> str:
        """Экспортирует схему в OpenQASM 2.0.

        Поддержано: все фиксированные однокубитные гейты (включая sx/sxdg),
        rx/ry/rz/p (как u1), cx/cy/cz/swap/cp (как cu1), crx/cry/crz,
        toffoli (ccx), fredkin (cswap), control()/controlled_gate() с
        пустым списком controls (экспортируется как обычный
        неуправляемый гейт на targets), reset, measure, barrier.

        Гейты, отсутствующие в самой первой/строгой версии qelib1.inc
        (sx, sxdg, swap, crx, cry, cswap), не полагаются на то, что
        конкретный инструмент, читающий экспортированный QASM, включил
        их в свою копию qelib1.inc — вместо вызова гейта по имени они
        раскладываются в эквивалентную последовательность гейтов из
        гарантированно присутствующего базового набора (u1/u3/cx/ccx/ry).

        НЕ поддержано (бросает CircuitError с понятным сообщением, а не
        тихо теряет часть схемы): несвязанные Parameter (сначала
        вызовите bind()), u()/U3, control()/controlled_gate() с
        произвольной пользовательской матрицей (кроме случая с пустыми
        controls, см. выше), mcx с более чем двумя controls — для всего
        этого потребовалась бы полная декомпозиция производной унитарной
        матрицы в базисные гейты (транспиляция общего вида), которая не
        реализована.

        Ограничение точности: числовые параметры, полученные из подписи
        инструкции (см. draw()), сериализуются с точностью ``%.6g`` — этого
        достаточно для визуальной проверки и большинства практических
        нужд, но не для побитового round-trip. Параметры, вычисляемые
        заново при разложении гейтов (например, ``theta/2`` для crx/cry),
        сериализуются с точностью ``%.12g``, чтобы не терять точность на
        лишнем шаге сериализации/парсинга внутри самого разложения.
        """
        lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{self.n}];']

        explicit_measured = sorted({ins[1] for ins in self.instructions if ins[0] == 'measure'})
        cbit_index = {q: i for i, q in enumerate(explicit_measured)}
        if explicit_measured:
            lines.append(f'creg c[{len(explicit_measured)}];')

        for ins in self.instructions:
            kind = ins[0]

            if kind == 'gate':
                _, label, g, q = ins
                if isinstance(g, _ParamGate):
                    raise CircuitError(
                        f"Нельзя экспортировать схему с несвязанным параметром "
                        f"{g.param.name!r} — сначала вызовите circuit.bind({{...}})."
                    )
                if isinstance(g, str) and g in self._QASM_NAMED:
                    lines.append(f'{self._QASM_NAMED[g]} q[{q}];')
                elif isinstance(g, str) and g == 'SX':
                    lines.extend(self._qasm_sx(q, dagger=False))
                elif isinstance(g, str) and g == 'SXDG':
                    lines.extend(self._qasm_sx(q, dagger=True))
                elif label.startswith('RX('):
                    lines.append(f'rx({_extract_angle(label)}) q[{q}];')
                elif label.startswith('RY('):
                    lines.append(f'ry({_extract_angle(label)}) q[{q}];')
                elif label.startswith('RZ('):
                    lines.append(f'rz({_extract_angle(label)}) q[{q}];')
                elif label.startswith('P('):
                    lines.append(f'u1({_extract_angle(label)}) q[{q}];')
                else:
                    raise CircuitError(
                        f"Гейт {label!r} нельзя экспортировать в OpenQASM 2.0 "
                        f"(например, u()/U3 — транспиляция не реализована)"
                    )

            elif kind == 'two':
                _, label, _mat, q0, q1 = ins
                if label == 'CX':
                    lines.append(f'cx q[{q0}],q[{q1}];')
                elif label == 'CY':
                    lines.append(f'cy q[{q0}],q[{q1}];')
                elif label == 'CZ':
                    lines.append(f'cz q[{q0}],q[{q1}];')
                elif label == 'SWAP':
                    lines.extend(self._qasm_swap(q0, q1))
                elif label.startswith('CP('):
                    lines.append(f'cu1({_extract_angle(label)}) q[{q0}],q[{q1}];')
                else:
                    raise CircuitError(f"Двухкубитный гейт {label!r} нельзя экспортировать в OpenQASM 2.0")

            elif kind == 'controlled':
                _, label, _mat, controls, targets = ins

                if not controls:
                    # controlled_gate([], ...) / control(..., []) не
                    # управляется никем — это просто обычный гейт на
                    # targets. Раньше это не совпадало ни с одним
                    # известным паттерном ниже (все они ожидают
                    # len(controls) >= 1) и падало с CircuitError, хотя
                    # эта конфигурация полностью определена и уже
                    # поддерживается на уровне run()/draw().
                    if label == 'X' and len(targets) == 1:
                        lines.append(f'x q[{targets[0]}];')
                    elif label == 'SWAP' and len(targets) == 2:
                        lines.extend(self._qasm_swap(targets[0], targets[1]))
                    elif label[:2] in ('RX', 'RY', 'RZ') and len(targets) == 1:
                        gate_name = {'RX': 'rx', 'RY': 'ry', 'RZ': 'rz'}[label[:2]]
                        lines.append(f'{gate_name}({_extract_angle(label)}) q[{targets[0]}];')
                    elif label.startswith('P(') and len(targets) == 1:
                        lines.append(f'u1({_extract_angle(label)}) q[{targets[0]}];')
                    else:
                        raise CircuitError(
                            f"Гейт {label!r} с пустым списком controls нельзя "
                            f"экспортировать в OpenQASM 2.0 (произвольная "
                            f"пользовательская матрица — транспиляция не реализована)"
                        )
                    continue

                if label == 'X' and len(controls) == 1 and len(targets) == 1:
                    lines.append(f'cx q[{controls[0]}],q[{targets[0]}];')
                elif label == 'X' and len(controls) == 2 and len(targets) == 1:
                    lines.append(f'ccx q[{controls[0]}],q[{controls[1]}],q[{targets[0]}];')
                elif label.startswith('RX(') and len(controls) == 1 and len(targets) == 1:
                    lines.extend(self._qasm_crx(_extract_angle(label), controls[0], targets[0]))
                elif label.startswith('RY(') and len(controls) == 1 and len(targets) == 1:
                    lines.extend(self._qasm_cry(_extract_angle(label), controls[0], targets[0]))
                elif label.startswith('RZ(') and len(controls) == 1 and len(targets) == 1:
                    # crz есть уже в самой первой версии qelib1.inc —
                    # разложение не требуется.
                    lines.append(f'crz({_extract_angle(label)}) q[{controls[0]}],q[{targets[0]}];')
                elif label.startswith('P(') and len(controls) == 1 and len(targets) == 1:
                    lines.append(f'cu1({_extract_angle(label)}) q[{controls[0]}],q[{targets[0]}];')
                elif label == 'SWAP' and len(controls) == 1 and len(targets) == 2:
                    lines.extend(self._qasm_cswap(controls[0], targets[0], targets[1]))
                else:
                    raise CircuitError(
                        f"Управляемый гейт {label!r} с {len(controls)} controls/"
                        f"{len(targets)} targets нельзя напрямую экспортировать в "
                        f"базовый OpenQASM 2.0 (нужна декомпозиция/транспиляция)"
                    )

            elif kind == 'reset':
                _, q = ins
                lines.append(f'reset q[{q}];')

            elif kind == 'measure':
                _, q, _cbit = ins
                lines.append(f'measure q[{q}] -> c[{cbit_index[q]}];')

            elif kind == 'barrier':
                lines.append('barrier ' + ','.join(f'q[{i}]' for i in range(self.n)) + ';')

            else:
                raise CircuitError(f"Неизвестная инструкция: {ins}")

        return '\n'.join(lines)

    def __repr__(self):
        return f"Circuit(n={self.n}, ops={len(self.instructions)})"
