"""Модели шума для statevector-симулятора.

Полноценная эволюция общей матрицы плотности с произвольными каналами
Крауса требует O(4^n) памяти вместо O(2^n) для statevector — для
statevector-бэкенда это неприемлемо уже при паре десятков кубитов.
Вместо этого шум здесь реализован через приём **quantum trajectories**
(stochastic unraveling): на каждом зашумлённом месте схемы, для каждого
отдельного shot, случайно (согласно вероятностям Крауса) выбирается один
конкретный оператор канала и применяется к чистому вектору состояния,
как обычный (не обязательно унитарный) однокубитный оператор с
последующей перенормировкой. Усреднение результатов по многим shots
статистически сходится к точной эволюции матрицы плотности — это
стандартная и точная техника (Monte Carlo wave function / quantum
trajectories), а не приближение более низкого порядка.

Из этого следует важное следствие: шум наблюдаем только через
`Circuit.run(shots=..., noise=...)` — каждый shot должен прогонять
схему заново со свежим случайным выбором Крауса на каждом зашумлённом
месте (как и mid-circuit measurement/reset, это переводит run() на
"медленный", по-shot-ный путь).
"""
from __future__ import annotations

import numpy as np

from . import gates as G
from ._common import check_real
from ._errors import GateError, QSimValueError


class NoiseChannel:
    """Канал шума, заданный операторами Крауса {K_i}.

    Корректный (TPCP) канал должен удовлетворять sum_i K_i^dagger K_i = I
    — это проверяется один раз при создании канала, а не на каждом shot.
    """

    def __init__(self, kraus_ops, name: str = "channel"):
        self.kraus_ops = [np.asarray(k, dtype=complex) for k in kraus_ops]
        self.name = name
        if not self.kraus_ops:
            raise QSimValueError(f"Канал {name!r} должен содержать хотя бы один оператор Крауса")
        self._validate()

    def _validate(self):
        dim = self.kraus_ops[0].shape[0]
        total = np.zeros((dim, dim), dtype=complex)
        self._kraus_M = []  # K_i^dagger K_i для каждого i — считаем один раз
        # здесь (уже нужно для суммы ниже) и переиспользуем в
        # sample_and_apply, вместо того чтобы каждый shot заново умножать
        # маленькие 2x2-матрицы.
        for k in self.kraus_ops:
            if k.ndim != 2 or k.shape != (dim, dim):
                raise GateError(
                    f"Все операторы Крауса канала {self.name!r} должны быть "
                    f"квадратными матрицами одного размера, получена форма {k.shape}"
                )
            m = k.conj().T @ k
            self._kraus_M.append(m)
            total += m
        if not np.allclose(total, np.eye(dim), atol=1e-8):
            raise GateError(
                f"Операторы Крауса канала {self.name!r} не образуют корректное "
                f"TPCP-отображение (sum K_i^dagger K_i != I)"
            )

    def sample_and_apply(self, state, qubit: int, rng) -> None:
        """Разыгрывает исход канала на одном кубите state и применяет его
        (мутирует state.vector на месте, с перенормировкой).

        Раньше вероятность каждого исхода p_i = <psi|K_i^dagger K_i|psi>
        считалась "в лоб": на каждый Kraus-оператор делалась полная копия
        состояния (state.copy(), O(2^n)) и полное применение оператора
        (apply_single, ещё O(2^n)), а результирующий вектор целиком
        сохранялся в списке trial_vectors до конца цикла — то есть на
        каждом зашумлённом месте каждого shot'а тратилось O(k * 2^n)
        и времени, и памяти (k копий вектора состояния одновременно) ради
        k чисел (вероятностей).

        K_i^dagger K_i — это 2x2 эрмитов оператор на ОДНОМ кубите, поэтому
        его среднее по состоянию можно получить из приведённой (2x2)
        матрицы плотности этого кубита: p_i = trace(rho_q @ K_i^dagger K_i)
        — тем же приёмом, что и Simulator.bloch_vector(). Сама rho_q
        считается один раз за O(2^n) (без копирования state целиком), а
        дальше вероятности — это k тривиальных операций над матрицами 2x2.
        Полное O(2^n)-применение оператора к state нужно сделать только
        один раз — для того исхода, который в итоге выпал.
        """
        t = np.moveaxis(state.tensor, qubit, 0).reshape(2, -1)
        rho = t @ t.conj().T  # приведённая матрица плотности кубита qubit

        raw_probs = np.array(
            [max(float(np.trace(m @ rho).real), 0.0) for m in self._kraus_M]
        )
        total = raw_probs.sum()
        if total <= 1e-12:
            # Вырожденный случай (численно нулевая полная вероятность) —
            # состояние не меняем, вместо деления на ноль.
            return
        choice = int(rng.choice(len(self.kraus_ops), p=raw_probs / total))

        state.apply_single(self.kraus_ops[choice], qubit)
        state.vector /= np.sqrt(raw_probs[choice])
        state.normalize()

    def __repr__(self):
        return f"NoiseChannel({self.name!r}, {len(self.kraus_ops)} Kraus ops)"


def depolarizing_channel(p: float) -> NoiseChannel:
    """Деполяризующий канал с параметром p: с вероятностью p происходит
    случайная ошибка Паули (равновероятно X, Y или Z), с вероятностью
    1-p — ничего не происходит. Это стандартная "twirl"-параметризация
    (Kraus: sqrt(1-p)*I, sqrt(p/3)*{X,Y,Z}).

    ВАЖНО про интерпретацию p: из-за этой параметризации <Z> заданного
    базисного состояния после канала равен не (1 - p), а (1 - 4p/3) —
    отличие от более грубой интуиции "p — это доля декогеренции к
    полностью смешанному состоянию". При p=3/4 канал деполяризует
    полностью (rho -> I/2); значения p из (3/4, 1] физически осмысленны
    (канал всё ещё TPCP), но дают <Z> < 0 для входа |0>, что может быть
    контринтуитивно — это следствие выбранной параметризации, а не
    ошибка."""
    p = check_real(p, "p")
    if not (0.0 <= p <= 1.0):
        raise QSimValueError(f"Вероятность деполяризации должна быть в [0, 1], получено {p}")
    return NoiseChannel([
        np.sqrt(1 - p) * G.I,
        np.sqrt(p / 3) * G.X,
        np.sqrt(p / 3) * G.Y,
        np.sqrt(p / 3) * G.Z,
    ], name=f"depolarizing(p={p:.4g})")


def bit_flip_channel(p: float) -> NoiseChannel:
    """С вероятностью p применяется X (bit-flip)."""
    p = check_real(p, "p")
    if not (0.0 <= p <= 1.0):
        raise QSimValueError(f"Вероятность bit-flip должна быть в [0, 1], получено {p}")
    return NoiseChannel([np.sqrt(1 - p) * G.I, np.sqrt(p) * G.X], name=f"bit_flip(p={p:.4g})")


def phase_flip_channel(p: float) -> NoiseChannel:
    """С вероятностью p применяется Z (phase-flip)."""
    p = check_real(p, "p")
    if not (0.0 <= p <= 1.0):
        raise QSimValueError(f"Вероятность phase-flip должна быть в [0, 1], получено {p}")
    return NoiseChannel([np.sqrt(1 - p) * G.I, np.sqrt(p) * G.Z], name=f"phase_flip(p={p:.4g})")


class TwoQubitNoiseChannel:
    """Двухкубитный канал шума, заданный операторами Крауса на 4-мерном
    (два кубита) пространстве.

    NoiseChannel (см. выше) зашумляет кубиты по отдельности: даже если
    один и тот же NoiseChannel применяется независимо к обоим кубитам
    двухкубитного гейта, получившийся шум по построению факторизуется
    (некоррелированная ошибка на каждом кубите) — он в принципе не может
    выразить шум, коррелированный между двумя кубитами (например,
    двухкубитную деполяризацию CX, где типичная физическая ошибка — это
    одновременная случайная ошибка Паули на ОБОИХ кубитах, а не на
    каждом по отдельности). TwoQubitNoiseChannel — та же идея
    quantum-trajectories (см. модуль docstring), что и у NoiseChannel,
    но операторы Крауса действуют на пару кубитов как единое целое.

    Используется через ``NoiseModel.add_two_qubit_channel(q0, q1, ch)``.
    """

    def __init__(self, kraus_ops, name: str = "two_qubit_channel"):
        self.kraus_ops = [np.asarray(k, dtype=complex) for k in kraus_ops]
        self.name = name
        if not self.kraus_ops:
            raise QSimValueError(f"Канал {name!r} должен содержать хотя бы один оператор Крауса")
        self._validate()

    def _validate(self):
        dim = self.kraus_ops[0].shape[0]
        if dim != 4:
            raise GateError(
                f"Операторы Крауса TwoQubitNoiseChannel {self.name!r} должны быть "
                f"4x4 (два кубита), получена размерность {dim}"
            )
        total = np.zeros((dim, dim), dtype=complex)
        self._kraus_M = []
        for k in self.kraus_ops:
            if k.ndim != 2 or k.shape != (dim, dim):
                raise GateError(
                    f"Все операторы Крауса канала {self.name!r} должны быть "
                    f"квадратными матрицами одного размера, получена форма {k.shape}"
                )
            m = k.conj().T @ k
            self._kraus_M.append(m)
            total += m
        if not np.allclose(total, np.eye(dim), atol=1e-8):
            raise GateError(
                f"Операторы Крауса канала {self.name!r} не образуют корректное "
                f"TPCP-отображение (sum K_i^dagger K_i != I)"
            )

    def sample_and_apply(self, state, q0: int, q1: int, rng) -> None:
        """Разыгрывает исход канала на паре кубитов (q0, q1) и применяет
        его к state (мутирует state.vector на месте, с перенормировкой).

        Как и в NoiseChannel.sample_and_apply, вероятности исходов
        считаются из приведённой (4x4, а не 2^n x 2^n) матрицы плотности
        пары кубитов — без полного O(k * 2^n) перебора копий состояния.
        """
        t = np.moveaxis(state.tensor, [q0, q1], [0, 1]).reshape(4, -1)
        rho = t @ t.conj().T  # приведённая 4x4 матрица плотности пары кубитов

        raw_probs = np.array(
            [max(float(np.trace(m @ rho).real), 0.0) for m in self._kraus_M]
        )
        total = raw_probs.sum()
        if total <= 1e-12:
            return
        choice = int(rng.choice(len(self.kraus_ops), p=raw_probs / total))

        # apply_gate — общая k-кубитная операция State; она не требует
        # унитарности оператора (проверка формы, не унитарности), что
        # нам и нужно здесь для отдельного (не обязательно унитарного)
        # оператора Крауса.
        state.apply_gate(self.kraus_ops[choice], [q0, q1])
        state.vector /= np.sqrt(raw_probs[choice])
        state.normalize()

    def __repr__(self):
        return f"TwoQubitNoiseChannel({self.name!r}, {len(self.kraus_ops)} Kraus ops)"


_PAULI_1Q = {'I': G.I, 'X': G.X, 'Y': G.Y, 'Z': G.Z}


def correlated_depolarizing_channel(p: float) -> TwoQubitNoiseChannel:
    """Коррелированный двухкубитный деполяризующий канал: с вероятностью
    p происходит случайная НЕтривиальная двухкубитная ошибка Паули
    (равновероятно любая из 15 комбинаций P⊗Q, кроме I⊗I), с
    вероятностью 1-p — ничего не происходит.

    Это прямое двухкубитное обобщение depolarizing_channel() (та же
    "twirl"-параметризация, sqrt(1-p)*I⊗I + sqrt(p/15)*{15 ненулевых
    P⊗Q}), пригождается для моделирования коррелированной ошибки
    двухкубитного гейта (например, CX) — в отличие от применения
    depolarizing_channel() независимо к каждому кубиту гейта, здесь
    ошибка на обоих кубитах разыгрывается СОВМЕСТНО одним броском, а не
    двумя независимыми, что и создаёт корреляцию между ошибками кубитов
    (при p, отличном от вырожденных 0 и 1)."""
    p = check_real(p, "p")
    if not (0.0 <= p <= 1.0):
        raise QSimValueError(f"Вероятность деполяризации должна быть в [0, 1], получено {p}")
    labels = [(a, b) for a in 'IXYZ' for b in 'IXYZ' if not (a == 'I' and b == 'I')]
    kraus = [np.sqrt(1 - p) * np.kron(G.I, G.I)]
    for a, b in labels:
        kraus.append(np.sqrt(p / 15) * np.kron(_PAULI_1Q[a], _PAULI_1Q[b]))
    return TwoQubitNoiseChannel(kraus, name=f"correlated_depolarizing(p={p:.4g})")


def amplitude_damping_channel(gamma: float) -> NoiseChannel:
    """Затухание амплитуды (релаксация |1> -> |0>) с параметром gamma."""
    gamma = check_real(gamma, "gamma")
    if not (0.0 <= gamma <= 1.0):
        raise QSimValueError(f"gamma должна быть в [0, 1], получено {gamma}")
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return NoiseChannel([k0, k1], name=f"amplitude_damping(gamma={gamma:.4g})")


class NoiseModel:
    """Привязка каналов шума к кубитам.

    По умолчанию (``add_all_qubit_channel``) канал применяется после
    КАЖДОГО гейта на КАЖДОМ затронутом кубите; ``add_qubit_channel``
    переопределяет канал для конкретного кубита. Явно заданное отсутствие
    шума на кубите можно получить, передав ``None`` в качестве канала.

    По умолчанию модель зашумляет отдельные кубиты после гейта (в том
    числе оба кубита двухкубитных/управляемых гейтов по отдельности) —
    это не может выразить шум, коррелированный между двумя кубитами.
    Для гейтов, которые действуют РОВНО на два кубита (обычные
    двухкубитные гейты, а также control()/controlled_gate() с одним
    control и одним target — например, CX), можно вместо этого
    зарегистрировать TwoQubitNoiseChannel для конкретной пары кубитов
    через ``add_two_qubit_channel(q0, q1, channel)``: если такой канал
    для пары найден, он ПОЛНОСТЬЮ заменяет собой независимый шум по
    отдельным кубитам для этой конкретной инструкции (не комбинируется
    с ним) — иначе шум был бы посчитан дважды. Многокубитные управляемые
    гейты (mcx и т.п., больше двух задействованных кубитов) всегда
    зашумляются по отдельным кубитам — двухкубитные каналы для них не
    подбираются.
    """

    def __init__(self):
        self._default_channel: NoiseChannel | None = None
        self._qubit_channels: dict[int, NoiseChannel | None] = {}
        self._two_qubit_channels: dict[frozenset, TwoQubitNoiseChannel] = {}

    def add_all_qubit_channel(self, channel: NoiseChannel) -> "NoiseModel":
        self._default_channel = channel
        return self

    def add_qubit_channel(self, qubit: int, channel: NoiseChannel | None) -> "NoiseModel":
        self._qubit_channels[qubit] = channel
        return self

    def add_two_qubit_channel(self, q0: int, q1: int, channel: TwoQubitNoiseChannel | None) -> "NoiseModel":
        """Регистрирует (или, если channel is None, снимает) коррелированный
        двухкубитный канал шума для НЕУПОРЯДОЧЕННОЙ пары кубитов
        (q0, q1) — действует одинаково для инструкций control(...,[q0],q1)
        и control(...,[q1],q0)."""
        key = frozenset((q0, q1))
        if channel is None:
            self._two_qubit_channels.pop(key, None)
        else:
            self._two_qubit_channels[key] = channel
        return self

    def channel_for(self, qubit: int):
        if qubit in self._qubit_channels:
            return self._qubit_channels[qubit]
        return self._default_channel

    def two_qubit_channel_for(self, q0: int, q1: int):
        return self._two_qubit_channels.get(frozenset((q0, q1)))

    def __repr__(self):
        return (
            f"NoiseModel(default={self._default_channel!r}, "
            f"per_qubit={self._qubit_channels!r}, "
            f"two_qubit={self._two_qubit_channels!r})"
        )
