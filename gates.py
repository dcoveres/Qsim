"""Одно- и много-кубитные гейты.

Соглашение об индексации кубитов используется единое во всём пакете:
кубит с индексом 0 — самый старший (левый) бит в двоичной записи индекса
базисного состояния. Это НЕ совпадает с little-endian конвенцией Qiskit
(там кубит 0 — младший бит) — раньше в state.py была вводящая в
заблуждение отсылка к "Qiskit-подобной конвенции", которая на деле не
соответствовала используемой здесь схеме; при интеграции с
Qiskit-экспортированными данными об этом нужно помнить (потребуется
перестановка/reverse кубитов).
"""
from __future__ import annotations

import numpy as np

from ._common import check_unitary
from ._errors import GateError

SQRT2 = np.sqrt(2)

# --- однокубитные, фиксированные ---
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / SQRT2
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = S.conj().T
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = T.conj().T
SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2
SXDG = SX.conj().T


# --- фабрики параметрических гейтов ---
def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]], dtype=complex)


def Phase(phi: float) -> np.ndarray:
    """Он же P / U1 — фаза на |1>."""
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)


def U(theta: float, phi: float, lam: float) -> np.ndarray:
    """Общий однокубитный гейт U3(theta, phi, lambda)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
    ], dtype=complex)


# --- реестр гейтов без параметров, по имени ---
NAMED = {
    'I': I, 'X': X, 'Y': Y, 'Z': Z,
    'H': H, 'S': S, 'SDG': SDG, 'T': T, 'TDG': TDG,
    'SX': SX, 'SXDG': SXDG,
}


def get(name: str) -> np.ndarray:
    """Достаёт гейт по имени (регистр не важен).

    Раньше здесь бросался голый KeyError, что выпадало из единой
    иерархии QSimError, введённой в _errors.py: например,
    Circuit.control('NONEXISTENT', ...) отдавал пользователю KeyError,
    хотя рядом, для невалидной пользовательской матрицы, тот же метод
    аккуратно бросает GateError/QubitError. GateError — это QSimValueError
    (некорректное *значение* имени гейта), поэтому такой же except
    ValueError/except QSimError, что ловит остальные ошибки пакета,
    ловит и эту.
    """
    key = name.upper()
    if key not in NAMED:
        raise GateError(
            f"Неизвестный гейт: {name!r}. Доступные имена: "
            f"{', '.join(sorted(NAMED))}"
        )
    return NAMED[key]


# --- 2-кубитные матрицы 4x4 (для apply_gate/apply_two_qubit) ---
CNOT_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)

CY_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, -1j],
    [0, 0, 1j, 0],
], dtype=complex)

CZ_MATRIX = np.diag([1, 1, 1, -1]).astype(complex)

SWAP_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)


def CPhase(phi: float) -> np.ndarray:
    return np.diag([1, 1, 1, np.exp(1j * phi)]).astype(complex)


def controlled_matrix(gate, num_controls: int = 1) -> np.ndarray:
    """Явно раздувает k-кубитный гейт до полной матрицы
    (2^num_controls * 2^k)-размера с контролем по |1...1>. Пригождается
    редко (обычно выгоднее State.apply_controlled), но полезно для
    инспекции/тестов.

    Контроли считаются старшими битами базисного индекса полной матрицы,
    target-биты — младшими (согласовано с остальной конвенцией пакета).
    """
    gate = np.asarray(gate, dtype=complex)
    if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
        raise ValueError(f"gate должен быть квадратной матрицей, получено {gate.shape}")
    dim = gate.shape[0]
    k = int(round(np.log2(dim)))
    if 2 ** k != dim:
        raise ValueError(f"Размер gate ({dim}) должен быть степенью двойки")
    if num_controls < 0:
        raise ValueError("num_controls должно быть >= 0")

    full_dim = dim * (2 ** num_controls)
    full = np.eye(full_dim, dtype=complex)
    off = full_dim - dim
    full[off:, off:] = gate
    return full


# --- самопроверка встроенных гейтов при импорте модуля ---
# Раньше ошибка в матрице фиксированного гейта (опечатка при
# копировании, неверный знак фазы и т.п.) осталась бы незамеченной до тех
# пор, пока кто-нибудь не заметил бы физически неверный результат
# симуляции. Эта проверка — дешёвая (константное число маленьких матриц,
# выполняется один раз при импорте) и ловит такие ошибки немедленно,
# с понятным сообщением, а не как "результат почему-то не тот".
for _name, _mat in NAMED.items():
    check_unitary(_mat, what=f"Встроенный гейт {_name!r}")
for _name, _mat in (('CNOT', CNOT_MATRIX), ('CY', CY_MATRIX),
                     ('CZ', CZ_MATRIX), ('SWAP', SWAP_MATRIX)):
    check_unitary(_mat, what=f"Встроенный гейт {_name!r}")
del _name, _mat
