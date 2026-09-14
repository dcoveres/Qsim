"""Общие константы и функции валидации.

До этого файла ``MAX_RECOMMENDED_QUBITS`` был продублирован в state.py и
circuit.py независимо, а проверки индексов кубитов в Circuit и State были
неполными и расходились друг с другом (например, Circuit пропускал
нецелые/булевы индексы кубитов, которые потом падали с невнятной ошибкой
глубоко внутри numpy). Здесь всё собрано в одном месте, чтобы оба класса
гарантированно применяли одинаковые, полные проверки.
"""
from __future__ import annotations

import numbers
import warnings

import numpy as np

from ._errors import QSimTypeError, QubitError, GateError, ShotsError

# Единый практический лимит числа кубитов для statevector-симуляции.
# Вектор состояния занимает 16 байт (complex128) * 2^n_qubits.
MAX_RECOMMENDED_QUBITS = 25  # ~512MB под вектор состояния

# С какого числа кубитов предупреждать пользователя о потенциально
# большом расходе памяти/времени (но ещё не запрещать).
_WARN_QUBITS_THRESHOLD = 20


class LargeStateWarning(UserWarning):
    """Предупреждение о потенциально большом расходе памяти/времени при
    большом n_qubits.

    Раньше здесь использовался ResourceWarning — но это семантически
    неверная категория (она предназначена для незакрытых ресурсов вроде
    файлов/сокетов), и, что важнее, в стандартном списке фильтров CPython
    есть ``ignore::ResourceWarning`` — предупреждение по умолчанию нигде
    не печаталось, и пользователь никак не мог о нём узнать без ручной
    настройки warnings.simplefilter(). UserWarning (и его подклассы)
    входят в default-политику CPython, поэтому предупреждение теперь
    реально видно "из коробки"."""


def check_qubit_count(n_qubits, max_qubits: int = MAX_RECOMMENDED_QUBITS) -> int:
    """Проверяет и нормализует число кубитов схемы/состояния.

    Раньше это проверялось по-разному в Circuit (пропускал float/bool) и
    State (проверял тип) — из-за этого ``Circuit(2.5)`` успешно строился и
    падал только при вызове .run(), с сообщением об ошибке, никак не
    указывающим на настоящую причину (создание Circuit, а не run()).
    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, numbers.Integral):
        raise QSimTypeError(
            f"n_qubits должно быть целым числом, получено "
            f"{type(n_qubits).__name__}: {n_qubits!r}"
        )
    n_qubits = int(n_qubits)
    if n_qubits < 1:
        raise QubitError(f"n_qubits должно быть >= 1, получено {n_qubits}")
    if n_qubits > max_qubits:
        raise QubitError(
            f"n_qubits={n_qubits} слишком много для statevector-симуляции: "
            f"понадобится примерно {16 * 2 ** n_qubits / 1e9:.1f}GB памяти. "
            f"Рекомендуемый максимум — {max_qubits}."
        )
    if n_qubits >= _WARN_QUBITS_THRESHOLD:
        warnings.warn(
            f"n_qubits={n_qubits}: вектор состояния займёт "
            f"~{16 * 2 ** n_qubits / 1e6:.0f}MB, операции могут быть медленными.",
            LargeStateWarning,
            stacklevel=3,
        )
    return n_qubits


def check_qubit_index(q, n: int, name: str = "Кубит") -> int:
    """Проверяет один индекс кубита. Явно отвергает bool и нецелые числа —

    ``isinstance(True, int)`` в Python равен True, а ``0 <= 1.5 < n`` может
    оказаться истиной, поэтому без явной проверки типа такие значения
    раньше тихо проходили валидацию и падали намного позже, в недрах numpy,
    с ошибкой не по адресу (например, ``TypeError: 'float' object is not
    iterable`` из np.moveaxis вместо понятного сообщения о неверном индексе
    кубита).
    """
    if isinstance(q, bool) or not isinstance(q, numbers.Integral):
        raise QSimTypeError(
            f"{name} должен быть целым числом, получено {type(q).__name__}: {q!r}"
        )
    q = int(q)
    if not (0 <= q < n):
        raise QubitError(f"{name} {q} вне диапазона [0, {n})")
    return q


def check_qubit_indices(qs, n: int, name: str = "Кубит") -> list[int]:
    return [check_qubit_index(q, n, name) for q in qs]


def check_shots(shots) -> int:
    """Проверяет число измерений (shots).

    Раньше эта проверка жила только внутри Simulator.sample(), которая
    вызывается лишь на "быстром" пути run() (без reset и без
    mid-circuit измерений). На "медленном" пути (с reset или
    mid-circuit измерением) run() просто делал ``for _ in range(shots)``:
    для отрицательных shots range() пустой, поэтому вызов молча
    возвращал ``({}, None)`` вместо ошибки — несогласованное поведение
    и нарушение контракта (State не может быть None).
    """
    if isinstance(shots, bool) or not isinstance(shots, numbers.Integral):
        raise QSimTypeError(
            f"shots должно быть целым числом, получено {type(shots).__name__}: {shots!r}"
        )
    shots = int(shots)
    if shots < 0:
        raise ShotsError(f"shots должно быть >= 0, получено {shots}")
    return shots


def check_real(value, name: str = "Параметр") -> float:
    """Проверяет, что параметр гейта (угол и т.п.) — вещественное число."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise QSimTypeError(
            f"{name} должен быть вещественным числом, получено "
            f"{type(value).__name__}: {value!r}"
        )
    return float(value)


def check_unitary(matrix, what: str = "Матрица гейта", atol: float = 1e-8) -> np.ndarray:
    """Проверяет унитарность матрицы (U†U == I).

    Раньше State.apply_single/apply_gate/apply_controlled принимали любую
    матрицу без проверки: применение неунитарной матрицы молча портило
    нормировку состояния (например, ``apply_single(2*I, 0)`` увеличивало
    норму вектора состояния вдвое без единого предупреждения). Для
    матриц, приходящих из встроенного реестра гейтов, унитарность
    проверяется один раз при импорте gates.py; для пользовательских
    матриц (Circuit.control/controlled_gate с произвольной матрицей) —
    здесь, в момент построения схемы, а не когда-то потом при run().
    """
    matrix = np.asarray(matrix, dtype=complex)
    dim = matrix.shape[0]
    product = matrix.conj().T @ matrix
    if not np.allclose(product, np.eye(dim), atol=atol):
        raise GateError(
            f"{what} не унитарна (U†U != I) — применение такой матрицы "
            f"нарушит нормировку состояния. Если неунитарность нужна "
            f"намеренно (моделирование шума и т.п.), используйте "
            f"State.apply_* напрямую в обход Circuit."
        )
    return matrix
