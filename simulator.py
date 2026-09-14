"""Измерения, сэмплирование распределения, инспекция состояния."""
from __future__ import annotations

import numpy as np

from ._common import check_qubit_index, check_qubit_indices, check_shots
from ._errors import QSimValueError
from . import gates as G


class Simulator:
    def __init__(self, state, seed: int | None = None):
        self.state = state
        self.rng = np.random.default_rng(seed)

    def _check_qubit(self, qubit: int) -> int:
        return check_qubit_index(qubit, self.state.n)

    def measure(self, qubit: int, collapse: bool = True) -> int:
        """Измеряет кубит. Возвращает 0/1."""
        qubit = self._check_qubit(qubit)
        t = self.state.tensor
        probs = np.abs(t) ** 2
        p0 = probs.take(0, axis=qubit).sum()
        outcome = 0 if self.rng.random() < p0 else 1

        if collapse:
            t = np.moveaxis(t, qubit, -1)
            mask = np.zeros(2, dtype=complex)
            mask[outcome] = 1.0
            t = t * mask
            t = np.moveaxis(t, -1, qubit)
            self.state.set_tensor(t)
            self.state.normalize()
        return outcome

    def measure_all(self, collapse: bool = True) -> str:
        return ''.join(str(self.measure(q, collapse))
                       for q in range(self.state.n))

    def sample(self, shots: int = 1024, qubits=None) -> dict:
        """Сэмплирует распределение исходов без пере-прогона схемы.
        Корректно для случая без mid-circuit измерений с обратной связью:
        достаточно один раз посчитать |amplitude|^2 и разыграть shots раз.

        qubits: какие кубиты попадают в итоговую битовую строку (в этом
        порядке чтения слева-направо == по возрастанию индекса). По
        умолчанию — все кубиты (как раньше). Если передан подсписок
        (например, только те кубиты, для которых в схеме реально стоял
        measure(...)), то в ключах словаря останутся только их биты,
        а не полная строка длины n — иначе, скажем, measure(0) на 2
        кубитах возвращал бы неотличимые от полного измерения строки
        '00'/'10' вместо ожидаемых '0'/'1'.

        Сама выборка исхода по-прежнему делается из полного совместного
        распределения (state.probabilities()), а не из отдельно
        посчитанного маргинала — для одного разыгранного индекса это
        эквивалентно (маргинализация после сэмплирования из совместного
        распределения даёт то же самое, что сэмплирование из маргинала),
        но не требует отдельного суммирования по неизмеряемым кубитам.
        """
        shots = check_shots(shots)

        probs = self.state.probabilities()
        nz = np.flatnonzero(probs > 0)
        if len(nz) == 0:
            raise QSimValueError("Невозможно сэмплировать нулевое распределение")
        weights = probs[nz]
        weights = weights / weights.sum()

        # Для малых shots это заметно дешевле на больших разреженных состояниях:
        # rng.choice работает только по ненулевой поддержке, а не по всему 2^n.
        outcomes = self.rng.choice(nz, size=shots, p=weights)
        n = self.state.n
        if qubits is None:
            qubits = list(range(n))
        else:
            qubits = [check_qubit_index(q, n, "Кубит в качестве отслеживаемого") for q in qubits]
            if len(set(qubits)) != len(qubits):
                # Раньше повтор кубита в qubits (например, sample(qubits=[0, 0]))
                # никак не проверялся: он молча дублировал соответствующий бит в
                # каждом ключе counts (например, '00' вместо '0'), без единого
                # предупреждения о том, что результат, скорее всего, не то, что
                # имел в виду вызывающий.
                raise QSimValueError(
                    f"Кубиты в качестве отслеживаемых (qubits) не должны "
                    f"повторяться, получено {qubits!r}"
                )
        counts = {}
        for o in outcomes:
            full_bits = format(o, f'0{n}b')
            bits = ''.join(full_bits[q] for q in qubits)
            counts[bits] = counts.get(bits, 0) + 1
        return counts

    # --- инспекция состояния ---
    def probabilities_dict(self, min_prob: float = 1e-10) -> dict:
        """{битовая строка: вероятность} без сэмплирования — точное распределение."""
        probs = self.state.probabilities()
        n = self.state.n
        out = {}
        nz = np.nonzero(probs > min_prob)[0]
        for i in nz:
            out[format(i, f'0{n}b')] = float(probs[i])
        return out

    def statevector(self, min_amp: float = 1e-10) -> dict:
        """{битовая строка: амплитуда} для ненулевых амплитуд."""
        vec = self.state.vector
        n = self.state.n
        out = {}
        nz = np.nonzero(np.abs(vec) > min_amp)[0]
        for i in nz:
            out[format(i, f'0{n}b')] = complex(vec[i])
        return out

    def expectation_z(self, qubit: int) -> float:
        """<Z> на одном кубите: p(0) - p(1)."""
        qubit = self._check_qubit(qubit)
        t = self.state.tensor
        probs = np.abs(t) ** 2
        p0 = probs.take(0, axis=qubit).sum()
        p1 = probs.take(1, axis=qubit).sum()
        return float(p0 - p1)

    def expectation(self, pauli_string: str, qubits=None) -> float:
        """<ψ|P|ψ> для строки Паули вида ``'XZI'`` (символы 'I','X','Y','Z',
        регистр не важен).

        qubits — на каких кубитах действует каждый символ строки, в том
        же порядке (по умолчанию — по возрастанию индекса, 0..len-1).
        Не путать с expectation_z(), которая исторически считает то же
        самое для одного кубита без построения полной матрицы Паули;
        этот метод — обобщение на произвольную тензорную строку Паули,
        реализованное без построения полной 2^n x 2^n матрицы: каждый
        однокубитный оператор строки применяется напрямую к тензору
        состояния (O(2^n) на символ), после чего берётся скалярное
        произведение с исходным вектором.
        """
        n = self.state.n
        if qubits is None:
            qubits = list(range(len(pauli_string)))
        if len(pauli_string) != len(qubits):
            raise QSimValueError(
                "Длина строки Паули должна совпадать с числом qubits: "
                f"{len(pauli_string)} символов, {len(qubits)} кубитов"
            )
        pauli_string = pauli_string.upper()
        valid = set('IXYZ')
        if any(ch not in valid for ch in pauli_string):
            raise QSimValueError(
                f"Строка Паули может состоять только из символов I/X/Y/Z, "
                f"получено {pauli_string!r}"
            )
        qubits = check_qubit_indices(qubits, n, "Кубит в строке Паули")
        if len(set(qubits)) != len(qubits):
            raise QSimValueError("Кубиты в строке Паули не должны повторяться")

        mats = {'I': G.I, 'X': G.X, 'Y': G.Y, 'Z': G.Z}
        psi = self.state.vector
        # .copy() здесь не нужен: reshape() возвращает view на psi, а
        # каждый шаг цикла ниже не пишет в этот view на месте — он
        # переприсваивает локальное имя tensor новому массиву,
        # созданному np.moveaxis/np.einsum. Раньше .copy() выполнялся
        # безусловно на каждый вызов (лишняя O(2^n) операция), даже когда
        # pauli_string состоит из одних 'I' и цикл вообще не заходит
        # внутрь (тогда tensor так и остаётся тем самым view).
        tensor = psi.reshape([2] * n)
        for ch, q in zip(pauli_string, qubits):
            if ch == 'I':
                continue
            mat = mats[ch]
            tensor = np.moveaxis(tensor, q, -1)
            tensor = np.einsum('ij,...j->...i', mat, tensor)
            tensor = np.moveaxis(tensor, -1, q)
        phi = tensor.reshape(-1)

        val = np.vdot(psi, phi)
        if abs(val.imag) > 1e-8:
            # Строка Паули эрмитова, поэтому <psi|P|psi> обязана быть
            # вещественной; ненулевая мнимая часть означает баг выше по
            # стеку (не может возникнуть на пользовательском вводе, раз
            # валидация символов прошла), а не законный результат.
            raise QSimValueError(
                f"<P> оказалось комплексным ({val!r}) — это указывает на "
                f"внутреннюю ошибку, а не на корректный физический результат"
            )
        return float(val.real)

    def bloch_vector(self, qubit: int) -> tuple:
        """Вектор Блоха (x, y, z) одного кубита.

        Вычисляется из приведённой (2x2) матрицы плотности этого кубита,
        полученной трассированием по всем остальным кубитам — без
        построения полной матрицы плотности 2^n x 2^n. Корректно
        определён и для сцепленных состояний: тогда |вектор| < 1 (кубит
        сам по себе — смешанное состояние), для несцепленного кубита в
        чистом состоянии |вектор| == 1.
        """
        qubit = self._check_qubit(qubit)
        n = self.state.n
        t = np.moveaxis(self.state.tensor, qubit, 0).reshape(2, -1)
        rho = t @ t.conj().T  # приведённая матрица плотности 2x2
        x = float(2 * rho[0, 1].real)
        y = float(-2 * rho[0, 1].imag)
        z = float((rho[0, 0] - rho[1, 1]).real)
        return (x, y, z)
