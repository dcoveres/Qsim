"""Вектор состояния и низкоуровневые операции над тензором.

Всё сделано на векторных numpy-операциях (без питоновских циклов по
2^n амплитудам), поэтому работает вплоть до ~20-24 кубит на обычной
машине.
"""
from __future__ import annotations

import numpy as np

from ._common import MAX_RECOMMENDED_QUBITS, check_qubit_count, check_qubit_index
from ._errors import GateError, QubitError


class State:
    def __init__(self, n_qubits: int):
        self.n = check_qubit_count(n_qubits, MAX_RECOMMENDED_QUBITS)
        self.dim = 2 ** self.n
        self.vector = np.zeros(self.dim, dtype=complex)
        self.vector[0] = 1.0  # |00...0>

    # --- представление ---
    @property
    def tensor(self):
        return self.vector.reshape([2] * self.n)

    def set_tensor(self, tensor):
        self.vector = np.ascontiguousarray(tensor).reshape(-1)

    def _check_qubit(self, q: int) -> int:
        return check_qubit_index(q, self.n)

    @staticmethod
    def _check_matrix_dim(matrix, k, what="targets"):
        """Проверяет, что matrix — это 2^k x 2^k, до того как её пустят
        в тяжёлые numpy-операции. Без этой проверки несоответствие
        размера всплывало бы глубоко внутри reshape/matmul с непонятной
        ошибкой, а не сразу и по делу."""
        matrix = np.asarray(matrix)
        expected = 2 ** k
        if matrix.ndim != 2 or matrix.shape != (expected, expected):
            raise GateError(
                f"Матрица гейта имеет форму {matrix.shape}, а для "
                f"{k} {what} ожидается {expected}x{expected}"
            )
        return matrix

    # --- однокубитные / многокубитные гейты без управления ---
    def apply_single(self, gate, qubit: int) -> None:
        """2x2 гейт на кубит qubit."""
        qubit = self._check_qubit(qubit)
        gate = self._check_matrix_dim(gate, 1, "target-кубит")
        t = self.tensor
        t = np.moveaxis(t, qubit, -1)
        t = np.einsum('ij,...j->...i', gate, t)
        t = np.moveaxis(t, -1, qubit)
        self.set_tensor(t)

    def apply_gate(self, matrix, qubits) -> None:
        """Общий k-кубитный гейт (матрица 2^k x 2^k) на список qubits.

        qubits[0] соответствует старшему (левому) биту в матрице —
        единой конвенции индексации кубитов, используемой во всём
        пакете (кубит 0 — старший бит базисного индекса). Обратите
        внимание: это НЕ совпадает с little-endian конвенцией Qiskit."""
        qubits = list(qubits)
        k = len(qubits)
        if len(set(qubits)) != k:
            raise QubitError("Индексы кубитов должны быть различны")
        qubits = [self._check_qubit(q) for q in qubits]
        matrix = self._check_matrix_dim(matrix, k, "target-кубит(а/ов)")
        t = self.tensor
        other = [i for i in range(self.n) if i not in qubits]
        axes = other + qubits
        t = np.transpose(t, axes)
        shape = t.shape
        t = t.reshape(-1, 2 ** k)
        t = t @ matrix.T
        t = t.reshape(shape)
        inv = np.argsort(axes)
        t = np.transpose(t, inv)
        self.set_tensor(t)

    def apply_two_qubit(self, gate4x4, q0: int, q1: int) -> None:
        """4x4 гейт на два кубита (q0 — старший в матрице)."""
        if q0 == q1:
            raise QubitError("q0 и q1 должны различаться")
        self.apply_gate(gate4x4, [q0, q1])

    # --- управляемые гейты (векторизовано, без питоновских циклов) ---
    def apply_controlled(self, matrix, controls, targets) -> None:
        """Гейт matrix (2^k x 2^k) на targets, активируется когда все
        control-кубиты равны |1>. controls и targets не пересекаются.

        Реализовано через базовую (не fancy) индексацию numpy — она
        возвращает view на self.vector, поэтому запись в срез обычно
        сразу меняет состояние без явного цикла по 2^n амплитудам. Но
        полагаться на то, что sub/t навсегда останутся view друг в
        друга и в self.vector — хрупко (например, если self.vector
        когда-нибудь перестанет быть contiguous). Поэтому в конце мы
        всё равно явно фиксируем результат через set_tensor(t), вместо
        того чтобы полагаться исключительно на побочный эффект записи
        в sub.
        """
        controls = list(controls)
        targets = list(targets)
        all_qubits = controls + targets
        if len(set(controls)) != len(controls):
            raise QubitError("controls не должны содержать повторов")
        if len(set(targets)) != len(targets):
            raise QubitError("targets не должны содержать повторов")
        if set(controls) & set(targets):
            raise QubitError("controls и targets должны быть непересекающимися")
        all_qubits = [self._check_qubit(q) for q in all_qubits]
        controls, targets = all_qubits[:len(controls)], all_qubits[len(controls):]

        k = len(targets)
        matrix = self._check_matrix_dim(matrix, k, "target-кубит(а/ов)")

        if not controls:
            self.apply_gate(matrix, targets)
            return

        t = self.tensor
        idx = [slice(None)] * self.n
        for c in controls:
            idx[c] = 1
        sub = t[tuple(idx)]  # view: базовая индексация -> подтензор без controls-осей

        control_set = sorted(controls)

        def shift(ax):
            return ax - sum(1 for c in control_set if c < ax)

        shifted_targets = [shift(ax) for ax in targets]
        m = sub.ndim
        other_axes = [a for a in range(m) if a not in shifted_targets]
        axes_order = other_axes + shifted_targets

        sub_t = np.transpose(sub, axes_order)
        shape = sub_t.shape
        sub_t = np.ascontiguousarray(sub_t).reshape(-1, 2 ** k)
        sub_t = sub_t @ matrix.T
        sub_t = sub_t.reshape(shape)
        inv = np.argsort(axes_order)
        sub_t = np.transpose(sub_t, inv)

        sub[...] = sub_t  # обычно уже обновляет self.vector напрямую

        # Явная фиксация — гарантирует корректность независимо от того,
        # остаются sub/t view'ами в self.vector или нет.
        self.set_tensor(t)

    def apply_controlled_single(self, gate2x2, controls, target: int) -> None:
        self.apply_controlled(gate2x2, controls, [target])

    # --- утилиты ---
    def normalize(self) -> None:
        norm = np.linalg.norm(self.vector)
        if norm > 1e-12:
            self.vector /= norm

    def probabilities(self) -> np.ndarray:
        return np.abs(self.vector) ** 2

    def amplitude(self, basis_state) -> complex:
        """Амплитуда одного базисного состояния.

        basis_state — либо битовая строка вида '010' длины ровно n
        (кубит 0 — старший/левый бит), либо целый индекс в [0, dim).

        Раньше принималась строка любой длины: 'amplitude("1")' на
        3-кубитном состоянии молча трактовалась как int('1', 2) == 1 —
        то есть тихо возвращала амплитуду совсем не того базисного
        состояния, которое, вероятно, имел в виду вызывающий (без
        всякой ошибки). Слишком длинная строка, наоборот, падала с
        низкоуровневым numpy IndexError без указания на то, что не так.
        Теперь длина и алфавит строки проверяются явно.
        """
        if isinstance(basis_state, str):
            if len(basis_state) != self.n or any(ch not in '01' for ch in basis_state):
                raise QubitError(
                    f"Битовая строка {basis_state!r} должна состоять "
                    f"ровно из {self.n} символов '0'/'1' "
                    f"(кубит 0 — старший/левый бит)"
                )
            idx = int(basis_state, 2)
        else:
            idx = check_qubit_index(basis_state, self.dim, name="Индекс базисного состояния")
        return self.vector[idx]

    def density_matrix(self) -> np.ndarray:
        """Матрица плотности чистого состояния: rho = |ψ><ψ|.

        State остаётся statevector-представлением (O(2^n) памяти); эта
        матрица (O(4^n) памяти) — вспомогательная величина для анализа/
        обучения и малых n (сцепленность, приведённые состояния через
        partial trace и т.п.), а не альтернативный бэкенд симуляции.
        Для приведённой матрицы плотности одного кубита используйте
        Simulator.bloch_vector(), которая не строит полную rho.
        """
        v = self.vector.reshape(-1, 1)
        return v @ v.conj().T

    def reset(self) -> None:
        self.vector = np.zeros(self.dim, dtype=complex)
        self.vector[0] = 1.0

    def copy(self) -> "State":
        s = State(self.n)
        s.vector = self.vector.copy()
        return s

    def num_bytes(self) -> int:
        return self.vector.nbytes

    def __repr__(self):
        return f"State(n={self.n}, dim={self.dim}, mem={self.num_bytes()/1e6:.1f}MB)"
