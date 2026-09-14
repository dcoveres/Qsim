"""Символьные параметры для параметризованных схем (VQE/QAOA и т.п.).

Поддерживаются однопараметрические гейты ``Circuit.rx/ry/rz/p``: вместо
числа им можно передать объект Parameter, а конкретное значение
подставить позже через ``Circuit.bind({param: value})``. Это позволяет
один раз построить структуру схемы и многократно запускать её с разными
значениями параметров (например, в цикле оптимизатора), не пересобирая
список инструкций заново на каждой итерации.

Параметрический ``Circuit.u()`` (theta, phi, lambda) НЕ поддерживается —
это осознанное ограничение объёма реализации, а не забытый случай.
"""
from __future__ import annotations

import itertools


class Parameter:
    """Именованный символьный параметр одного вещественного гейта.

    Два объекта Parameter с одинаковым именем — РАЗНЫЕ параметры
    (сравнение и хеш — по identity, а не по имени): это соответствует
    поведению Parameter в других библиотеках (Qiskit и т.п.) и позволяет
    иметь несколько разных параметров с одинаковым «человеческим» именем
    без конфликтов. Если нужен один и тот же параметр в нескольких
    местах схемы — переиспользуйте один и тот же объект Parameter.
    """
    _counter = itertools.count()

    def __init__(self, name: str | None = None):
        self.name = name if name is not None else f"θ{next(Parameter._counter)}"

    def __repr__(self):
        return f"Parameter({self.name!r})"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class _ParamGate:
    """Внутренний маркер: однокубитный гейт с несвязанным параметром.

    Хранится в instructions вместо готовой матрицы до вызова
    Circuit.bind(). Класс не экспортируется публично — пользователю
    достаточно Parameter и Circuit.bind()/Circuit.parameters.
    """
    __slots__ = ('kind', 'param')

    _MATRIX_FACTORIES = {}  # заполняется в circuit.py, чтобы избежать
                             # циклического импорта gates<->parameters

    def __init__(self, kind: str, param: Parameter):
        self.kind = kind
        self.param = param

    def build_matrix(self, value: float):
        return self._MATRIX_FACTORIES[self.kind](value)

    def __repr__(self):
        return f"_ParamGate({self.kind}, {self.param!r})"
