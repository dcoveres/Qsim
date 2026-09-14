# NOTICE

`qsim` — собственный код проекта. Лицензия самого пакета не определена
(см. раздел «Лицензия» в README) — добавьте её перед публикацией.

Пакет не включает (vendored) исходный код сторонних библиотек — они
подключаются как обычные Python-зависимости через `pip install` и
распространяются отдельно от `qsim`, под собственными лицензиями. Этот
файл перечисляет их в информационных целях.

## Обязательные зависимости

### NumPy

- **Используется в**: `state.py`, `circuit.py`, `simulator.py`, `gates.py`,
  `noise.py`, `_common.py` — вся линейная алгебра и векторизованные
  операции над вектором состояния.
- **Домашняя страница**: https://numpy.org
- **Исходный код**: https://github.com/numpy/numpy
- **Лицензия**: BSD 3-Clause License
- **Правообладатель**: Copyright (c) 2005–2025, NumPy Developers. All
  rights reserved.
- Полный текст лицензии: https://numpy.org/doc/stable/license.html

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of the NumPy Developers nor the names of any
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

NumPy, в свою очередь, дополнительно поставляет (bundled) в своём
дистрибутиве ряд сторонних компонентов (LAPACK/OpenBLAS, pocketfft и
др.) под совместимыми BSD-подобными лицензиями — они не используются
`qsim` напрямую (пакет не задействует `numpy.linalg`/`numpy.fft`), но
устанавливаются вместе с NumPy. Подробности — в собственном NOTICE
NumPy: https://github.com/numpy/numpy/blob/main/LICENSE.txt

## Опциональные зависимости

### Matplotlib

- **Используется в**: `visualization.py` (`plot_histogram`,
  `plot_bloch`) — **не** импортируется автоматически из `qsim/__init__.py`
  и не требуется для основной функциональности пакета; ставится только
  если нужна отрисовка гистограмм/сферы Блоха.
- **Домашняя страница**: https://matplotlib.org
- **Исходный код**: https://github.com/matplotlib/matplotlib
- **Лицензия**: Matplotlib License (BSD-совместимая, основана на
  лицензии PSF)
- **Правообладатель**: Copyright (c) 2012– Matplotlib Development Team;
  Copyright (c) 2002–2012 John Hunter, Darren Dale, Eric Firing, Michael
  Droettboom and the matplotlib development team. All Rights Reserved.
- Полный текст лицензии: https://matplotlib.org/stable/project/license.html

## Тестовое окружение (dev-зависимости)

- **pytest** *(опционально — тесты запускаются и через встроенный
  `unittest`, см. README)* — Apache License 2.0.
- **unittest** — часть стандартной библиотеки Python, отдельного
  указания лицензии не требует (регулируется лицензией CPython, PSF
  License).

---

Список актуален для зависимостей, фактически импортируемых кодом
пакета на момент составления этого файла. При добавлении новых
зависимостей (например, в `pyproject.toml`/`requirements.txt`)
дополните этот файл соответствующей записью.
