#!/usr/bin/env bash
# Запуск редактора. Создаёт виртуальное окружение .venv и ставит/доставляет зависимости.
set -e
cd "$(dirname "$0")"
VENV=.venv
PY="$VENV/bin/python"

# Недоделанное окружение (например, venv создался без pip) — пересоздаём.
if [ -d "$VENV" ] && ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "Окружение $VENV повреждено — пересоздаю."
    rm -rf "$VENV"
fi

if [ ! -x "$PY" ]; then
    if ! python3 -m venv "$VENV"; then
        rm -rf "$VENV"
        echo
        echo "Не удалось создать виртуальное окружение."
        echo "Установите модуль venv:  sudo apt install python3-venv"
        exit 1
    fi
fi

# Проверяем, что зависимости реально импортируются, а не просто что папка существует.
if ! "$PY" -c "import numpy, PySide6.QtWidgets" >/dev/null 2>&1; then
    echo "Устанавливаю зависимости…"
    "$PY" -m pip install --upgrade pip
    if ! "$PY" -m pip install -r requirements.txt; then
        echo
        echo "Не удалось установить зависимости (см. ошибку pip выше)."
        echo "Проверьте подключение к интернету и запустите ./run.sh ещё раз."
        exit 1
    fi
fi

command -v ffmpeg >/dev/null 2>&1 || echo "Внимание: ffmpeg не найден — установите: sudo apt install ffmpeg"

exec "$PY" main.py "$@"
