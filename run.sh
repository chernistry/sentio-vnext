#!/bin/bash
# Скрипт-обертка для запуска CLI команд Sentio без установки проекта

# Добавляем текущую директорию в PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Запускаем CLI с переданными аргументами
python -m src.cli.main "$@"
