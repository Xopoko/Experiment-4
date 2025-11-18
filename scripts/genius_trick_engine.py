#!/usr/bin/env python3
"""Utility that generates "GeniusTrickEngine" prompts for a task description.

Example:
    python3 scripts/genius_trick_engine.py --task "точное аналитическое решение 3D модели Изинга"
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict


class GeniusTrickEngine:
    """Emit seven conceptual reframings of a task, following the user sketch."""

    def __init__(self, task_description: str) -> None:
        task = task_description.strip()
        if not task:
            raise ValueError("Task description must be non-empty")
        self.task = task

    def _wrap(self, template: str) -> str:
        return template.format(task=self.task)

    def trick_1_recode(self) -> str:
        return self._wrap(
            "Перекодировка: представь '{task}' как граф, контурные интегралы,"
            " фермионы, матрицы или пути интегрирования. Что изменится, если"
            " записать задачу в этих координатах, и какие переменные станут"
            " естественными?"
        )

    def trick_2_expand(self) -> str:
        return self._wrap(
            "Временное/пространственное расширение: добавь комплексные"
            " параметры, лишнее измерение или призрачные спины к '{task}'."
            " Реши расширенную модель, затем аккуратно «отними» избыточные"
            " степени свободы."
        )

    def trick_3_dual(self) -> str:
        return self._wrap(
            "Дуальность: поменяй местами сильное и слабое, порядок и хаос,"
            " высокие и низкие температуры в '{task}'. Как выглядит двойственная"
            " формулировка и в чём она проще?"
        )

    def trick_4_symmetry_boost(self) -> str:
        return self._wrap(
            "Сверхсимметрия: предположи, что '{task}' инвариантна под"
            " конформной, суперсимметричной или модулярной трансформацией."
            " Какие закономерности и уравнения тогда следуют автоматически?"
        )

    def trick_5_isomorphism(self) -> str:
        return self._wrap(
            "Изоморфизм: какая модель (плиткование, случайные блуждания,"
            " димеры, перколяция и т. п.) — это '{task}' в другом костюме?"
            " Определи отображение и что оно сохраняет."
        )

    def trick_6_combinatorial_collapse(self) -> str:
        return self._wrap(
            "Комбинаторный коллапс: сумма по 2^N конфигураций в '{task}'"
            " равна количеству каких геометрических или топологических"
            " объектов? Найди способ пересчитать сумму через геометрию."
        )

    def trick_7_coarse_grain(self) -> str:
        return self._wrap(
            "Масштабный коллапс: реши '{task}' на иерархической/крупной решётке"
            " или в пределе большой размерности. Совпадает ли эффект с"
            " исходной задачей — и почему?"
        )

    def apply_all(self) -> Dict[str, str]:
        return {
            "1. Перекодировка": self.trick_1_recode(),
            "2. Расширение": self.trick_2_expand(),
            "3. Дуальность": self.trick_3_dual(),
            "4. Симметрия": self.trick_4_symmetry_boost(),
            "5. Изоморфизм": self.trick_5_isomorphism(),
            "6. Комбинаторный коллапс": self.trick_6_combinatorial_collapse(),
            "7. Масштабный коллапс": self.trick_7_coarse_grain(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate seven GeniusTrickEngine prompts for a task description."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", help="Task description string")
    group.add_argument(
        "--task-file",
        type=pathlib.Path,
        help="Path to a file whose content will be used as the task description",
    )
    parser.add_argument(
        "--json-output",
        type=pathlib.Path,
        help="Optional path to store the tricks as JSON",
    )
    parser.add_argument(
        "--text-output",
        type=pathlib.Path,
        help="Optional path to store the formatted text output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task is not None:
        task_description = args.task
    else:
        task_description = args.task_file.read_text(encoding="utf-8")

    engine = GeniusTrickEngine(task_description)
    tricks = engine.apply_all()

    formatted_lines = []
    for name, idea in tricks.items():
        formatted_lines.append(f"{name}\n{idea}\n")
    formatted_text = "\n".join(formatted_lines).strip()

    print(formatted_text)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"task": engine.task, "tricks": tricks}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    if args.text_output:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(formatted_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
