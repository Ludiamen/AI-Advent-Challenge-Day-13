"""StateValidator — проверка ответа кодом, а не уговорами.

На лекции это сказано прямо: текстового инварианта в промпте недостаточно.
Модель — вежливый собеседник, и если пользователь настойчиво просит «ну сделай
на Laravel», она согласится, несмотря на системный промпт. Код не согласится.

Поэтому проверка раздвоена:

  ЖЁСТКИЕ инварианты (стек, БД, секреты в URL) проверяет этот модуль поиском по
  тексту ответа. Результат детерминирован: одна и та же строка всегда даёт один
  и тот же вердикт, и переубедить его нельзя.

  МЯГКИЕ требования (стиль, тон, «показывай соответствие legacy-компоненту»)
  проверяет дешёвая модель. Они субъективны, регуляркой их не выразить, и
  ошибка проверки здесь ничего не ломает — это замечание, а не отказ.

Отдельная забота — ложные срабатывания. Ответ «Laravel здесь не подойдёт,
потому что…» запрещённое слово содержит, но инвариант не нарушает. Поэтому
найденное слово рассматривается в контексте: если рядом стоит отрицание или
речь о legacy-системе, нарушением это не считается. Внутри блоков кода
поблажки нет — там слово означает именно предложение его использовать.

Публичное API:
  Violation                        — одно нарушение
  StateValidator(profile, client)  — .check(answer) / .check_soft(answer) / .check_transition(...)
  REMINDER                         — добавка к промпту на повторную попытку
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agent import prompts
from agent.llm import Client, LLMError
from agent.memory.long import ProfileStore
from agent.memory.working import TaskState, TransitionError

log = logging.getLogger("agent.validator")

# Слова, рядом с которыми упоминание запрещённой технологии означает отказ от
# неё, а не предложение. Список получен разбором реальных ответов моделей.
NEGATIONS = (
    "не ", "нет", "нельзя", "вместо", "отказ", "запрещ", "устарев", "legacy",
    "старой", "старом", "старый", "прежн", "исключ", "мигрир", "переход",
    "переносим", "уходим", "было", "раньше", "текущей системе", "не подходит",
    "не рекоменд", "не буду", "не стану", "не могу",
)

# Ширина окна вокруг найденного слова, в котором ищется отрицание. Замерено на
# ответах моделей: пояснение «почему не» почти всегда стоит в том же предложении.
WINDOW = 90

_БЛОК_КОДА = re.compile(r"```.*?```", re.DOTALL)

REMINDER = """\
ВНИМАНИЕ. Предыдущий ответ нарушил инвариант проекта и был отклонён
автоматической проверкой: {нарушения}
Инварианты — жёсткое ограничение, а не пожелание. Переделай ответ в их рамках.
Если пользователь просит именно нарушить ограничение, объясни, что мешает, и
предложи допустимое решение."""


@dataclass
class Violation:
    """Одно нарушение: какой инвариант, что нашли и где именно."""

    code: str
    rule: str
    found: str
    where: str          # «код» или «текст»
    excerpt: str = ""

    def __str__(self) -> str:
        return f"{self.rule} (найдено «{self.found}» в блоке {self.where})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "код": self.code,
            "правило": self.rule,
            "найдено": self.found,
            "где": self.where,
            "фрагмент": self.excerpt,
        }


@dataclass
class SoftResult:
    """Вердикт мягкой проверки: замечания без права вето."""

    ok: bool = True
    notes: list[str] = field(default_factory=list)
    model_key: str = ""
    failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "годится": self.ok,
            "замечания": self.notes,
            "модель": self.model_key,
            "сбой": self.failed,
        }


class StateValidator:
    """Проверяет ответы модели и переходы задачи."""

    def __init__(self, profile: ProfileStore, client: Client | None = None,
                 soft_model: str = "") -> None:
        self.profile = profile
        self.client = client
        self.soft_model = soft_model

    # --- жёсткая проверка: только код ----------------------------------------

    def check(self, answer: str) -> list[Violation]:
        """Ищет нарушения жёстких инвариантов. Модель здесь не участвует."""
        нарушения: list[Violation] = []
        текст = answer or ""
        куски_кода = _БЛОК_КОДА.findall(текст)
        проза = _БЛОК_КОДА.sub(" ", текст)

        for инвариант in self.profile.invariants(hard_only=True):
            тип = инвариант.get("тип")
            правило = инвариант.get("правило", инвариант.get("код", ""))
            код = инвариант.get("код", "")
            for значение in инвариант.get("значения", []):
                if тип == "запрет-слов":
                    # В коде запрещённое слово означает именно использование —
                    # никаких поблажек на контекст.
                    for кусок in куски_кода:
                        совпадение = _найти(кусок, значение)
                        if совпадение:
                            нарушения.append(Violation(код, правило, значение, "код",
                                                       _вырезка(кусок, совпадение)))
                            break
                    else:
                        совпадение = _найти(проза, значение)
                        if совпадение and not _отрицание(проза, совпадение):
                            нарушения.append(Violation(код, правило, значение, "текст",
                                                       _вырезка(проза, совпадение)))
                elif тип == "запрет-в-коде":
                    # Упоминание legacy-компонента обязательно по профилю, а вот
                    # код на нём — уже нарушение. Поэтому смотрим только в блоки
                    # кода и не трогаем прозу вовсе.
                    for кусок in куски_кода:
                        совпадение = _найти(кусок, значение)
                        if совпадение:
                            нарушения.append(Violation(код, правило, значение, "код",
                                                       _вырезка(кусок, совпадение)))
                            break
                elif тип == "запрет-регулярок":
                    выражение = re.compile(значение, re.IGNORECASE)
                    совпадение = выражение.search(текст)
                    if совпадение:
                        нарушения.append(Violation(код, правило, совпадение.group(0),
                                                   "ответ", _вырезка(текст, совпадение.start())))
        return нарушения

    def reminder(self, violations: list[Violation]) -> str:
        """Текст добавки к промпту для повторной попытки."""
        перечень = "; ".join(str(н) for н in violations)
        return REMINDER.format(нарушения=перечень)

    # --- мягкая проверка: дешёвая модель -------------------------------------

    def check_soft(self, answer: str) -> SoftResult:
        """Проверяет стиль ответа. Замечания не блокируют — только сообщают."""
        if self.client is None:
            return SoftResult(failed=True, notes=["модель для мягкой проверки не задана"])
        стиль = self.profile.load().get("стиль") or {}
        if not стиль:
            return SoftResult(notes=["требований к стилю в профиле нет"])

        from agent import catalog
        ключ = self.soft_model or catalog.for_role("мягкая-проверка")
        требования = "\n".join(f"- {к}: {з}" for к, з in стиль.items())
        сообщения = [
            {"role": "system", "content": prompts.SOFT_CHECK},
            {"role": "user", "content": f"Требования:\n{требования}\n\nОтвет агента:\n{answer[:3000]}"},
        ]
        try:
            ответ = self.client.call(ключ, сообщения, max_tokens=300,
                                     temperature=0.0, low_effort=True)
        except LLMError as exc:
            log.warning("Мягкая проверка недоступна: %s", exc)
            return SoftResult(model_key=ключ, failed=True, notes=[str(exc)[:120]])

        текст = ответ.text.strip()
        начало, конец = текст.find("{"), текст.rfind("}")
        if начало == -1 or конец <= начало:
            return SoftResult(model_key=ключ, failed=True,
                              notes=["проверяющая модель вернула неразбираемый ответ"])
        try:
            данные = json.loads(текст[начало:конец + 1])
        except json.JSONDecodeError:
            return SoftResult(model_key=ключ, failed=True,
                              notes=["проверяющая модель вернула неразбираемый ответ"])
        return SoftResult(
            ok=bool(данные.get("годится", True)),
            notes=[str(з) for з in (данные.get("замечания") or [])],
            model_key=ключ,
        )

    # --- переходы задачи -----------------------------------------------------

    @staticmethod
    def check_transition(state: TaskState, stage: str) -> tuple[bool, str]:
        """Допустим ли переход. Сам переход не делает — только выносит вердикт."""
        try:
            копия = TaskState.from_dict(state.to_dict())
            копия.transition(stage)
        except TransitionError as exc:
            return False, str(exc)
        return True, f"переход {state.stage} -> {stage} разрешён"


def _найти(text: str, слово: str) -> int | None:
    """Ищет слово целиком, без учёта регистра; возвращает позицию или None."""
    выражение = re.compile(rf"(?<![\w.]){re.escape(слово)}(?![\w])", re.IGNORECASE)
    совпадение = выражение.search(text)
    return совпадение.start() if совпадение else None


def _отрицание(text: str, позиция: int) -> bool:
    """Есть ли рядом с найденным словом признак отказа от него."""
    начало = max(0, позиция - WINDOW)
    окно = text[начало:позиция + WINDOW].lower()
    return any(маркер in окно for маркер in NEGATIONS)


def _вырезка(text: str, позиция: int, ширина: int = 70) -> str:
    начало = max(0, позиция - ширина // 2)
    кусок = text[начало:позиция + ширина].replace("\n", " ").strip()
    return ("…" if начало else "") + кусок + "…"
