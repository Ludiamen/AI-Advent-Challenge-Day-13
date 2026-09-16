"""MemoryAgent — агент с явной моделью памяти.

Отличие этого дня от предыдущих в том, что у агента больше нет «истории» как
одной сущности. Есть три слоя с разным сроком жизни, разными хранилищами и
разными правилами записи, и каждый шаг агента проходит через них явно:

    вопрос
      -> маршрутизатор: нужно ли что-то сохранить надолго       (MemoryManager)
      -> запись реплики в краткосрочную память                  (MemoryManager)
      -> сборка промпта из слоёв по политике стадии             (PromptBuilder)
      -> вызов модели                                           (llm.Client)
      -> проверка ответа кодом по жёстким инвариантам           (StateValidator)
      -> при нарушении: повтор, затем эскалация модели
      -> запись ответа в краткосрочную память                   (MemoryManager)

Наружу агент отдаёт не только текст ответа, но и трейс: какие записи какого
слоя попали в промпт и во что это обошлось. Без трейса выполнить требование
задания «проверьте, какие данные попадают в каждый слой» нельзя — пришлось бы
верить на слово.

Публичное API:
  MemoryAgent(...)                  — создать агента
  .ask(question)                    — спросить с учётом всех включённых слоёв
  .plan()                           — составить план задачи и записать в рабочую память
  .start_task / .use_task / .transition / .finish_task
  .remember(target, ...)            — записать в указанный слой явно
  .set_layers(...)                  — включить и выключить слои (аблация)
  .info() / .stats() / .journal() / .files()
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from agent import catalog, interview, preferences, seed as seed_module
from agent.builder import BuiltPrompt, PromptBuilder
from agent.llm import Client, LLMError, Reply
from agent.memory.long import LongTermError
from agent.memory.manager import AUTO, LONG, SHORT, WORKING, MemoryManager
from agent.memory.router import DEFAULT_THRESHOLD, Routing
from agent.memory.short import DEFAULT_MAX_CHARS, DEFAULT_MAX_MESSAGES
from agent.memory.working import (
    DONE, PLANNING, ПО_КОМАНДЕ, ПРОДОЛЖИТЬ, TaskState,
    TransitionError, WorkingMemoryError,
)
from agent.preferences import Deviation, PreferenceChecker
from agent.scenarios import RunResult, Scenario, ScenarioRunner
from agent.validator import SoftResult, StateValidator, Violation

log = logging.getLogger("agent")

ALL_LAYERS = {SHORT, WORKING, LONG}

# Сколько раз агент пытается получить ответ, не нарушающий инварианты.
# Первая попытка — обычная; вторая — с напоминанием; третья — на модели
# следующей ступени. Дальше уже честнее отказать, чем жечь лимиты.
MAX_ATTEMPTS = 3

_ШАГ_ПЛАНА = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.{3,})$", re.MULTILINE)


class AgentError(RuntimeError):
    """Единственный тип ошибки, который агент выпускает наружу."""


@dataclass
class Answer:
    """Ответ агента вместе со всем, что понадобилось, чтобы его получить."""

    text: str
    prompt: BuiltPrompt | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    violations: list[Violation] = field(default_factory=list)
    deviations: list[Deviation] = field(default_factory=list)
    blocked: bool = False            # ответ так и не уложился в инварианты
    escalated_to: str = ""
    routing: Routing | None = None
    routing_entry: dict[str, Any] = field(default_factory=dict)
    soft: SoftResult | None = None

    def layers(self) -> dict[str, dict[str, int]]:
        return self.prompt.by_layer() if self.prompt else {}

    def trace(self) -> list[dict[str, Any]]:
        return self.prompt.trace() if self.prompt else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "attempts": self.attempts,
            "blocked": self.blocked,
            "escalated_to": self.escalated_to,
            "usage": self.usage,
            "violations": [н.to_dict() for н in self.violations],
            "deviations": [о.to_dict() for о in self.deviations],
            "layers": self.layers(),
            "trace": self.trace(),
            "routing": self.routing.to_dict() if self.routing else None,
            "routing_entry": self.routing_entry,
            "soft": self.soft.to_dict() if self.soft else None,
            "stage": self.prompt.stage if self.prompt else "",
            "task_id": self.prompt.task_id if self.prompt else "",
        }


class MemoryAgent:
    """Агент, у которого память разложена по трём слоям явно."""

    def __init__(
        self,
        model_key: str = "",
        user_id: str = "инженер",
        session: str = "основная",
        base_dir: str = "",
        task_id: str = "",
        layers: set[str] | None = None,
        router_mode: str = AUTO,
        threshold: float = DEFAULT_THRESHOLD,
        router_model: str = "",
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_chars: int = DEFAULT_MAX_CHARS,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        soft_check: bool = False,
        follow_profile: bool = True,
        seed_project: bool = True,
    ) -> None:
        # Пустой model_key означает «брать модель по роли»: на планировании и
        # на исполнении роли разные, и жёстко фиксировать одну модель не нужно.
        self.model_key = model_key
        self.layers = set(layers) if layers is not None else set(ALL_LAYERS)
        self.soft_check = soft_check

        self.client = Client(temperature=temperature, max_tokens=max_tokens)
        база = base_dir or os.getenv("MEMORY_DIR", "память")
        try:
            self.memory = MemoryManager(
                base_dir=база, user_id=user_id, session=session, client=self.client,
                router_mode=router_mode, threshold=threshold, router_model=router_model,
            )
        except (OSError, ValueError, LongTermError) as exc:
            # Сюда приходит и недопустимое имя пользователя: оно превращается в
            # путь, и проверка живёт в хранилище. Наружу это должно выйти
            # понятным отказом, а не стеком из недр памяти.
            raise AgentError(f"Не удалось открыть память: {exc}") from exc

        if seed_project:
            seed_module.seed(self.memory)

        self.builder = PromptBuilder(self.memory, max_messages=max_messages, max_chars=max_chars)
        self.validator = StateValidator(self.memory.long.profile, self.client)
        # Проверка предпочтений создаётся на каждый запрос заново: профиль
        # правят посреди разговора, и держать его слепок в поле значит однажды
        # проверить ответ по вчерашним настройкам.
        self.follow_profile = follow_profile
        self.runner = ScenarioRunner(self)

        self.task: TaskState | None = None
        if task_id:
            self.use_task(task_id)

    # --- свойства ------------------------------------------------------------

    @property
    def session(self) -> str:
        return self.memory.session

    @property
    def user_id(self) -> str:
        return self.memory.user_id

    @property
    def stage(self) -> str:
        return self.task.stage if self.task else ""

    def model_for(self, stage: str = "") -> str:
        """Какая модель отвечает на этой стадии.

        Планирование и исполнение разведены по ролям: на планировании цена
        ошибки выше, потому что на плане строится всё остальное.
        """
        if self.model_key:
            return self.model_key
        роль = "планирование" if (stage or self.stage) == PLANNING else "исполнение"
        return catalog.for_role(роль, offset=0)

    # --- слои ----------------------------------------------------------------

    def set_layers(self, layers: set[str]) -> None:
        """Включает и выключает слои. Этим делается аблация в сравнение.py."""
        неизвестные = layers - ALL_LAYERS
        if неизвестные:
            raise AgentError(
                f"Неизвестные слои: {', '.join(неизвестные)}. "
                f"Допустимы: {', '.join(sorted(ALL_LAYERS))}."
            )
        self.layers = set(layers)

    # --- задачи --------------------------------------------------------------

    def start_task(self, task_id: str, title: str = "", overwrite: bool = False) -> TaskState:
        """Заводит задачу и делает её текущей."""
        try:
            self.task = self.memory.working.create(task_id, title, overwrite=overwrite)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        self.memory._log("шаг-задачи", WORKING, task_id, title, applied=True,
                         reason="задача заведена, стадия planning")
        return self.task

    def use_task(self, task_id: str) -> TaskState:
        """Поднимает задачу из рабочей памяти — в том числе спустя дни."""
        try:
            self.task = self.memory.working.load(task_id)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        return self.task

    def drop_task(self) -> None:
        self.task = None

    def save_task(self) -> None:
        """Кладёт состояние текущей задачи на диск.

        Вызывается после каждого шага и каждой остановки: именно записанное
        состояние, а не переменная в памяти процесса, позволяет вернуться к
        задаче завтра и с другого запуска.
        """
        if self.task is not None:
            self.memory.working.save(self.task)

    # --- пауза и продолжение -------------------------------------------------

    def pause_task(self, причина: str = ПО_КОМАНДЕ, пояснение: str = "") -> TaskState:
        """Просит задачу остановиться. Текущий шаг при этом доводится до конца.

        Флаг ставится на том же объекте состояния, с которым работает исполнитель
        сценария, поэтому он увидит паузу перед следующим шагом. Прерывать шаг
        посреди вызова модели незачем: ответ уже оплачен, и выбрасывать его —
        значит платить дважды.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет — останавливать нечего.")
        if self.task.finished:
            raise AgentError("Задача уже завершена.")
        self.task.остановить(причина, ПРОДОЛЖИТЬ, пояснение)
        self.save_task()
        self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                         f"пауза: {причина}", applied=True,
                         reason=пояснение or "остановлено по команде")
        return self.task

    def continue_task(self) -> TaskState:
        """Снимает паузу, не запуская исполнение. Само продолжение — у сценария."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        self.task.продолжить()
        self.save_task()
        return self.task

    def answer_task(self, текст: str) -> TaskState:
        """Принимает ответ человека на остановку «не хватает сведений»."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        try:
            self.task.ответить(текст)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        self.save_task()
        self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                         текст, applied=True, reason="ответ человека на вопрос шага")
        return self.task

    def resume_scenario(
        self,
        task_id: str = "",
        ответ: str = "",
        on_step: Any = None,
        on_result: Any = None,
        по_шагам: bool = False,
    ) -> RunResult:
        """Продолжает отложенную задачу с того шага, где она стоит."""
        task_id = task_id or (self.task.task_id if self.task else "")
        if not task_id:
            raise AgentError("Не указано, какую задачу продолжать.")
        try:
            return self.runner.resume(task_id, ответ=ответ, on_step=on_step,
                                      on_result=on_result, по_шагам=по_шагам)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"Не удалось продолжить задачу «{task_id}»: {exc}") from exc

    def task_state(self) -> dict[str, Any]:
        """Полное состояние задачи для интерфейсов: этап, шаг, ожидание, пауза."""
        if self.task is None:
            return {"есть": False}
        з = self.task
        return {
            "есть": True,
            "task_id": з.task_id,
            "title": з.title,
            "сценарий": з.сценарий,
            "этап": з.stage,
            "этап_словами": з.stage_label,
            "разрешено": list(з.allowed()),
            "шаг": з.текущий_шаг,
            "шагов": з.шагов,
            "шаг_словами": з.шаг_словами,
            "шаги": [ш.to_dict() for ш in з.шаги],
            "ожидание": з.ожидание,
            "ожидание_текст": з.ожидание_текст,
            "пауза": з.пауза,
            "причина_паузы": з.причина_паузы,
            "ответы": list(з.ответы),
            "собрано": len(з.collected),
            "словами": з.состояние_словами,
        }

    def transition(self, stage: str, note: str = "") -> TaskState:
        """Переводит задачу на новую стадию; запрещённый переход — ошибка."""
        if self.task is None:
            raise AgentError("Активной задачи нет: сначала --задача или --новая-задача.")
        try:
            self.task.transition(stage, note)
        except TransitionError as exc:
            raise AgentError(str(exc)) from exc
        self.memory.working.save(self.task)
        self.memory._log("шаг-задачи", WORKING, self.task.task_id, f"-> {stage}", applied=True,
                         reason=note or "смена стадии")
        return self.task

    def finish_task(self, note: str = "") -> dict[str, Any]:
        """Завершает задачу: свёртка в журнал решений и очистка рабочей памяти."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        if self.task.stage != DONE and not self.task.can_go(DONE):
            raise AgentError(
                f"Из стадии «{self.task.stage}» нельзя сразу в done. "
                f"Разрешено: {', '.join(self.task.allowed())}."
            )
        запись = self.memory.finish_task(self.task, note, model_key=self.model_key)
        self.task = None
        return запись

    # --- запись в память -----------------------------------------------------

    def remember(self, target: str, value: str, key: str = "", section: str = "",
                 reason: str = "") -> dict[str, Any]:
        """Явная запись в указанный слой. Модель не участвует."""
        try:
            return self.memory.remember_explicit(
                target, value, key=key, section=section, reason=reason,
                task_id=self.task.task_id if self.task else "",
            )
        except Exception as exc:  # LongTermError, WorkingMemoryError
            raise AgentError(str(exc)) from exc

    def remember_reply(self, role: str, text: str) -> None:
        """Кладёт реплику в краткосрочную память, если слой включён.

        Нужен исполнителю сценария: сам он ходит в ask() служебными вызовами,
        которые в диалог не пишут, а вопрос человека и итоговый ответ в истории
        разговора быть должны.
        """
        if SHORT in self.layers:
            self.memory.remember_message(
                role, text,
                task_id=self.task.task_id if self.task else "",
                stage=self.stage,
            )

    def remember_step(self, key: str, value: str) -> TaskState:
        if self.task is None:
            raise AgentError("Активной задачи нет: промежуточный результат некуда класть.")
        self.task = self.memory.remember_step(self.task, key, value)
        return self.task

    # --- основной цикл -------------------------------------------------------

    def checker(self) -> PreferenceChecker:
        """Проверка ответа по текущему профилю пользователя."""
        return PreferenceChecker(self.memory.long.profile.load())

    def ask(
        self,
        question: str,
        layers: set[str] | None = None,
        model_key: str = "",
        step_role: str = "",
        personal: bool = True,
        internal: bool = False,
    ) -> Answer:
        """Полный цикл: маршрутизация, сборка, вызов, проверка, запись.

        model_key и step_role задаёт исполнитель сценария: у каждого шага своя
        модель и своя роль, и общие настройки агента их не переопределяют.

        personal=False — промежуточный шаг сценария: настройки пользователя ни в
        промпт не идут, ни по ответу не проверяются.

        internal=True — вызов служебный, а не разговор с человеком: шаг сценария
        получает на вход машинный текст, собранный из результатов предыдущих
        шагов. Такой текст нельзя ни разбирать маршрутизатором, ни класть в
        диалог. Проверено, что бывает иначе: в базу знаний попадали записи вроде
        «Объяснительное сообщение о разделении работы между расширением и
        схемой» — маршрутизатор принял кусок ответа архитектора за факт,
        сказанный пользователем, и записал его навсегда.
        """
        question = (question or "").strip()
        if not question:
            raise AgentError("Пустой вопрос.")
        слои = set(layers) if layers is not None else set(self.layers)

        # Правило 5: может быть, в реплике есть что-то для долговременной памяти.
        # Если долговременный слой выключен, спрашивать маршрутизатор незачем.
        маршрут: Routing | None = None
        запись_маршрута: dict[str, Any] = {}
        if LONG in слои and not internal:
            маршрут, запись_маршрута = self.memory.route(question)

        # Правило 2: сама реплика — в краткосрочную память.
        if SHORT in слои and not internal:
            self.memory.remember_message(
                "user", question,
                task_id=self.task.task_id if self.task else "",
                stage=self.stage,
            )

        промпт = self.builder.build(question, self.task, слои, step_role=step_role,
                                    personal=personal)
        ответ, попытки, нарушения, расхождения, эскалация = self._answer_within_rules(
            промпт, question, слои, model_key, step_role, personal
        )

        мягкая: SoftResult | None = None
        if self.soft_check and not нарушения:
            мягкая = self.validator.check_soft(ответ.text)

        if SHORT in слои and not internal:
            self.memory.remember_message(
                "assistant", ответ.text,
                task_id=self.task.task_id if self.task else "",
                stage=self.stage,
                tokens=ответ.total_tokens, cost=ответ.cost,
            )

        return Answer(
            text=ответ.text,
            prompt=промпт,
            usage=ответ.to_dict(),
            attempts=попытки,
            violations=нарушения,
            deviations=расхождения,
            blocked=bool(нарушения),
            escalated_to=эскалация,
            routing=маршрут,
            routing_entry=запись_маршрута,
            soft=мягкая,
        )

    def _answer_within_rules(
        self,
        prompt: BuiltPrompt,
        question: str,
        layers: set[str],
        model_key: str = "",
        step_role: str = "",
        personal: bool = True,
    ) -> tuple[Reply, int, list[Violation], list[Deviation], str]:
        """Получает ответ, который укладывается и в инварианты, и в профиль.

        Проверок две, и они разной силы. Инвариант — запрет проекта: ответ,
        который его нарушает, отдавать нельзя, и если переделать не удалось,
        агент честно говорит об этом. Расхождение с профилем — это «не так, как
        просил пользователь»: повторить стоит, но ответ по существу верен, и
        отдать его лучше, чем не отдать ничего.

        Лестница повторов общая: сначала просим переделать ту же модель, и
        только если не помогло — поднимаемся на ступень. Прыгать сразу на самую
        дорогую незачем, чаще всего хватает напоминания.
        """
        ключ_модели = model_key or self.model_for(prompt.stage)
        сообщения = prompt.messages
        эскалация = ""
        нарушения: list[Violation] = []
        расхождения: list[Deviation] = []

        for попытка in range(1, MAX_ATTEMPTS + 1):
            try:
                ответ = self.client.call(ключ_модели, сообщения)
            except LLMError as exc:
                raise AgentError(str(exc)) from exc

            нарушения = self.validator.check(ответ.text) if LONG in layers else []
            расхождения = (
                self.checker().check(ответ.text)
                if personal and self.follow_profile and LONG in layers else []
            )
            жёсткие = [о for о in расхождения if о.hard]
            if not нарушения and not жёсткие:
                return ответ, попытка, [], расхождения, эскалация

            log.warning(
                "Попытка %d: нарушений %d, расхождений с профилем %d",
                попытка, len(нарушения), len(жёсткие),
            )
            if попытка == MAX_ATTEMPTS:
                break

            напоминания = []
            if нарушения:
                напоминания.append(self.validator.reminder(нарушения))
            if жёсткие:
                напоминания.append(self.checker().reminder(жёсткие))
            повтор = self.builder.build(
                question, self.task, layers,
                extra_note="\n\n".join(напоминания), step_role=step_role,
                personal=personal,
            )
            сообщения = повтор.messages
            # Эскалация — только из-за инварианта. Расхождение с профилем
            # косметическое: платить за ответ вчетверо дороже потому, что он на
            # двадцать слов длиннее просимого, — плохая сделка. Повторяем на той
            # же модели: напоминание обычно помогает.
            if попытка >= 2 and нарушения:
                следующая = catalog.escalate(ключ_модели)
                if следующая != ключ_модели:
                    ключ_модели, эскалация = следующая, следующая

        return ответ, MAX_ATTEMPTS, нарушения, расхождения, эскалация

    # --- планирование --------------------------------------------------------

    def plan(self, note: str = "") -> Answer:
        """Просит модель составить план и кладёт его в рабочую память.

        Это единственное место, где ответ модели превращается в структуру, а не
        остаётся текстом: план — рабочие данные задачи, и жить он должен в
        рабочей памяти, а не в переписке.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет: планировать нечего.")
        if self.task.stage != PLANNING:
            raise AgentError(
                f"План составляется на стадии planning, а задача сейчас в «{self.task.stage}»."
            )
        вопрос = (
            f"Составь план задачи «{self.task.title or self.task.task_id}». "
            + (note or "")
            + " Дай нумерованный список шагов, по одному шагу в строке, без пояснений между ними."
        )
        ответ = self.ask(вопрос)
        шаги = [ш.strip() for ш in _ШАГ_ПЛАНА.findall(ответ.text)][:12]
        if шаги:
            self.task = self.memory.remember_plan(self.task, шаги)
        return ответ

    # --- профиль пользователя ------------------------------------------------

    def profile(self) -> dict[str, Any]:
        return self.memory.long.profile.load()

    def needs_setup(self) -> bool:
        """Профиль ещё не настраивали — стоит предложить мастер."""
        return interview.needs_setup(self.profile())

    @staticmethod
    def setup_questions() -> list[dict[str, Any]]:
        return interview.questions()

    def setup(self, answers: dict[str, Any]) -> dict[str, Any]:
        """Применяет ответы мастера настройки к профилю."""
        try:
            профиль = interview.apply(self.profile(), answers)
        except preferences.PreferenceError as exc:
            raise AgentError(str(exc)) from exc
        self.memory.long.profile.save(профиль)
        self.memory._log("явное-указание", LONG, "профиль",
                         preferences.summary(профиль), applied=True,
                         reason="мастер настройки пройден")
        return профиль

    def use_template(self, name: str) -> dict[str, Any]:
        """Берёт готовую заготовку профиля целиком."""
        try:
            профиль = interview.from_template(name, self.profile())
        except preferences.PreferenceError as exc:
            raise AgentError(str(exc)) from exc
        self.memory.long.profile.save(профиль)
        self.memory._log("явное-указание", LONG, "профиль",
                         f"заготовка «{name}»: {preferences.summary(профиль)}",
                         applied=True, reason="выбрана готовая заготовка профиля")
        return профиль

    def set_preference(self, section: str, key: str, value: Any) -> dict[str, Any]:
        """Правит одно предпочтение; недопустимое значение отклоняется."""
        try:
            return self.memory.long.profile.update(section, key, value)
        except Exception as exc:
            raise AgentError(str(exc)) from exc

    # --- сценарии --------------------------------------------------------------

    def scenarios(self) -> list[Scenario]:
        return self.memory.long.scenarios.all()

    def match_scenario(self, query: str) -> Scenario | None:
        """Есть ли сценарий, чей триггер сработал на этом запросе."""
        return self.memory.long.scenarios.match(query)

    def add_scenario(self, scenario: Scenario) -> Scenario:
        try:
            итог = self.memory.long.scenarios.add(scenario)
        except Exception as exc:
            raise AgentError(str(exc)) from exc
        self.memory._log("явное-указание", LONG, "сценарии", итог.digest(), applied=True,
                         reason="сценарий добавлен пользователем")
        return итог

    def remove_scenario(self, name: str) -> bool:
        return self.memory.long.scenarios.remove(name)

    def run_scenario(
        self,
        query: str,
        name: str = "",
        on_step: Any = None,
        finish: bool = True,
        on_result: Any = None,
        по_шагам: bool = False,
    ) -> RunResult:
        """Исполняет сценарий: каждый шаг — свой агент, своя модель, своя стадия."""
        сценарий = (
            self.memory.long.scenarios.get(name) if name else self.match_scenario(query)
        )
        if сценарий is None:
            подсказка = (
                f"Нет сценария «{name}»." if name
                else "Ни один сценарий не сработал на этом запросе."
            )
            есть = ", ".join(с.имя for с in self.scenarios()) or "ни одного"
            raise AgentError(f"{подсказка} Заведено сценариев: {есть}.")
        try:
            return self.runner.run(сценарий, query, on_step=on_step, finish=finish,
                                   on_result=on_result, по_шагам=по_шагам)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"Сценарий «{сценарий.имя}» прервался: {exc}") from exc

    # --- сводка --------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Состояние агента для интерфейсов."""
        модель = self.model_for()
        описание = catalog.get(модель)
        return {
            "model_key": модель,
            "model_label": описание.label,
            "model_fixed": bool(self.model_key),
            "provider": описание.provider,
            "free": описание.free,
            "user_id": self.user_id,
            "session": self.session,
            "layers": sorted(self.layers),
            "router_mode": self.memory.router_mode,
            "threshold": self.memory.threshold,
            "task": self.task.to_dict() if self.task else None,
            "stage": self.stage,
            "allowed": list(self.task.allowed()) if self.task else [],
            "soft_check": self.soft_check,
            "follow_profile": self.follow_profile,
            "profile": preferences.summary(self.profile()),
            "needs_setup": self.needs_setup(),
            "scenarios": [с.digest() for с in self.scenarios()],
            "spent": dict(self.client.spent),
        }

    def stats(self) -> dict[str, Any]:
        return self.memory.stats()

    def journal(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.memory.journal(limit)

    def files(self) -> dict[str, str]:
        return self.memory.files()

    def tasks(self) -> list[dict[str, Any]]:
        return self.memory.working.tasks()

    def close(self) -> None:
        self.client.close()
