#!/usr/bin/env python3
"""Тесты модели памяти. По умолчанию без сети.

    python tests.py              — все тесты без обращений к API
    python tests.py --живые      — плюс проверки, которым нужен реальный ключ
    python tests.py -v           — подробный вывод

Сетевых вызовов в основном наборе нет намеренно: правила маршрутизации, границы
слоёв и проверка инвариантов — это код, и он должен проверяться без оглядки на
доступность провайдера и на лимиты бесплатного тарифа.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import catalog, interview, preferences, seed as seed_module
from agent.builder import POLICY, PromptBuilder
from agent.memory.long import LongTermError, LongTermMemory
from agent.memory.manager import LONG, OFF, SHORT, WORKING, MemoryManager
from agent.memory.router import Routing, _parse
from agent.memory.short import ShortTermMemory
from agent.memory.working import (
    DONE, EXECUTION, PLANNING, VALIDATION, TaskState, TaskStep, TransitionError,
    WorkingMemory, WorkingMemoryError, ЗАПУСТИТЬ, ИЗ_ПЛАНА, ИЗ_СЦЕНАРИЯ,
    НАРУШЕН_ИНВАРИАНТ, НА_ПЕРЕХОДЕ, НЕТ_СВЕДЕНИЙ, НИЧЕГО, ОЖИДАНИЯ, ОТВЕТ,
    ПОДТВЕРДИТЬ, ПО_КОМАНДЕ, ПРОДОЛЖИТЬ, РЕШЕНИЕ,
)
from agent.preferences import PreferenceChecker, PreferenceError
from agent.scenarios import Scenario, ScenarioError, ScenarioStore, Step
from agent.validator import StateValidator

ЖИВЫЕ = "--живые" in sys.argv
if ЖИВЫЕ:
    sys.argv.remove("--живые")


class ВременнаяПамять(unittest.TestCase):
    """Общий каркас: каждый тест работает на своей копии памяти."""

    def setUp(self) -> None:
        self.каталог = tempfile.mkdtemp(prefix="тест-памяти-")
        self.память = MemoryManager(base_dir=self.каталог, router_mode=OFF)

    def tearDown(self) -> None:
        shutil.rmtree(self.каталог, ignore_errors=True)


# --- краткосрочная память -----------------------------------------------------

class КраткосрочнаяПамять(unittest.TestCase):

    def setUp(self) -> None:
        self.память = ShortTermMemory(":memory:")

    def test_окно_ограничено_числом_сообщений(self):
        for i in range(20):
            self.память.append("с", "user" if i % 2 == 0 else "assistant", f"реплика {i}")
        окно = self.память.window("с", max_messages=6)
        self.assertEqual(len(окно), 6)
        self.assertEqual(окно[-1]["content"], "реплика 19")

    def test_окно_ограничено_символами(self):
        self.память.append("с", "user", "х" * 5000)
        self.память.append("с", "assistant", "короткая")
        окно = self.память.window("с", max_messages=10, max_chars=1000)
        # Первая реплика не влезает по символам, но одна запись остаётся всегда:
        # пустое окно хуже, чем окно из одного сообщения.
        self.assertEqual(len(окно), 1)
        self.assertEqual(окно[0]["content"], "короткая")

    def test_сессии_не_смешиваются(self):
        self.память.append("работа", "user", "про работу")
        self.память.append("черновик", "user", "про черновик")
        self.assertEqual(len(self.память.all("работа")), 1)
        self.assertEqual(len(self.память.all("черновик")), 1)

    def test_пустая_реплика_не_сохраняется(self):
        with self.assertRaises(Exception):
            self.память.append("с", "user", "   ")

    def test_неизвестная_роль_отклоняется(self):
        with self.assertRaises(Exception):
            self.память.append("с", "system", "текст")


# --- рабочая память -----------------------------------------------------------

class РабочаяПамять(unittest.TestCase):

    def setUp(self) -> None:
        self.каталог = tempfile.mkdtemp()
        self.память = WorkingMemory(self.каталог)

    def tearDown(self) -> None:
        shutil.rmtree(self.каталог, ignore_errors=True)

    def test_разрешённый_маршрут_проходит_целиком(self):
        задача = self.память.create("з1", "тест")
        for стадия in (EXECUTION, VALIDATION, DONE):
            задача.transition(стадия)
        self.assertTrue(задача.finished)
        self.assertEqual(len(задача.transitions), 3)

    def test_прыжок_через_стадию_отклоняется(self):
        задача = self.память.create("з2")
        with self.assertRaises(TransitionError):
            задача.transition(DONE)
        self.assertEqual(задача.stage, PLANNING)

    def test_возвраты_разрешены(self):
        задача = self.память.create("з3")
        задача.transition(EXECUTION)
        задача.transition(PLANNING)          # план оказался негодным
        задача.transition(EXECUTION)
        задача.transition(VALIDATION)
        задача.transition(EXECUTION)         # нашли дефект
        self.assertEqual(задача.stage, EXECUTION)

    def test_из_done_никуда(self):
        задача = self.память.create("з4")
        for стадия in (EXECUTION, VALIDATION, DONE):
            задача.transition(стадия)
        self.assertEqual(задача.allowed(), ())
        with self.assertRaises(TransitionError):
            задача.transition(PLANNING)

    def test_состояние_переживает_перезапуск(self):
        задача = self.память.create("з5", "перенос")
        задача.transition(EXECUTION)
        задача.remember("таблиц", "37")
        задача.set_plan(["шаг один", "шаг два"])
        self.память.save(задача)

        другая = WorkingMemory(self.каталог).load("з5")
        self.assertEqual(другая.stage, EXECUTION)
        self.assertEqual(другая.collected["таблиц"], "37")
        self.assertEqual(другая.plan, ["шаг один", "шаг два"])

    def test_повторное_создание_отклоняется(self):
        self.память.create("з6")
        with self.assertRaises(WorkingMemoryError):
            self.память.create("з6")

    def test_недопустимый_идентификатор(self):
        for плохой in ("../побег", "имя с пробелом", "", "a" * 100):
            with self.assertRaises(WorkingMemoryError):
                self.память.create(плохой)

    def test_отсутствующая_задача_даёт_понятную_ошибку(self):
        with self.assertRaises(WorkingMemoryError):
            self.память.load("нет-такой")


class СостояниеЗадачи(unittest.TestCase):
    """Три части состояния: этап, шаг и ожидаемое действие."""

    def setUp(self) -> None:
        self.каталог = tempfile.mkdtemp()
        self.память = WorkingMemory(self.каталог)

    def tearDown(self) -> None:
        shutil.rmtree(self.каталог, ignore_errors=True)

    def _с_шагами(self, ид="з"):
        задача = self.память.create(ид, "проверка")
        задача.set_steps([
            TaskStep(1, "аналитик", ИЗ_СЦЕНАРИЯ, PLANNING, может_спросить=True),
            TaskStep(2, "backend", ИЗ_СЦЕНАРИЯ, EXECUTION),
        ], сценарий="проба")
        return задача

    def test_свежая_задача_ждёт_запуска(self):
        задача = self.память.create("новая")
        self.assertEqual(задача.ожидание, ЗАПУСТИТЬ)
        # Пояснение заполняется само, иначе интерфейс печатает «ожидание: — ».
        self.assertTrue(задача.ожидание_текст)

    def test_указатель_шага_двигается_при_завершении(self):
        задача = self._с_шагами()
        self.assertEqual(задача.шаг.имя, "аналитик")
        задача.начать_шаг()
        self.assertEqual(задача.шаг.состояние, "идёт")
        задача.закончить_шаг("требования собраны")
        self.assertEqual(задача.шаг.имя, "backend")
        self.assertEqual(задача.шаги[0].состояние, "готов")
        self.assertIn("требования", задача.шаги[0].выжимка)

    def test_пройденные_шаги_видно_по_состоянию(self):
        задача = self._с_шагами()
        задача.закончить_шаг("раз")
        задача.закончить_шаг("два")
        self.assertTrue(задача.шаги_пройдены)

    def test_пауза_не_меняет_этап(self):
        # Пауза — флаг поверх стадии, а не пятая стадия автомата.
        задача = self._с_шагами()
        задача.transition(EXECUTION)
        задача.остановить(ПО_КОМАНДЕ)
        self.assertTrue(задача.пауза)
        self.assertEqual(задача.stage, EXECUTION)
        self.assertEqual(задача.allowed(), (VALIDATION, PLANNING))

    def test_пауза_снимается(self):
        задача = self._с_шагами()
        задача.остановить(НА_ПЕРЕХОДЕ, ПОДТВЕРДИТЬ, "перейти к исполнению?")
        self.assertEqual(задача.ожидание, ПОДТВЕРДИТЬ)
        задача.продолжить()
        self.assertFalse(задача.пауза)
        self.assertEqual(задача.ожидание, ПРОДОЛЖИТЬ)

    def test_ответ_снимает_паузу_и_ложится_отдельно(self):
        задача = self._с_шагами()
        задача.остановить(НЕТ_СВЕДЕНИЙ, ОТВЕТ, "какой SRID?")
        задача.ответить("3857")
        self.assertFalse(задача.пауза)
        self.assertEqual(len(задача.ответы), 1)
        self.assertEqual(задача.ответы[0]["ответ"], "3857")
        # Ответ человека — не то же, что собранное агентом.
        self.assertEqual(задача.collected, {})

    def test_пустой_ответ_отклоняется(self):
        задача = self._с_шагами()
        задача.остановить(НЕТ_СВЕДЕНИЙ, ОТВЕТ, "какой SRID?")
        with self.assertRaises(WorkingMemoryError):
            задача.ответить("   ")

    def test_неизвестное_ожидание_отклоняется(self):
        задача = self._с_шагами()
        with self.assertRaises(WorkingMemoryError):
            задача.ждать("подумать")

    def test_неизвестная_причина_паузы_отклоняется(self):
        задача = self._с_шагами()
        with self.assertRaises(WorkingMemoryError):
            задача.остановить("настроение")

    def test_смена_этапа_сбрасывает_прежнее_ожидание(self):
        # Иначе после перехода на экране остаётся «подтвердите переход».
        задача = self._с_шагами()
        задача.остановить(НА_ПЕРЕХОДЕ, ПОДТВЕРДИТЬ, "перейти?")
        задача.продолжить()
        задача.transition(EXECUTION)
        self.assertEqual(задача.ожидание, ПРОДОЛЖИТЬ)

    def test_завершение_ставит_ожидание_ничего(self):
        задача = self._с_шагами()
        for стадия in (EXECUTION, VALIDATION, DONE):
            задача.transition(стадия)
        self.assertEqual(задача.ожидание, НИЧЕГО)

    def test_состояние_переживает_запись_и_чтение(self):
        # Главное свойство: продолжить можно из другого процесса.
        задача = self._с_шагами("живучая")
        задача.начать_шаг()
        задача.закончить_шаг("готово")
        задача.запрос = "исходный запрос"
        задача.остановить(НЕТ_СВЕДЕНИЙ, ОТВЕТ, "какой SRID?")
        self.память.save(задача)

        другая = WorkingMemory(self.каталог).load("живучая")
        self.assertTrue(другая.пауза)
        self.assertEqual(другая.ожидание, ОТВЕТ)
        self.assertEqual(другая.текущий_шаг, 2)
        self.assertEqual(другая.запрос, "исходный запрос")
        # Шаги должны подняться объектами, а не словарями.
        self.assertIsInstance(другая.шаг, TaskStep)
        self.assertEqual(другая.шаг.имя, "backend")

    def test_план_превращается_в_шаги(self):
        задача = self.память.create("ручная", "без сценария")
        задача.set_plan(["выписать таблицы", "описать модели"])
        self.assertEqual(задача.шагов, 2)
        self.assertEqual(задача.шаг.источник, ИЗ_ПЛАНА)

    def test_план_не_затирает_шаги_сценария(self):
        задача = self._с_шагами()
        задача.set_plan(["посторонний пункт"])
        self.assertEqual([ш.имя for ш in задача.шаги], ["аналитик", "backend"])

    def test_состояние_словами_называет_все_три_части(self):
        задача = self._с_шагами()
        строка = задача.состояние_словами
        for кусок in ("этап", "шаг", "ждём"):
            self.assertIn(кусок, строка)

    def test_все_ожидания_имеют_пояснение(self):
        from agent.memory.working import ОЖИДАНИЯ_СЛОВАМИ
        self.assertEqual(set(ОЖИДАНИЯ), set(ОЖИДАНИЯ_СЛОВАМИ))


# --- долговременная память ----------------------------------------------------

class ДолговременнаяПамять(unittest.TestCase):

    def setUp(self) -> None:
        self.каталог = tempfile.mkdtemp()
        self.память = LongTermMemory(self.каталог, "кто-то")

    def tearDown(self) -> None:
        shutil.rmtree(self.каталог, ignore_errors=True)

    def test_каждое_хранилище_в_своём_файле(self):
        self.память.profile.update("ограничения", "бд", "PostgreSQL")
        self.память.decisions.add("Стек", "Django")
        self.память.knowledge.add("ф1", "факт")
        self.память.scenarios.add(Scenario(имя="с", шаги=[Step("а", "делай")]))
        пути = set(self.память.files().values())
        # Профиль, сценарии, решения и знания — четыре разных файла: у каждого
        # свой режим записи, и смешивать их значит терять это различие.
        self.assertEqual(len(пути), 4)
        for путь in пути:
            self.assertTrue(os.path.exists(путь), путь)

    def test_профиль_перезаписывается_а_решения_дописываются(self):
        self.память.profile.update("ограничения", "бд", "PostgreSQL 16")
        self.память.profile.update("ограничения", "бд", "PostgreSQL 17")
        self.assertEqual(self.память.profile.load()["ограничения"]["бд"], "PostgreSQL 17")

        self.память.decisions.add("Первое", "текст один")
        self.память.decisions.add("Второе", "текст два")
        self.assertEqual(len(self.память.decisions.all()), 2)

    def test_имя_пользователя_не_выводит_за_каталог(self):
        # Имя пользователя превращается в путь. Без проверки «--кто ../../чужой»
        # записывает профиль за пределы каталога памяти — проверено, записывал.
        for плохое in ("../../чужой", "кто/то", "..", "", "a" * 100):
            with self.assertRaises(LongTermError, msg=плохое):
                LongTermMemory(self.каталог, плохое)

    def test_обычное_имя_принимается(self):
        память = LongTermMemory(self.каталог, "инженер-2")
        self.assertTrue(os.path.realpath(память.directory).startswith(
            os.path.realpath(self.каталог)))

    def test_неизвестный_раздел_профиля(self):
        with self.assertRaises(LongTermError):
            self.память.profile.update("настроение", "тон", "бодрый")

    def test_недопустимое_значение_предпочтения_отклоняется(self):
        # Записать «длина: очень кратко» значило бы сохранить то, что не попадёт
        # ни в промпт, ни в проверку, — и никто бы не понял почему.
        with self.assertRaises(LongTermError):
            self.память.profile.update("формат", "длина", "очень кратко")

    def test_профиль_дополняется_умолчаниями_при_чтении(self):
        профиль = self.память.profile.load()
        for раздел in preferences.SECTIONS:
            for поле in preferences.fields(раздел):
                self.assertIn(поле.key, профиль[раздел])

    def test_жёсткий_инвариант_без_значений_отклоняется(self):
        # Инвариант, который нечем проверить, хуже, чем его отсутствие:
        # он создаёт ложное чувство защиты.
        with self.assertRaises(LongTermError):
            self.память.profile.add_invariant(
                {"код": "пустой", "правило": "нельзя", "тип": "запрет-слов", "значения": []}
            )

    def test_негодная_регулярка_отклоняется(self):
        with self.assertRaises(LongTermError):
            self.память.profile.add_invariant(
                {"код": "битый", "правило": "нельзя", "тип": "запрет-регулярок",
                 "значения": ["[незакрытая"]}
            )

    def test_инвариант_обновляется_по_коду(self):
        for правило in ("первая версия", "вторая версия"):
            self.память.profile.add_invariant(
                {"код": "один", "правило": правило, "тип": "мягкий"}
            )
        правила = self.память.profile.invariants()
        self.assertEqual(len(правила), 1)
        self.assertEqual(правила[0]["правило"], "вторая версия")

    def test_знания_отбираются_по_релевантности(self):
        self.память.knowledge.add("схема", "Схема gissys: account, group, organization",
                                  tags=["planning", "бд"])
        self.память.knowledge.add("фронт", "OpenLayers 2.13 рисует слои", tags=["execution"])
        отобрано = [ф["id"] for ф in self.память.knowledge.relevant("что в схеме gissys")]
        self.assertEqual(отобрано, ["схема"])

    def test_несовпавший_запрос_не_тянет_ничего(self):
        self.память.knowledge.add("схема", "Схема gissys", tags=["planning"])
        self.assertEqual(self.память.knowledge.relevant("погода в Москве"), [])

    def test_факт_уточняется_а_не_дублируется(self):
        self.память.knowledge.add("в", "PostGIS 3.4")
        self.память.knowledge.add("в", "PostGIS 3.6")
        факты = self.память.knowledge.all()
        self.assertEqual(len(факты), 1)
        self.assertEqual(факты[0]["текст"], "PostGIS 3.6")


# --- правила маршрутизации ----------------------------------------------------

class ПравилаМаршрутизации(ВременнаяПамять):

    def test_реплика_идёт_в_короткую_память(self):
        self.память.remember_message("user", "вопрос")
        self.assertEqual(self.память.stats()[SHORT]["реплик"], 1)
        последняя = self.память.journal(1)[0]
        self.assertEqual(последняя["правило"], "реплика-диалога")
        self.assertEqual(последняя["слой"], SHORT)

    def test_шаг_задачи_идёт_в_рабочую_память(self):
        задача = self.память.working.create("з", "тест")
        self.память.remember_step(задача, "таблиц", "37")
        self.assertEqual(self.память.working.load("з").collected["таблиц"], "37")
        self.assertEqual(self.память.journal(1)[0]["слой"], WORKING)

    def test_явное_указание_идёт_куда_сказано(self):
        self.память.remember_explicit("знания", "В gisdata 37 таблиц", key="состав")
        запись = self.память.journal(1)[0]
        self.assertEqual(запись["правило"], "явное-указание")
        self.assertEqual(запись["подслой"], "знания")
        self.assertEqual(len(self.память.long.knowledge.all()), 1)

    def test_явное_указание_в_неизвестный_слой_отклоняется(self):
        with self.assertRaises(LongTermError):
            self.память.remember_explicit("подсознание", "что-то")

    def test_свёртка_идёт_на_выбранной_модели(self):
        # Иначе пять шагов честно идут на выбранной модели, а свёртка в конце
        # уходит к своей роли — и роняет весь прогон на последнем шаге.
        задача = self.память.working.create("з", "перенос")
        for стадия in (EXECUTION, VALIDATION):
            задача.transition(стадия)
        self.память.working.save(задача)

        class Считающий:
            def __init__(self): self.модели = []
            spent = {"calls": 0, "tokens": 0, "cost": 0.0}
            def call(self, model_key, messages, **kwargs):
                from agent.llm import Reply
                self.модели.append(model_key)
                return Reply(text='{"заголовок":"и","решение":"р","причина":"п"}',
                             model_key=model_key)
            def close(self): pass

        считающий = Считающий()
        self.память.client = считающий
        self.память.finish_task(задача, model_key="ds-flash")
        self.assertEqual(считающий.модели, ["ds-flash"])

    def test_без_выбора_свёртка_идёт_по_роли(self):
        задача = self.память.working.create("з2", "перенос")
        for стадия in (EXECUTION, VALIDATION):
            задача.transition(стадия)
        self.память.working.save(задача)
        self.assertEqual(self.память.summarizer_model, catalog.for_role("сжатие", offset=0))

    def test_завершение_задачи_переносит_её_в_решения(self):
        задача = self.память.working.create("з", "перенос моделей")
        self.память.remember_step(задача, "итог", "модели описаны")
        for стадия in (EXECUTION, VALIDATION):
            задача.transition(стадия)
        self.память.working.save(задача)

        запись = self.память.finish_task(задача)
        self.assertEqual(len(self.память.long.decisions.all()), 1)
        self.assertIn("перенос моделей", запись["заголовок"])
        # Рабочая память задачи очищена: её итог теперь живёт в журнале решений.
        self.assertEqual(self.память.working.tasks(), [])

    def test_очистка_диалога_не_трогает_другие_слои(self):
        self.память.remember_message("user", "реплика")
        задача = self.память.working.create("з")
        self.память.remember_step(задача, "к", "з")
        self.память.remember_explicit("знания", "факт", key="ф")

        self.память.short.clear(self.память.session)
        сводка = self.память.stats()
        self.assertEqual(сводка[SHORT]["реплик"], 0)
        self.assertEqual(сводка[WORKING]["задач"], 1)
        self.assertEqual(сводка[LONG]["знаний"], 1)

    def test_маршрутизатор_выключен_ничего_не_пишет(self):
        было = self.память.long.profile.load()
        предложение, запись = self.память.route("Отвечай кратко")
        self.assertFalse(предложение.wants_write)
        self.assertFalse(запись["применено"])
        # Профиль не пуст даже без записей: предпочтения всегда имеют умолчания.
        # Значит, проверять надо неизменность, а не пустоту.
        self.assertEqual(self.память.long.profile.load()["формат"], было["формат"])

    def test_отклонённое_предложение_видно_в_журнале(self):
        # Ниже порога — записи нет, но след остаётся: потом видно, что именно
        # агент решил не запоминать.
        self.память.router_mode = "авто"
        self.память.router = _ЗаглушкаМаршрутизатора(
            Routing(target="знания", key="к", value="факт", confidence=0.3)
        )
        предложение, запись = self.память.route("какая-то реплика")
        self.assertFalse(запись["применено"])
        self.assertIn("ниже порога", запись["причина"])
        self.assertEqual(len(self.память.long.knowledge.all()), 0)

    def test_уверенное_предложение_применяется(self):
        self.память.router_mode = "авто"
        self.память.router = _ЗаглушкаМаршрутизатора(
            Routing(target="знания", key="версия", value="PostGIS 3.6", confidence=0.9)
        )
        _, запись = self.память.route("у нас PostGIS 3.6")
        self.assertTrue(запись["применено"])
        self.assertEqual(self.память.long.knowledge.all()[0]["текст"], "PostGIS 3.6")

    def test_сбой_маршрутизатора_не_ломает_запись(self):
        self.память.router_mode = "авто"
        self.память.router = _ЗаглушкаМаршрутизатора(Routing(failed=True))
        предложение, запись = self.память.route("реплика")
        self.assertTrue(предложение.failed)
        self.assertFalse(запись["применено"])


class _ЗаглушкаМаршрутизатора:
    """Маршрутизатор с заранее известным ответом — чтобы тесты не ходили в сеть."""

    def __init__(self, routing: Routing) -> None:
        self.routing = routing

    def classify(self, text: str) -> Routing:
        return self.routing


# --- разбор ответа маршрутизатора ---------------------------------------------

class РазборОтветаМодели(unittest.TestCase):

    def test_чистый_json(self):
        разбор = _parse('{"слой":"знания","ключ":"к","значение":"з","уверенность":0.8}')
        self.assertEqual(разбор.target, "знания")
        self.assertAlmostEqual(разбор.confidence, 0.8)

    def test_json_в_markdown(self):
        разбор = _parse('```json\n{"слой":"профиль","раздел":"стиль","значение":"кратко",'
                        '"уверенность":0.9}\n```')
        self.assertEqual(разбор.target, "профиль")
        self.assertEqual(разбор.section, "стиль")

    def test_json_с_болтовнёй_вокруг(self):
        разбор = _parse('Конечно! Вот ответ: {"слой":"нет","уверенность":0} — надеюсь, помог.')
        self.assertEqual(разбор.target, "нет")

    def test_хвост_после_объекта_не_мешает(self):
        # Слабые модели присылают валидный объект и следом обрывок служебного
        # тега. Срез «от первой { до последней }» на этом ломается.
        разбор = _parse('{"слой":"нет","уверенность":0}</think>{обрывок')
        self.assertEqual(разбор.target, "нет")

    def test_берётся_первый_из_двух_объектов(self):
        разбор = _parse('{"слой":"знания","значение":"факт","уверенность":0.9} {"слой":"нет"}')
        self.assertEqual(разбор.value, "факт")

    def test_скобка_внутри_строки_не_обрывает_разбор(self):
        разбор = _parse('{"слой":"знания","значение":"вот } скобка","уверенность":0.9}')
        self.assertEqual(разбор.value, "вот } скобка")

    def test_мусор_даёт_none(self):
        self.assertIsNone(_parse("я не понял вопроса"))

    def test_неизвестный_слой_даёт_none(self):
        self.assertIsNone(_parse('{"слой":"подсознание","уверенность":1}'))

    def test_уверенность_загоняется_в_границы(self):
        self.assertEqual(_parse('{"слой":"нет","уверенность":7}').confidence, 1.0)
        self.assertEqual(_parse('{"слой":"нет","уверенность":-3}').confidence, 0.0)


# --- сборка промпта -----------------------------------------------------------

class СборкаПромпта(ВременнаяПамять):

    def setUp(self) -> None:
        super().setUp()
        seed_module.seed(self.память)
        self.память.remember_message("user", "прошлая реплика")
        self.сборщик = PromptBuilder(self.память)

    def test_выключенный_слой_не_даёт_записей(self):
        промпт = self.сборщик.build("вопрос", layers={SHORT})
        self.assertNotIn(LONG, промпт.by_layer())
        причины = [б.why for б in промпт.blocks if б.layer == LONG]
        self.assertTrue(all("выключена" in п for п in причины))

    def test_инварианты_идут_всегда(self):
        for стадия in (PLANNING, EXECUTION, VALIDATION, DONE):
            задача = TaskState(task_id="з", stage=стадия)
            промпт = self.сборщик.build("вопрос", задача)
            блок = [б for б in промпт.blocks if б.name == "инварианты"][0]
            self.assertTrue(блок.included, f"инварианты пропали на стадии {стадия}")

    @staticmethod
    def _фактов(промпт) -> int:
        """Сколько записей знаний попало в промпт (0, если блок не включён)."""
        блоки = [б for б in промпт.included if б.name == "знания"]
        return len(блоки[0].entries) if блоки else 0

    def test_знания_зависят_от_стадии(self):
        # Факты о системе нужны, когда строят план, и мешают, когда проверяют
        # уже написанный код. Политика стадий именно это и задаёт.
        вопрос = "как перенести схему gissys"
        планирование = self.сборщик.build(вопрос, TaskState("з", stage=PLANNING))
        проверка = self.сборщик.build(вопрос, TaskState("з", stage=VALIDATION))
        self.assertGreater(self._фактов(планирование), self._фактов(проверка))

    def test_на_завершённой_задаче_знаний_нет(self):
        промпт = self.сборщик.build("итог?", TaskState("з", stage=DONE))
        self.assertEqual(self._фактов(промпт), 0)

    def test_настройки_идут_на_всех_стадиях(self):
        # Иначе стадия, где профиль не показали, гарантированно дала бы
        # расхождение с ним и лишний повтор.
        for стадия in (PLANNING, EXECUTION, VALIDATION, DONE):
            промпт = self.сборщик.build("вопрос", TaskState("з", stage=стадия))
            блок = [б for б in промпт.blocks if б.name == "настройки пользователя"][0]
            self.assertTrue(блок.included, f"настройки пропали на стадии {стадия}")

    def test_на_промежуточном_шаге_настроек_нет(self):
        промпт = self.сборщик.build("вопрос", personal=False)
        блок = [б for б in промпт.blocks if б.name == "настройки пользователя"][0]
        self.assertFalse(блок.included)
        self.assertIn("следующий агент", блок.why)

    def test_роль_шага_попадает_в_ядро(self):
        промпт = self.сборщик.build("вопрос", step_role="РОЛЬ НА ЭТОМ ШАГЕ: аналитик.")
        блок = [б for б in промпт.blocks if б.name == "роль шага"][0]
        self.assertTrue(блок.included)
        имена = [б.name for б in промпт.blocks]
        self.assertLess(имена.index("роль агента"), имена.index("роль шага"))

    def test_план_и_собранное_попадают_на_исполнении(self):
        задача = TaskState("з", stage=EXECUTION, plan=["шаг раз"], collected={"к": "з"})
        промпт = self.сборщик.build("вопрос", задача)
        имена = {б.name for б in промпт.included}
        self.assertIn("план", имена)
        self.assertIn("собранные данные", имена)

    def test_трейс_объясняет_каждый_блок(self):
        промпт = self.сборщик.build("вопрос")
        for блок in промпт.blocks:
            self.assertTrue(блок.why, f"блок «{блок.name}» без объяснения")

    def test_порядок_блоков_фиксирован(self):
        промпт = self.сборщик.build("вопрос")
        имена = [б.name for б in промпт.blocks]
        self.assertLess(имена.index("инварианты"), имена.index("настройки пользователя"))
        self.assertEqual(имена[-1], "вопрос пользователя")

    def test_без_задачи_берётся_политика_планирования(self):
        промпт = self.сборщик.build("вопрос")
        self.assertEqual(промпт.stage, PLANNING)

    def test_все_стадии_описаны_политикой(self):
        for стадия in (PLANNING, EXECUTION, VALIDATION, DONE):
            self.assertIn(стадия, POLICY)

    def test_шагу_сценария_рабочая_память_не_дублируется(self):
        # Иначе результат предыдущего шага уходит в запрос дважды: как явный
        # вход шага и как собранные данные задачи.
        задача = TaskState("з", stage=EXECUTION, collected={"шаг «аналитик»": "требования"})
        промпт = self.сборщик.build("вопрос", задача, step_role="РОЛЬ: backend")
        блок = [б for б in промпт.blocks if б.name == "собранные данные"][0]
        self.assertFalse(блок.included)
        self.assertIn("получает вход явно", блок.why)

    def test_вне_сценария_собранные_данные_показываются(self):
        задача = TaskState("з", stage=EXECUTION, collected={"к": "з"})
        промпт = self.сборщик.build("вопрос", задача)
        блок = [б for б in промпт.blocks if б.name == "собранные данные"][0]
        self.assertTrue(блок.included)

    def test_длинная_запись_обрезается_для_промпта(self):
        # Рабочая память хранит результат шага целиком, а в промпт идёт столько,
        # сколько туда помещается.
        задача = TaskState("з", stage=EXECUTION, collected={"шаг": "х" * 5000})
        промпт = self.сборщик.build("вопрос", задача)
        блок = [б for б in промпт.included if б.name == "собранные данные"][0]
        self.assertLess(len(блок.text), 3000)
        self.assertIn("обрезано", блок.text)


# --- проверка инвариантов -----------------------------------------------------

class ПроверкаИнвариантов(ВременнаяПамять):

    def setUp(self) -> None:
        super().setUp()
        seed_module.seed(self.память)
        self.валидатор = StateValidator(self.память.long.profile)

    def test_предложение_чужого_стека_ловится(self):
        нарушения = self.валидатор.check("Возьмём Laravel, на нём быстрее.")
        self.assertEqual(len(нарушения), 1)
        self.assertEqual(нарушения[0].code, "стек-бэкенд")

    def test_отказ_от_чужого_стека_не_считается_нарушением(self):
        чисто = self.валидатор.check(
            "Laravel здесь не подойдёт: геометрия только через сырой SQL, берём GeoDjango."
        )
        self.assertEqual(чисто, [])

    def test_упоминание_legacy_разрешено(self):
        чисто = self.валидатор.check(
            "Контроллер userpgplace.php из CodeIgniter 1 превращается в Django-вьюху."
        )
        self.assertEqual(чисто, [])

    def test_код_на_старом_стеке_ловится(self):
        нарушения = self.валидатор.check('```php\n<?php\n$this->load->model("x");\n```')
        self.assertTrue(нарушения)
        self.assertEqual(нарушения[0].where, "код")

    def test_чужая_субд_в_коде_ловится(self):
        нарушения = self.валидатор.check("```python\nDATABASES = {'ENGINE': 'mysql'}\n```")
        self.assertTrue(нарушения)

    def test_секрет_в_url_ловится(self):
        нарушения = self.валидатор.check("Дёргайте /api/export?token=abc123")
        self.assertEqual(нарушения[0].code, "секреты-в-url")

    def test_чистый_ответ_проходит(self):
        self.assertEqual(self.валидатор.check(
            "```python\nfrom django.contrib.gis.db import models\n\n"
            "class Pipe(models.Model):\n    geom = models.LineStringField(srid=3857)\n```"
        ), [])

    def test_напоминание_содержит_нарушение(self):
        нарушения = self.валидатор.check("Сделаем на Laravel.")
        напоминание = self.валидатор.reminder(нарушения)
        self.assertIn("laravel", напоминание.lower())

    def test_переход_проверяется_без_изменения_состояния(self):
        задача = TaskState("з", stage=PLANNING)
        можно, пояснение = StateValidator.check_transition(задача, DONE)
        self.assertFalse(можно)
        self.assertIn("не разрешён", пояснение)
        self.assertEqual(задача.stage, PLANNING)   # состояние не тронуто

    def test_мягкие_инварианты_не_проверяются_кодом(self):
        жёсткие = {и["код"] for и in self.память.long.profile.invariants(hard_only=True)}
        self.assertNotIn("1С-источник-истины", жёсткие)


# --- предпочтения -------------------------------------------------------------

class Предпочтения(unittest.TestCase):

    def test_умолчания_заполняют_все_поля(self):
        каркас = preferences.blank()
        for поле in preferences.FIELDS:
            self.assertIn(поле.key, каркас[поле.section])

    def test_недопустимое_значение_отклоняется(self):
        with self.assertRaises(PreferenceError):
            preferences.set_value(preferences.blank(), "формат", "длина", "как-нибудь")

    def test_неизвестное_поле_отклоняется(self):
        with self.assertRaises(PreferenceError):
            preferences.set_value(preferences.blank(), "формат", "цвет", "синий")

    def test_нормализация_чинит_испорченный_профиль(self):
        # Профиль правят руками и присылают формы: значение может оказаться чем
        # угодно, и дальше по коду оно должно быть уже корректным.
        профиль = preferences.normalize({"формат": {"длина": "ОЧЕНЬ КРАТКО"},
                                         "обращение": {"на_ты": "да"}})
        self.assertEqual(профиль["формат"]["длина"], "подробно")   # умолчание
        self.assertIs(профиль["обращение"]["на_ты"], True)

    def test_предел_длины_соответствует_выбору(self):
        для_кратко = preferences.set_value(preferences.blank(), "формат", "длина", "кратко")
        self.assertEqual(preferences.word_limit(для_кратко), 180)
        подробно = preferences.set_value(preferences.blank(), "формат", "длина", "подробно")
        self.assertEqual(preferences.word_limit(подробно), 0)

    def test_язык_кода_молчит_когда_код_не_нужен(self):
        # Иначе в промпт уходит противоречие: «кода не показывай» и «примеры на
        # Python» одновременно.
        профиль = preferences.set_value(preferences.blank(), "формат", "код", "не_нужен")
        self.assertFalse(any("Python" in с for с in preferences.describe(профиль)))

    def test_каждое_поле_умеет_попасть_в_промпт_или_молчать(self):
        профиль = preferences.normalize(preferences.blank())
        for поле in preferences.FIELDS:
            текст = поле.to_prompt(профиль[поле.section][поле.key])
            self.assertIsInstance(текст, str)


class ПроверкаПредпочтений(unittest.TestCase):

    @staticmethod
    def _профиль(**значения):
        профиль = preferences.blank()
        for путь, значение in значения.items():
            раздел, _, ключ = путь.partition("__")
            профиль = preferences.set_value(профиль, раздел, ключ, значение)
        return профиль

    def коды(self, профиль, ответ):
        return {о.code for о in PreferenceChecker(профиль).check(ответ)}

    def test_длина_считается_без_кода(self):
        # Десять строк модели — это не многословие, а ровно то, что просили.
        профиль = self._профиль(формат__длина="кратко")
        ответ = "Коротко.\n```python\n" + "x = 1\n" * 300 + "```"
        self.assertNotIn("длина", self.коды(профиль, ответ))

    def test_длинный_текст_ловится(self):
        профиль = self._профиль(формат__длина="кратко")
        self.assertIn("длина", self.коды(профиль, "слово " * 300))

    def test_допуск_не_придирается_к_паре_слов(self):
        профиль = self._профиль(формат__длина="кратко")
        self.assertNotIn("длина", self.коды(профиль, "слово " * 190))

    def test_код_запрещён_и_найден(self):
        профиль = self._профиль(формат__код="не_нужен")
        self.assertIn("код", self.коды(профиль, "Вот как:\n```python\nx=1\n```"))

    def test_отсутствие_кода_только_замечание(self):
        профиль = self._профиль(формат__код="обязательно")
        расхождения = PreferenceChecker(профиль).check("Объясню словами.")
        по_коду = [о for о in расхождения if о.code == "код"]
        self.assertTrue(по_коду)
        self.assertFalse(по_коду[0].hard, "требовать код на любой вопрос нельзя")

    def test_обращение_на_вы_при_профиле_на_ты(self):
        профиль = self._профиль(обращение__на_ты=True)
        self.assertIn("на_ты", self.коды(профиль, "Вам нужно перенести вашу таблицу."))

    def test_обращение_на_ты_при_профиле_на_вы(self):
        профиль = self._профиль(обращение__на_ты=False)
        self.assertIn("на_ты", self.коды(профиль, "Тебе нужно перенести твою таблицу."))

    def test_вы_внутри_кода_не_считается(self):
        профиль = self._профиль(обращение__на_ты=True)
        ответ = "Сделай так:\n```python\n# передай вам параметр\nf(вам=1)\n```"
        self.assertNotIn("на_ты", self.коды(профиль, ответ))

    def test_имя_требуется_только_когда_просили(self):
        без_имени = self._профиль(обращение__имя="Максим", обращение__по_имени=False)
        self.assertNotIn("имя", self.коды(без_имени, "Ответ без имени."))
        с_именем = self._профиль(обращение__имя="Максим", обращение__по_имени=True)
        self.assertIn("имя", self.коды(с_именем, "Ответ без имени."))
        self.assertNotIn("имя", self.коды(с_именем, "Максим, вот ответ."))

    def test_язык_ответа(self):
        профиль = self._профиль(формат__язык="русский")
        английский = "This is a long answer written entirely in English without any Russian."
        self.assertIn("язык", self.коды(профиль, английский))
        self.assertNotIn("язык", self.коды(профиль, "Это длинный ответ по-русски, "
                                                    "с именами вроде LineStringField."))

    def test_короткая_строка_не_считается_сменой_языка(self):
        профиль = self._профиль(формат__язык="русский")
        self.assertNotIn("язык", self.коды(профиль, "OK"))

    def test_структура_проверяется_мягко(self):
        профиль = self._профиль(формат__структура="таблицы")
        расхождения = PreferenceChecker(профиль).check("Просто текст без таблицы.")
        self.assertTrue(расхождения)
        self.assertFalse(any(о.hard for о in расхождения if о.code == "структура"))

    def test_подходящий_ответ_проходит_чисто(self):
        профиль = self._профиль(обращение__имя="Максим", обращение__по_имени=True,
                                обращение__на_ты=True, формат__длина="кратко",
                                формат__код="не_нужен", формат__структура="списки")
        ответ = ("Максим, порядок такой:\n"
                 "- выгрузи схему таблицы\n"
                 "- опиши модель\n"
                 "- прогони миграцию")
        self.assertEqual(PreferenceChecker(профиль).check(ответ), [])

    def test_напоминание_говорит_что_исправить(self):
        профиль = self._профиль(формат__длина="кратко")
        расхождения = PreferenceChecker(профиль).check("слово " * 300)
        self.assertIn("180", PreferenceChecker(профиль).reminder(расхождения))


# --- мастер настройки ---------------------------------------------------------

class МастерНастройки(unittest.TestCase):

    def test_вопросы_берутся_из_описания_полей(self):
        # Одно описание на всё приложение: добавили предпочтение — вопрос
        # появился сам, и разъехаться им негде.
        self.assertEqual(len(interview.questions()), len(preferences.FIELDS))

    def test_пустой_профиль_просит_настройки(self):
        self.assertTrue(interview.needs_setup({}))

    def test_после_мастера_настройка_не_нужна(self):
        профиль = interview.apply({}, {"формат/длина": "кратко"})
        self.assertFalse(interview.needs_setup(профиль))

    def test_пропущенный_вопрос_ничего_не_меняет(self):
        профиль = interview.apply({}, {"формат/длина": "кратко", "обращение/имя": "  "})
        self.assertEqual(профиль["формат"]["длина"], "кратко")
        self.assertEqual(preferences.normalize(профиль)["обращение"]["имя"], "")

    def test_ключ_без_раздела_тоже_понимается(self):
        профиль = interview.apply({}, {"длина": "средне"})
        self.assertEqual(профиль["формат"]["длина"], "средне")

    def test_негодный_ответ_отклоняется(self):
        with self.assertRaises(PreferenceError):
            interview.apply({}, {"формат/длина": "быстро"})

    def test_заготовка_стирает_то_чего_в_ней_нет(self):
        # Иначе человек берёт «тимлид» с именем Максим, переключается на
        # «инженер», у которой имени нет, — и остаётся Максимом.
        профиль = interview.from_template("тимлид")
        self.assertEqual(preferences.normalize(профиль)["обращение"]["имя"], "Максим")
        профиль = interview.from_template("инженер", профиль)
        self.assertEqual(preferences.normalize(профиль)["обращение"]["имя"], "")
        self.assertFalse(preferences.normalize(профиль)["обращение"]["по_имени"])

    def test_пропущенный_вопрос_мастера_по_прежнему_не_стирает(self):
        # У мастера правило обратное: пустой ответ — «оставить как есть».
        профиль = interview.from_template("тимлид")
        профиль = interview.apply(профиль, {"обращение/имя": "   "})
        self.assertEqual(preferences.normalize(профиль)["обращение"]["имя"], "Максим")

    def test_все_заготовки_корректны(self):
        for имя in interview.TEMPLATES:
            профиль = interview.from_template(имя)
            self.assertFalse(interview.needs_setup(профиль))
            self.assertTrue(профиль.get("контекст"))

    def test_заготовки_действительно_разные(self):
        сводки = {и: preferences.summary(interview.from_template(и))
                  for и in interview.TEMPLATES}
        self.assertEqual(len(set(сводки.values())), len(сводки), сводки)


# --- сценарии -----------------------------------------------------------------

class Сценарии(unittest.TestCase):

    def setUp(self) -> None:
        self.каталог = tempfile.mkdtemp()
        self.хранилище = ScenarioStore(os.path.join(self.каталог, "scenarios.json"))

    def tearDown(self) -> None:
        shutil.rmtree(self.каталог, ignore_errors=True)

    @staticmethod
    def _рабочий():
        return Scenario(
            имя="напиши фичу", триггеры=["напиши фичу"],
            шаги=[
                Step("аналитик", "собрать требования", роль="планирование", стадия=PLANNING),
                Step("backend", "написать код", роль="исполнение", стадия=EXECUTION,
                     вход=["аналитик"]),
                Step("ревьюер", "проверить", роль="исполнение", стадия=VALIDATION,
                     вход=["backend"]),
            ],
        )

    def test_корректный_сценарий_проходит(self):
        self._рабочий().validate()

    def test_маршрут_по_стадиям_проверяется(self):
        # Сценарий, который сломается на середине, должен отвалиться до первого
        # вызова модели, а не после того, как потратил токены.
        кривой = Scenario(имя="кривой", шаги=[
            Step("а", "раз", стадия=PLANNING), Step("б", "два", стадия=DONE),
        ])
        with self.assertRaises(ScenarioError):
            кривой.validate()

    def test_повтор_имён_шагов_отклоняется(self):
        двойной = Scenario(имя="двойной", шаги=[
            Step("а", "раз", стадия=PLANNING), Step("а", "два", стадия=EXECUTION),
        ])
        with self.assertRaises(ScenarioError):
            двойной.validate()

    def test_неизвестная_роль_модели_отклоняется(self):
        плохой = Scenario(имя="п", шаги=[Step("а", "раз", роль="телепатия")])
        with self.assertRaises(ScenarioError):
            плохой.validate()

    def test_сценарий_без_шагов_отклоняется(self):
        with self.assertRaises(ScenarioError):
            Scenario(имя="пустой").validate()

    def test_несколько_шагов_на_одной_стадии_разрешены(self):
        подряд = Scenario(имя="подряд", шаги=[
            Step("а", "раз", стадия=PLANNING), Step("б", "два", стадия=PLANNING),
            Step("в", "три", стадия=EXECUTION),
        ])
        подряд.validate()

    def test_триггер_ищется_в_запросе(self):
        сценарий = self._рабочий()
        self.assertTrue(сценарий.matches("Слушай, напиши фичу для отключений"))
        self.assertFalse(сценарий.matches("Как устроена схема gissys?"))

    def test_хранилище_переживает_перезапись(self):
        self.хранилище.add(self._рабочий())
        другое = ScenarioStore(self.хранилище.path)
        self.assertEqual(len(другое.all()), 1)
        self.assertEqual(другое.get("напиши фичу").шаги[0].агент, "аналитик")

    def test_совпадение_по_триггеру_из_хранилища(self):
        self.хранилище.add(self._рабочий())
        self.assertIsNotNone(self.хранилище.match("напиши фичу: подсветка участков"))
        self.assertIsNone(self.хранилище.match("что такое PostGIS?"))

    def test_удаление(self):
        self.хранилище.add(self._рабочий())
        self.assertTrue(self.хранилище.remove("напиши фичу"))
        self.assertFalse(self.хранилище.remove("напиши фичу"))

    def test_негодный_сценарий_не_сохраняется(self):
        with self.assertRaises(ScenarioError):
            self.хранилище.add(Scenario(имя="пустой"))
        self.assertEqual(self.хранилище.all(), [])

    def test_вход_шага_собирается_из_названных_источников(self):
        from agent.scenarios import ScenarioRunner
        шаг = Step("backend", "написать код", вход=["аналитик"])
        текст = ScenarioRunner._вход(шаг, "исходный запрос", {"аналитик": "требования"})
        self.assertIn("требования", текст)
        self.assertNotIn("исходный запрос", текст)
        self.assertIn("написать код", текст)

    def test_отсутствующий_вход_не_ломает_шаг(self):
        from agent.scenarios import ScenarioRunner
        шаг = Step("backend", "написать код", вход=["архитектор"])
        текст = ScenarioRunner._вход(шаг, "исходный запрос", {})
        self.assertIn("исходный запрос", текст)

    def test_длинный_вход_обрезается(self):
        from agent.scenarios import ScenarioRunner, ВХОД_ШАГА
        шаг = Step("b", "делай", вход=["a"])
        текст = ScenarioRunner._вход(шаг, "q", {"a": "х" * (ВХОД_ШАГА * 3)})
        self.assertIn("обрезано", текст)
        self.assertLess(len(текст), ВХОД_ШАГА * 2)


class _ЗаглушкаКлиента:
    """Клиент, который всегда отвечает одним и тем же — и помнит, кого звали."""

    def __init__(self, текст: str) -> None:
        self.текст = текст
        self.вызовы: list[str] = []
        self.spent = {"calls": 0, "tokens": 0, "cost": 0.0}

    def call(self, model_key, messages, **kwargs):
        from agent.llm import Reply
        self.вызовы.append(model_key)
        return Reply(text=self.текст, model_key=model_key)

    def close(self) -> None:
        pass


class ЛестницаПовторов(unittest.TestCase):
    """Из-за чего агент повторяет запрос и из-за чего меняет модель."""

    def setUp(self) -> None:
        from agent import MemoryAgent
        self.каталог = tempfile.mkdtemp()
        self.агент = MemoryAgent(base_dir=self.каталог, router_mode=OFF,
                                 model_key="groq-20b")

    def tearDown(self) -> None:
        self.агент.close()
        shutil.rmtree(self.каталог, ignore_errors=True)

    def _подменить(self, текст: str) -> _ЗаглушкаКлиента:
        заглушка = _ЗаглушкаКлиента(текст)
        self.агент.client = заглушка
        return заглушка

    def test_расхождение_с_профилем_повторяет_на_той_же_модели(self):
        # Платить за ответ вчетверо дороже потому, что он на двадцать слов
        # длиннее просимого, — плохая сделка.
        self.агент.set_preference("формат", "длина", "кратко")
        заглушка = self._подменить("слово " * 400)
        ответ = self.агент.ask("вопрос")
        self.assertTrue(ответ.deviations)
        self.assertGreater(len(заглушка.вызовы), 1, "повтора не было")
        self.assertEqual(set(заглушка.вызовы), {"groq-20b"})
        self.assertEqual(ответ.escalated_to, "")

    def test_нарушение_инварианта_поднимает_модель(self):
        заглушка = self._подменить("Возьмём Laravel, на нём быстрее.")
        ответ = self.агент.ask("вопрос")
        self.assertTrue(ответ.violations)
        self.assertGreater(len(set(заглушка.вызовы)), 1, "эскалации не было")
        self.assertEqual(заглушка.вызовы[0], "groq-20b")

    def test_подходящий_ответ_не_повторяется(self):
        self.агент.set_preference("формат", "длина", "кратко")
        заглушка = self._подменить("Перенесите таблицу миграцией Django.")
        ответ = self.агент.ask("вопрос")
        self.assertEqual(ответ.attempts, 1)
        self.assertEqual(len(заглушка.вызовы), 1)

    def test_служебный_вызов_не_трогает_память(self):
        # Шаг сценария получает на вход машинный текст из результатов предыдущих
        # шагов. Разбирать его маршрутизатором и класть в диалог нельзя: в базу
        # знаний так попадали куски ответов агентов, принятые за слова человека.
        self.агент.memory.router_mode = "авто"
        self.агент.memory.router = _ЗаглушкаМаршрутизатора(
            Routing(target="знания", key="к", value="машинный текст", confidence=0.99)
        )
        self._подменить("Готово.")
        было_знаний = len(self.агент.memory.long.knowledge.all())
        было_реплик = self.агент.memory.short.stats(self.агент.session)["messages"]

        ответ = self.агент.ask("РЕЗУЛЬТАТ ШАГА «архитектор»: …", internal=True)

        self.assertTrue(ответ.text)
        self.assertEqual(len(self.агент.memory.long.knowledge.all()), было_знаний)
        self.assertEqual(
            self.агент.memory.short.stats(self.агент.session)["messages"], было_реплик
        )

    def test_обычный_вызов_память_пополняет(self):
        self.агент.memory.router_mode = "выкл"
        self._подменить("Готово.")
        было = self.агент.memory.short.stats(self.агент.session)["messages"]
        self.агент.ask("обычный вопрос")
        self.assertEqual(
            self.агент.memory.short.stats(self.агент.session)["messages"], было + 2
        )

    def test_промежуточный_шаг_профилем_не_проверяется(self):
        self.агент.set_preference("формат", "длина", "кратко")
        заглушка = self._подменить("слово " * 400)
        ответ = self.агент.ask("вопрос", personal=False)
        self.assertEqual(ответ.deviations, [])
        self.assertEqual(len(заглушка.вызовы), 1)


class _Сценарная:
    """Клиент, отвечающий по списку заготовленных ответов."""

    def __init__(self, ответы: list[str]) -> None:
        self.ответы = list(ответы)
        self.вызовы = 0
        self.spent = {"calls": 0, "tokens": 0, "cost": 0.0}

    def call(self, model_key, messages, **kwargs):
        from agent.llm import Reply
        текст = self.ответы[min(self.вызовы, len(self.ответы) - 1)]
        self.вызовы += 1
        return Reply(text=текст, model_key=model_key)

    def close(self) -> None:
        pass


class ПаузаИПродолжение(unittest.TestCase):
    """Четыре точки останова и продолжение с того же шага.

    Сети тесты не трогают: проверяется машинерия состояния, а не ответы модели.
    """

    ИТОГ = '{"заголовок":"И","решение":"р","причина":"п"} Сделано.'

    def setUp(self) -> None:
        self.каталог = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.каталог, ignore_errors=True)

    def _агент(self, ответы: list[str], **kwargs):
        from agent import MemoryAgent
        агент = MemoryAgent(base_dir=self.каталог, router_mode=OFF, **kwargs)
        клиент = _Сценарная(ответы)
        агент.client = клиент
        агент.memory.client = клиент
        агент.memory.router.client = клиент
        агент.validator.client = клиент
        агент.заглушка = клиент
        return агент

    @staticmethod
    def _сценарий():
        return Scenario(имя="проба", триггеры=["проба"], шаги=[
            Step("аналитик", "собрать требования", роль="планирование",
                 стадия=PLANNING, вход=["запрос"], может_спросить=True),
            Step("backend", "написать код", роль="исполнение",
                 стадия=EXECUTION, вход=["аналитик"]),
        ])

    def test_шаг_спрашивает_и_сценарий_встаёт(self):
        агент = self._агент(["Часть ясна.\nНУЖНЫ СВЕДЕНИЯ: какой SRID?", self.ИТОГ])
        try:
            агент.add_scenario(self._сценарий())
            итог = агент.run_scenario("проба: слой", name="проба")
            self.assertTrue(итог.на_паузе)
            self.assertEqual(итог.причина_паузы, НЕТ_СВЕДЕНИЙ)
            self.assertEqual(итог.ожидание, ОТВЕТ)
            self.assertIn("SRID", итог.ожидание_текст)
            # Название спросившего шага должно быть в пояснении: указатель к
            # этому моменту уже стоит на следующем.
            self.assertIn("аналитик", итог.ожидание_текст)
        finally:
            агент.close()

    def test_шагу_без_разрешения_вопрос_не_засчитывается(self):
        # backend спрашивать не вправе: получив проект, он должен писать код.
        сценарий = self._сценарий()
        сценарий.шаги[0].может_спросить = False
        агент = self._агент(["Ответ.\nНУЖНЫ СВЕДЕНИЯ: а что именно?", self.ИТОГ])
        try:
            агент.add_scenario(сценарий)
            итог = агент.run_scenario("проба: слой", name="проба")
            self.assertFalse(итог.на_паузе)
        finally:
            агент.close()

    def test_продолжение_идёт_с_того_же_шага_в_новом_агенте(self):
        первый = self._агент(["Часть ясна.\nНУЖНЫ СВЕДЕНИЯ: какой SRID?"])
        try:
            первый.add_scenario(self._сценарий())
            итог = первый.run_scenario("проба: слой", name="проба")
            task_id = итог.task_id
            сделано_до = len(итог.шаги)
        finally:
            первый.close()

        # Новый агент — то же, что новый запуск процесса: всё берётся с диска.
        второй = self._агент([self.ИТОГ])
        try:
            состояние = второй.use_task(task_id)
            self.assertTrue(состояние.пауза)
            итог2 = второй.resume_scenario(task_id, ответ="SRID 3857")
            self.assertFalse(итог2.на_паузе)
            # Главное: пройденный шаг не переигрывается.
            self.assertEqual(сделано_до, 1)
            self.assertEqual(len(итог2.шаги), 1)
            self.assertEqual(второй.заглушка.вызовы, 2)   # шаг + свёртка задачи
        finally:
            второй.close()

    def test_ответ_человека_попадает_в_следующий_шаг(self):
        первый = self._агент(["Часть ясна.\nНУЖНЫ СВЕДЕНИЯ: какой SRID?"])
        try:
            первый.add_scenario(self._сценарий())
            task_id = первый.run_scenario("проба: слой", name="проба").task_id
        finally:
            первый.close()

        второй = self._агент([self.ИТОГ])
        перехвачено = []
        настоящий = второй.client.call

        def подглядеть(model_key, messages, **kwargs):
            перехвачено.append(messages[-1]["content"])
            return настоящий(model_key, messages, **kwargs)

        второй.client.call = подглядеть
        try:
            второй.use_task(task_id)
            второй.resume_scenario(task_id, ответ="SRID 3857, как у остальных слоёв")
            self.assertTrue(any("3857" in т for т in перехвачено),
                            "ответ человека не дошёл до шага")
        finally:
            второй.close()

    def test_режим_по_шагам_останавливает_на_переходе(self):
        агент = self._агент(["Требования собраны.", self.ИТОГ])
        try:
            агент.add_scenario(self._сценарий())
            итог = агент.run_scenario("проба: слой", name="проба", по_шагам=True)
            self.assertTrue(итог.на_паузе)
            self.assertEqual(итог.причина_паузы, НА_ПЕРЕХОДЕ)
            self.assertEqual(итог.ожидание, ПОДТВЕРДИТЬ)
            # Стадия при этом не сменилась: переход ещё не подтверждён.
            self.assertEqual(агент.task.stage, PLANNING)
            итог2 = агент.resume_scenario(итог.task_id)
            self.assertFalse(итог2.на_паузе)
            self.assertEqual(агент.заглушка.вызовы, 3)   # два шага и свёртка
        finally:
            агент.close()

    def test_подтверждение_перехода_выполняет_переход(self):
        # Иначе цикл снова видит несменённую стадию и просит подтвердить тот же
        # переход — и так до бесконечности. Ровно это и было.
        агент = self._агент(["Требования собраны.", self.ИТОГ])
        try:
            агент.add_scenario(self._сценарий())
            итог = агент.run_scenario("проба: слой", name="проба", по_шагам=True)
            self.assertEqual(итог.причина_паузы, НА_ПЕРЕХОДЕ)
            итог2 = агент.resume_scenario(итог.task_id, по_шагам=True)
            self.assertEqual(агент.task.stage if агент.task else DONE, DONE,
                             "переход так и не состоялся")
            self.assertFalse(итог2.на_паузе, итог2.ожидание_текст)
        finally:
            агент.close()

    def test_согласие_действует_на_один_переход(self):
        # Три шага на трёх стадиях: подтвердили первый переход — второй должен
        # снова спросить.
        сценарий = self._сценарий()
        сценарий.шаги.append(Step("ревьюер", "проверить", роль="исполнение",
                                  стадия=VALIDATION, вход=["backend"]))
        агент = self._агент(["Раз.", "Два.", "Три.", self.ИТОГ])
        try:
            агент.add_scenario(сценарий)
            итог = агент.run_scenario("проба: слой", name="проба", по_шагам=True)
            self.assertEqual(итог.причина_паузы, НА_ПЕРЕХОДЕ)
            итог2 = агент.resume_scenario(итог.task_id, по_шагам=True)
            self.assertTrue(итог2.на_паузе, "второй переход прошёл без подтверждения")
            self.assertEqual(итог2.причина_паузы, НА_ПЕРЕХОДЕ)
        finally:
            агент.close()

    def test_нарушение_инварианта_ставит_на_паузу_а_не_роняет(self):
        агент = self._агент(["Возьмём Laravel, на нём быстрее."])
        try:
            агент.add_scenario(self._сценарий())
            итог = агент.run_scenario("проба: слой", name="проба")
            self.assertTrue(итог.на_паузе)
            self.assertEqual(итог.причина_паузы, НАРУШЕН_ИНВАРИАНТ)
            self.assertEqual(итог.ожидание, РЕШЕНИЕ)
        finally:
            агент.close()

    def test_пауза_по_команде_останавливает_перед_следующим_шагом(self):
        агент = self._агент(["Требования собраны.", self.ИТОГ])
        try:
            агент.add_scenario(self._сценарий())
            # Пауза ставится из обработчика «после шага» — так же, как её
            # ставит кнопка на странице во время прогона.
            def после(результат, номер, всего):
                if номер == 1:
                    агент.pause_task()

            итог = агент.run_scenario("проба: слой", name="проба", on_result=после)
            self.assertTrue(итог.на_паузе)
            self.assertEqual(итог.причина_паузы, ПО_КОМАНДЕ)
            self.assertEqual(len(итог.шаги), 1)
            self.assertEqual(агент.заглушка.вызовы, 1)
        finally:
            агент.close()

    def test_продолжать_нечего_если_задача_не_по_сценарию(self):
        агент = self._агент([self.ИТОГ])
        try:
            from agent import AgentError
            агент.start_task("ручная", "без сценария")
            with self.assertRaises(AgentError):
                агент.resume_scenario("ручная")
        finally:
            агент.close()


class РазборВопросаШага(unittest.TestCase):
    """Маркер остановки разбирается кодом, а решение принимает модель."""

    def test_маркер_в_конце_ловится(self):
        from agent.scenarios import вопрос_шага
        self.assertEqual(
            вопрос_шага("Всё описано.\nНУЖНЫ СВЕДЕНИЯ: какой SRID у слоя?"),
            "какой SRID у слоя?")

    def test_маркер_в_разметке_ловится(self):
        from agent.scenarios import вопрос_шага
        self.assertIn("SRID", вопрос_шага("Текст.\n**НУЖНЫ СВЕДЕНИЯ:** какой SRID?"))

    def test_упоминание_в_середине_не_считается(self):
        # Иначе пересказ инструкции самой моделью останавливал бы сценарий.
        from agent.scenarios import вопрос_шага
        текст = ("Если бы не хватало данных, я бы написал НУЖНЫ СВЕДЕНИЯ: и перечислил.\n"
                 + "Но данных хватает.\n" * 6)
        self.assertEqual(вопрос_шага(текст), "")

    def test_без_маркера_пусто(self):
        from agent.scenarios import вопрос_шага
        self.assertEqual(вопрос_шага("Обычный ответ без вопросов."), "")


# --- каталог моделей ----------------------------------------------------------

class КаталогМоделей(unittest.TestCase):

    def test_у_каждой_роли_есть_модель(self):
        for роль in catalog.ROLES:
            self.assertIn(catalog.for_role(роль, offset=0), catalog.MODELS)

    def test_частые_роли_чередуют_провайдеров(self):
        модели = {catalog.for_role("маршрутизация", offset=i) for i in range(2)}
        провайдеры = {catalog.get(м).provider for м in модели}
        self.assertGreater(len(провайдеры), 1, "частая роль сидит на одном провайдере")

    def test_эскалация_поднимает_на_ступень(self):
        self.assertEqual(catalog.escalate("groq-allam7b"), "groq-20b")
        self.assertEqual(catalog.escalate("groq-20b"), "groq-120b")
        self.assertEqual(catalog.escalate("groq-120b"), "ds-pro")

    def test_с_вершины_лестницы_некуда(self):
        self.assertEqual(catalog.escalate("ds-pro"), "ds-pro")

    def test_модель_вне_лестницы_идёт_на_сильную_бесплатную(self):
        self.assertEqual(catalog.escalate("groq-qwen27b"), "groq-120b")

    def test_неизвестная_роль_даёт_ошибку(self):
        with self.assertRaises(KeyError):
            catalog.for_role("телепатия")


# --- начальное наполнение -----------------------------------------------------

class НачальноеНаполнение(ВременнаяПамять):

    def test_наполняет_пустую_память(self):
        сводка = seed_module.seed(self.память)
        self.assertGreater(сводка["знания"], 0)
        self.assertEqual(self.память.long.stats()["инварианты"], len(seed_module.INVARIANTS))

    def test_не_перезаписывает_заполненную(self):
        seed_module.seed(self.память)
        self.память.long.profile.update("обращение", "тон", "дружелюбный")
        seed_module.seed(self.память)
        self.assertEqual(
            self.память.long.profile.load()["обращение"]["тон"], "дружелюбный"
        )

    def test_жёсткие_инварианты_проверяемы(self):
        seed_module.seed(self.память)
        for инвариант in self.память.long.profile.invariants(hard_only=True):
            self.assertTrue(инвариант.get("значения"),
                            f"инвариант «{инвариант['код']}» нечем проверять")


class ЛимитыПровайдера(unittest.TestCase):
    """Минутный лимит проходит сам, суточный — нет, и путать их нельзя."""

    def test_минутный_лимит_не_считается_суточным(self):
        from agent.llm import _суточный_лимит
        self.assertFalse(_суточный_лимит(
            "Rate limit reached on tokens per minute (TPM): Limit 8000"))

    def test_суточный_лимит_узнаётся(self):
        from agent.llm import _суточный_лимит
        for сообщение in ("on tokens per day (TPD): Limit 200000",
                          "on requests per day (RPD)",
                          "daily quota exceeded"):
            self.assertTrue(_суточный_лимит(сообщение), сообщение)

    def test_суточный_лимит_не_уходит_в_повторы(self):
        # Ждать по минуте четыре раза, чтобы в конце получить ту же ошибку, —
        # это несколько минут, потраченных впустую.
        import httpx
        from agent.llm import Client, LLMError

        клиент = Client()
        попыток = {"счёт": 0}

        def ответ(запрос: httpx.Request) -> httpx.Response:
            попыток["счёт"] += 1
            return httpx.Response(429, json={"error": {
                "message": "Rate limit reached on tokens per day (TPD): Limit 200000"}})

        клиент._http = httpx.Client(transport=httpx.MockTransport(ответ))
        try:
            with self.assertRaises(LLMError) as поймано:
                клиент.call("groq-120b", [{"role": "user", "content": "привет"}])
            self.assertEqual(попыток["счёт"], 1, "суточный лимит ушёл в повторы")
            self.assertIn("завтра", str(поймано.exception))
        finally:
            клиент.close()


# --- веб-интерфейс ------------------------------------------------------------

class ВебИнтерфейс(unittest.TestCase):
    """Всё, что можно сделать из консоли, должно быть доступно и со страницы.

    Тесты идут через тестовый клиент Flask и сети не трогают: проверяются ручки
    управления, а не ответы модели.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.каталог = tempfile.mkdtemp(prefix="web-test-")
        # web.py создаёт агента при импорте, поэтому каталог памяти задаётся до
        # него — иначе тест писал бы в рабочую память проекта.
        os.environ["MEMORY_DIR"] = cls.каталог
        import importlib
        import web as модуль
        cls.web = importlib.reload(модуль)
        cls.клиент = cls.web.app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("MEMORY_DIR", None)
        shutil.rmtree(cls.каталог, ignore_errors=True)

    def состояние(self) -> dict:
        return self.клиент.get("/api/state").get_json()

    def test_страница_открывается(self):
        ответ = self.клиент.get("/")
        self.assertEqual(ответ.status_code, 200)

    def test_состояние_описывает_форму_профиля(self):
        # Из этого описания страница рисует и мастер, и поля профиля: списка
        # полей в разметке нет, иначе он разъехался бы с FIELDS.
        состояние = self.состояние()
        self.assertEqual(len(состояние["profile_fields"]), len(preferences.FIELDS))
        self.assertEqual(len(состояние["setup_questions"]), len(preferences.FIELDS))
        self.assertTrue(состояние["roles"])
        self.assertTrue(состояние["stages"])

    def test_мастер_настройки_со_страницы(self):
        ответ = self.клиент.post("/api/profile", json={
            "action": "мастер",
            "answers": {"обращение/имя": "Ольга", "обращение/по_имени": "да",
                        "формат/длина": "кратко"},
        })
        self.assertEqual(ответ.status_code, 200)
        состояние = ответ.get_json()["state"]
        self.assertIn("Ольга", состояние["profile_summary"])
        self.assertFalse(состояние["needs_setup"])

    def test_негодное_предпочтение_отклоняется_с_объяснением(self):
        ответ = self.клиент.post("/api/profile", json={
            "action": "настройка", "section": "формат", "key": "длина",
            "value": "моментально",
        })
        self.assertEqual(ответ.status_code, 400)
        self.assertIn("кратко", ответ.get_json()["error"])

    def test_свой_сценарий_заводится_и_ловится_по_триггеру(self):
        свой = {
            "имя": "проверь миграцию", "описание": "две ступени",
            "триггеры": ["проверь миграцию"],
            "шаги": [
                {"агент": "сверка", "задача": "сверить схему",
                 "роль": "планирование", "стадия": "planning", "вход": ["запрос"]},
                {"агент": "вывод", "задача": "сделать вывод",
                 "роль": "исполнение", "стадия": "execution", "вход": ["сверка"]},
            ],
        }
        ответ = self.клиент.post("/api/scenario/save", json=свой)
        self.assertEqual(ответ.status_code, 200)
        имена = {с["имя"] for с in ответ.get_json()["state"]["scenarios"]}
        self.assertIn("проверь миграцию", имена)

        совпадение = self.клиент.post(
            "/api/match", json={"query": "проверь миграцию справочника"}
        ).get_json()["matched"]
        self.assertEqual(совпадение["имя"], "проверь миграцию")

        удаление = self.клиент.post("/api/scenario/delete",
                                    json={"name": "проверь миграцию"})
        self.assertEqual(удаление.status_code, 200)

    def test_кривой_сценарий_не_сохраняется(self):
        ответ = self.клиент.post("/api/scenario/save", json={
            "имя": "кривой",
            "шаги": [{"агент": "а", "задача": "раз", "стадия": "planning"},
                     {"агент": "б", "задача": "два", "стадия": "done"}],
        })
        self.assertEqual(ответ.status_code, 400)
        # Отказ должен объяснять, что именно не сошлось: страница показывает
        # это пользователю, а не молча теряет правку.
        ошибка = ответ.get_json()["error"]
        self.assertIn("done", ошибка)
        self.assertIn("planning", ошибка)

    def test_автозапуск_сценариев_выключается(self):
        self.клиент.post("/api/settings", json={"auto_scenarios": False})
        совпадение = self.клиент.post(
            "/api/match", json={"query": "напиши фичу: подсветка участков"}
        ).get_json()["matched"]
        self.assertIsNone(совпадение, "автозапуск выключен, а сценарий предложен")
        self.клиент.post("/api/settings", json={"auto_scenarios": True})
        совпадение = self.клиент.post(
            "/api/match", json={"query": "напиши фичу: подсветка участков"}
        ).get_json()["matched"]
        self.assertIsNotNone(совпадение)

    def test_переключение_пользователя(self):
        ответ = self.клиент.post("/api/user", json={"user": "новый-человек"})
        self.assertEqual(ответ.status_code, 200)
        состояние = ответ.get_json()["state"]
        self.assertEqual(состояние["info"]["user_id"], "новый-человек")
        self.assertTrue(состояние["needs_setup"], "новому пользователю не предложили мастер")

    def test_имя_пользователя_с_побегом_отклоняется(self):
        ответ = self.клиент.post("/api/user", json={"user": "../../чужой"})
        self.assertEqual(ответ.status_code, 400)
        # Агент при этом должен остаться прежним, а не исчезнуть.
        self.assertTrue(self.состояние()["info"]["user_id"])

    def test_явная_запись_в_слой(self):
        ответ = self.клиент.post("/api/remember", json={
            "target": "знания", "key": "проба", "value": "в gisdata 37 таблиц",
        })
        self.assertEqual(ответ.status_code, 200)
        self.assertEqual(ответ.get_json()["entry"]["подслой"], "знания")

    def _подменить_клиента(self, клиент):
        """Ставит клиента во все места, где агент его держит, и возвращает прежнего.

        Мест четыре, и это выяснилось неприятным образом: первая версия
        подменяла только два, а маршрутизатор реплик держит свою ссылку — и
        «тесты без сети» тихо ходили в API, отчего набор шёл двадцать пять
        секунд вместо секунды.
        """
        прежний = self.web.agent.client
        self.web.agent.client = клиент
        self.web.agent.memory.client = клиент
        self.web.agent.memory.router.client = клиент
        self.web.agent.validator.client = клиент
        return прежний

    def _без_сети(self, текст: str = "Готово."):
        заглушка = _ЗаглушкаКлиента(текст)
        return заглушка, self._подменить_клиента(заглушка)

    def _вернуть(self, прежний) -> None:
        self._подменить_клиента(прежний)

    def _дождаться(self, предел: float = 20.0) -> dict:
        конец = time.monotonic() + предел
        while time.monotonic() < конец:
            прогон = self.клиент.get("/api/scenario/status").get_json()["run"]
            if прогон and прогон["готово"]:
                return прогон
            time.sleep(0.05)
        self.fail("сценарий не завершился за отведённое время")

    def test_сценарий_запускается_фоном_и_сразу_отдаёт_страницу(self):
        # Пять шагов идут минуту и дольше. Если держать на это время один
        # HTTP-запрос, страница молчит и отличить работу от зависания нельзя —
        # именно так первая версия и выглядела.
        заглушка, прежний = self._без_сети()
        try:
            пуск = self.клиент.post("/api/scenario",
                                    json={"name": "оцени задачу", "query": "оцени задачу"})
            self.assertEqual(пуск.status_code, 200)
            self.assertIn("run_id", пуск.get_json())
            прогон = self._дождаться()
            self.assertFalse(прогон["ошибка"], прогон["ошибка"])
            self.assertEqual(len(прогон["шаги"]), прогон["всего"])
            self.assertIsNotNone(прогон["state"], "в конце состояние памяти не отдано")
        finally:
            self._вернуть(прежний)

    def test_во_время_прогона_другие_действия_отклоняются(self):
        заглушка, прежний = self._без_сети()
        try:
            self.клиент.post("/api/scenario",
                             json={"name": "оцени задачу", "query": "оцени задачу"})
            # Агент один на процесс, и вести две задачи сразу он не может.
            коды = {
                self.клиент.post("/api/ask", json={"question": "вопрос"}).status_code,
                self.клиент.post("/api/user", json={"user": "кто-то"}).status_code,
            }
            self._дождаться()
            self.assertTrue(коды <= {409, 200},
                            f"неожиданные коды во время прогона: {коды}")
        finally:
            self._вернуть(прежний)

    def test_ошибка_прогона_не_теряется(self):
        # При синхронном запросе сбой возвращался кодом ответа. Теперь прогон
        # идёт в потоке, и ошибку надо донести до страницы отдельно.
        class Падающий(_ЗаглушкаКлиента):
            def call(self, model_key, messages, **kwargs):
                from agent.llm import LLMError
                raise LLMError("провайдер недоступен")

        прежний = self._подменить_клиента(Падающий(""))
        try:
            self.клиент.post("/api/scenario",
                             json={"name": "оцени задачу", "query": "оцени задачу"})
            прогон = self._дождаться()
            self.assertIn("недоступен", прогон["ошибка"])
        finally:
            self._вернуть(прежний)

    def test_статус_без_прогона(self):
        свежий = self.web.app.test_client()
        self.web._прогон.clear()
        self.assertIsNone(свежий.get("/api/scenario/status").get_json()["run"])

    def test_состояние_задачи_отдаётся_страницей(self):
        self.клиент.post("/api/task", json={"action": "создать", "task_id": "сост",
                                            "title": "проверка"})
        состояние = self.состояние()["task_state"]
        self.assertTrue(состояние["есть"])
        for поле in ("этап", "шаг_словами", "ожидание", "ожидание_текст", "пауза"):
            self.assertIn(поле, состояние)
        self.клиент.post("/api/task", json={"action": "отпустить"})

    def test_пауза_и_продолжение_через_страницу(self):
        self.клиент.post("/api/task", json={"action": "создать", "task_id": "пауза-веб",
                                            "title": "проверка"})
        пауза = self.клиент.post("/api/task", json={"action": "пауза"})
        self.assertEqual(пауза.status_code, 200)
        self.assertTrue(пауза.get_json()["state"]["task_state"]["пауза"])
        дальше = self.клиент.post("/api/task", json={"action": "продолжить"})
        self.assertFalse(дальше.get_json()["state"]["task_state"]["пауза"])
        self.клиент.post("/api/task", json={"action": "отпустить"})

    def test_пауза_разрешена_во_время_прогона(self):
        # В этом и смысл кнопки: остановить то, что идёт прямо сейчас. Если
        # блокировать её наравне с остальными действиями, паузы нет вовсе.
        заглушка, прежний = self._без_сети()
        try:
            self.клиент.post("/api/scenario",
                             json={"name": "оцени задачу", "query": "оцени задачу"})
            ответ = self.клиент.post("/api/task", json={"action": "пауза"})
            self._дождаться()
            self.assertIn(ответ.status_code, (200, 400),
                          "пауза во время прогона не должна отклоняться как 409")
        finally:
            self._вернуть(прежний)

    def test_продолжение_сценария_со_страницы(self):
        заглушка, прежний = self._без_сети(
            "Требования собраны.\nНУЖНЫ СВЕДЕНИЯ: какой SRID?")
        try:
            пуск = self.клиент.post(
                "/api/scenario", json={"name": "оцени задачу", "query": "оцени задачу"})
            self.assertEqual(пуск.status_code, 200)
            прогон = self._дождаться()
            self.assertTrue(прогон["на_паузе"], "сценарий не встал на вопросе шага")
            self.assertEqual(прогон["ожидание"], "ответ-пользователя")

            self._вернуть(прежний)
            заглушка2, прежний = self._без_сети("Готово.")
            task_id = self.состояние()["task_state"]["task_id"]
            продолжение = self.клиент.post(
                "/api/scenario/resume",
                json={"task_id": task_id, "answer": "SRID 3857"})
            self.assertEqual(продолжение.status_code, 200)
            итог = self._дождаться()
            self.assertFalse(итог["ошибка"], итог["ошибка"])
        finally:
            self._вернуть(прежний)

    def test_продолжение_учитывает_выбор_страницы(self):
        # «Продолжить» раньше не применяло настройки страницы вовсе: прогон
        # уходил на модель по роли, хотя в шапке выбрана другая. Заметно это
        # становилось после перезапуска сервера, когда агент о выборе человека
        # уже ничего не знал.
        # Задача нарочно несуществующая: тогда ручка отвечает отказом сразу, не
        # запуская фонового прогона, — а настройки страницы к этому моменту уже
        # применены, что и проверяется.
        ответ = self.клиент.post("/api/scenario/resume",
                                 json={"task_id": "нет-такой-задачи", "model": "ds-flash"})
        self.assertEqual(ответ.status_code, 400)
        self.assertEqual(self.web.agent.model_key, "ds-flash")
        self.web.agent.model_key = ""

    def test_запрещённый_переход_задачи_отклоняется(self):
        self.клиент.post("/api/task", json={"action": "создать", "task_id": "проба-веб",
                                            "title": "проверка"})
        ответ = self.клиент.post("/api/task", json={"action": "стадия", "stage": "done"})
        self.assertEqual(ответ.status_code, 400)
        self.клиент.post("/api/task", json={"action": "отпустить"})


# --- разметка страницы --------------------------------------------------------

class СтраницаЦела(unittest.TestCase):
    """Структурные проверки скрипта страницы.

    Появились после поломки, которую не поймал ни один прежний тест: при
    рефакторинге был снят не тот заголовок функции, и «запуститьСценарий»
    оказался объявлен ВНУТРИ «нарисоватьТрейс». Синтаксис при этом остался
    корректным — `node --check` молчал, — а обработчик кнопки падал с
    ReferenceError, и сценарий не запускался вовсе. Проверки Python-кода такого
    не видят в принципе, поэтому нужна отдельная.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import re
        разметка = pathlib.Path(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "templates", "index.html")
        ).read_text(encoding="utf-8")
        найдено = re.search(r"<script>(.*?)</script>", разметка, re.DOTALL)
        assert найдено, "в шаблоне нет блока <script>"
        cls.js = найдено.group(1)
        cls.разметка = разметка

    @staticmethod
    def _без_литералов(строка: str) -> str:
        import re
        return re.sub(r"'[^']*'|\"[^\"]*\"|`[^`]*`|//.*", "", строка)

    def _объявления(self):
        """Имена функций и глубина вложенности, на которой они объявлены."""
        import re
        глубина, итог = 0, []
        for строка in self.js.split("\n"):
            найдено = re.match(r"\s*(async\s+)?function\s+([А-Яа-яёA-Za-z_]+)", строка)
            if найдено:
                итог.append((найдено.group(2), глубина))
            без = self._без_литералов(строка)
            глубина += без.count("{") - без.count("}")
        return итог

    def test_скобки_сходятся(self):
        глубина = 0
        for строка in self.js.split("\n"):
            без = self._без_литералов(строка)
            глубина += без.count("{") - без.count("}")
        self.assertEqual(глубина, 0, "скобки в скрипте страницы не сходятся")

    def test_все_функции_объявлены_на_верхнем_уровне(self):
        вложенные = [(имя, г) for имя, г in self._объявления() if г != 0]
        self.assertEqual(вложенные, [],
                         f"функции объявлены внутри других: {вложенные}")

    def test_обработчики_видят_нужные_функции(self):
        """Всё, что зовут обработчики, должно быть объявлено на верхнем уровне."""
        объявлены = {имя for имя, г in self._объявления() if г == 0}
        обязательные = {
            "запуститьСценарий", "следитьЗаПрогоном", "продолжитьЗадачу",
            "нарисоватьСостояние", "открытьРедактор", "нарисоватьРедактор",
            "нарисоватьПрофиль", "нарисоватьМастер", "нарисоватьСценарии",
            "нарисоватьСлои", "нарисоватьТрейс", "применить", "спросить",
            "перейтиКПользователю", "правитьПрофиль", "действиеЗадачи",
            "добавить", "запрос", "экранировать",
        }
        self.assertEqual(обязательные - объявлены, set(),
                         "обработчики зовут функции, которых нет на верхнем уровне")

    def test_каждый_id_из_скрипта_есть_в_разметке(self):
        """$('имя') должно находить элемент, иначе обработчик молча не навесится."""
        import re
        имена = set(re.findall(r"\$\('([^']+)'\)", self.js))
        # Эти элементы рисуются самим скриптом, в статической разметке их нет.
        рисуемые = {"мастер-сохранить", "новый-сценарий", "сц-имя", "сц-описание",
                    "сц-триггеры", "сц-добавить", "сц-сохранить", "сц-отмена"}
        в_разметке = set(re.findall(r'id="([^"]+)"', self.разметка))
        пропавшие = имена - в_разметке - рисуемые
        self.assertEqual(пропавшие, set(), f"в разметке нет элементов: {пропавшие}")


# --- живые проверки -----------------------------------------------------------

@unittest.skipUnless(ЖИВЫЕ, "нужен ключ API; запускать с --живые")
class ЖивыеПроверки(unittest.TestCase):

    def test_маршрутизатор_отличает_вопрос_от_факта(self):
        from agent.llm import Client
        from agent.memory.router import Router
        клиент = Client()
        try:
            маршрутизатор = Router(клиент)
            вопрос = маршрутизатор.classify("А как в GeoDjango сделать индекс по геометрии?")
            факт = маршрутизатор.classify("У нас в схеме gisdata 37 таблиц")
            self.assertFalse(вопрос.wants_write, f"вопрос принят за факт: {вопрос.to_dict()}")
            self.assertTrue(факт.wants_write or факт.failed, факт.to_dict())
        finally:
            клиент.close()

    def test_агент_отвечает_и_не_нарушает_инвариантов(self):
        from agent import MemoryAgent
        каталог = tempfile.mkdtemp()
        агент = MemoryAgent(base_dir=каталог, router_mode=OFF, temperature=0.0)
        try:
            ответ = агент.ask("Какой ORM использовать для геометрии в новой системе?")
            self.assertTrue(ответ.text)
            self.assertFalse(ответ.blocked, [str(н) for н in ответ.violations])
            self.assertIn(LONG, ответ.layers())
        finally:
            агент.close()
            shutil.rmtree(каталог, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1)
