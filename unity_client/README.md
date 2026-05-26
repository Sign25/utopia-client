# unity_client — Слой 2 (графический клиент)

**Статус:** подготовительная фаза (Этап 4 формально стартует после shipping Body Migration — дорожки 3a Этапа 3).

Графический клиент проекта NeuroCore/Утопия на Unity 6.3 LTS. Сосед-папка `utopia_client/` (Слой 1, Python headless) живёт в этом же репо `Sign25/utopia-client`. Один installer, один dist, две части.

## Что здесь будет

Подготовительная фаза (не зависит от embodied API):

- Рендер мира — terrain из Perlin noise (тот же seed что `environment/world.py`), 4 биома по `BIOME_MODIFIERS`, флора по биомам
- Камера TPS, навигация
- UI каркас (HUD, login panel, debug overlay)
- Сетевой слой — WebSocket к VPS-брокеру `divisci.com` (`/ws/feed` для AOI snapshot Мира, REST auth flow)
- Шейдеры/материалы
- Сборка pipeline (IL2CPP, Windows Standalone)

Основная фаза (после shipping 3a — embodied API стабилизирован):

- Рендер реального Зодчего по `visual_params` (geo+chroma+dyn)
- Биохимия Z7 в UI (8 веществ как progress-bars, mental_break как badge)
- Полное подключение к embodied API через контракт мир↔организм (дорожка 3c)
- Связь Unity ↔ Слой 1 (`utopia_client`) на той же машине — локальный IPC или общий VPS

## Tech stack

- **Unity 6.3 LTS** — поддержка до декабря 2027
- **URP** (Universal Render Pipeline)
- **C# / .NET Standard 2.1** через **IL2CPP** для релиза
- **Целевая платформа:** Windows 10/11 x64 (старт), macOS/Linux позже
- **DPAPI** (`ProtectedData`) для хранения JWT
- WS: `System.Net.WebSockets.ClientWebSocket` (встроенный .NET)
- REST: `HttpClient` + `System.Text.Json`

Полная спецификация — `DiviSci/NeuroCore/docs/UNITY_CLIENT_TECH_STACK.md`.

## Архитектура клиента

Три слоя на cheef-PC, **никогда не дублирующие друг друга**:

1. **P40-сервер (Ubuntu)** — несёт **только мир** после Body Migration (физика, ресурсы, дикая фауна, погода)
2. **Слой 1 — `utopia_client/`** (Python headless, эта же машина) — несёт **embodied организм**: мозг, биохимия Z7, ткани, NEAT, память
3. **Слой 2 — `unity_client/`** (этот каталог) — **только графика и UI**

## Контекст для разработчиков

Полный onboarding и план первой недели — в `DiviSci/NeuroCore/docs/UNITY_CLIENT_HANDBOOK.md`. Содержит:

- Roadmap-фазу проекта
- Команду (Шеф / Фрай / Хьюберт / Бендер)
- Архитектуру и envelope-протокол
- Два закона расчётов (φ и Фибоначчи)
- Минимальный план первой недели

## Структура

```
unity_client/
├── Assets/              # C#-скрипты, шейдеры, материалы, prefabs
│   ├── Scripts/
│   │   ├── Net/         # WS-клиенты, REST API
│   │   ├── World/       # Terrain, Biome, AOI
│   │   ├── Organisms/   # Рендер Зодчих, биохимия в UI
│   │   └── UI/          # HUD, login, settings
│   ├── Shaders/         # HLSL (R-cell облик)
│   ├── Materials/
│   ├── Prefabs/
│   └── Tests/           # Unity Test Framework
├── ProjectSettings/     # Версии Unity, render pipeline, build config
├── Packages/            # UPM manifest + lock
└── README.md            # этот файл
```

## Запуск

После того как Unity 6.3 LTS установлен через Unity Hub:

1. Откройте Unity Hub → Open → выберите эту папку (`unity_client/`).
2. Hub попросит подтвердить версию Editor — выберите 6.3 LTS.
3. Editor откроет сцену по умолчанию.

## Сборка

```
Unity Editor → File → Build Settings → Windows Standalone → Player Settings → Scripting Backend = IL2CPP → Build
```

Релизный билд складывается в `Builds/` (gitignored).

## Зависимость от Слоя 1

В **подготовительной фазе** Unity-клиент работает **независимо** от `utopia_client/`: подписывается на `/ws/feed` напрямую к VPS, видит мир без своих организмов, Зодчий показан как capsule-заглушка.

В **основной фазе** (после Body Migration) Unity тянет embodied API из контракта 3c — `visual_params`, биохимию, mental_break. Связь с локальным `utopia_client.py` — через named pipe или общий VPS-канал; точный механизм определяется в рамках 3c.

## История

- **26.05.2026** — создана пустая структура `unity_client/` в репо `Sign25/utopia-client`. .gitignore (Unity 6.x). README с обзором двух фаз и tech stack. Unity Editor 6.3 LTS — установка в процессе.
