# 3D Ising HT-Series Workspace

## 0. Назначение
Репозиторий обслуживает задачу "3d-ising" из AGENTS.md: вычисление высокотемпературных рядов восприимчивости χ(K) на бесконечной SC‑решётке с ближайшими соседями. Основная цель текущей фазы — довести χ(K) до K^10–K^11 за счёт префиксного батчинга, симметрий и мостовых прунингов.

## 1. Структура
- `TASK.md` — формальная постановка (уровни L0–L9, метрики, зависимости).
- `experiments.json` — журнал раундов (JSON lines) по схеме из AGENTS.md.
- `engrams/` — роли-энграммы (Einstein, Tao, и т.д.). Используются в разделе "Форум" каждого отчёта.
- `scripts/`:
  - `idea_lab.py` — генератор TRIZ/Morph/Serendipity ракурсов (обязательный шаг перед экспериментом).
  - `q2_2_run_prefix_batches.py` — диспетчер префиксных задач: умеет запускать канонические состояния (`--state-library`) или грубые направления.
  - `q2_2_ht_susceptibility_cpp.py` — Python-драйвер C++ backend’а (`build/cluster_enum`), поддерживает `--prefix-seq`, `--prefix-weight`, `--prefix-state-id`.
  - `q2_2_prefix_state_lib.py` — генератор библиотек канонических префиксных состояний (использует 48 симметрий куба).
- `cpp/cluster_enum.cpp` и `build/cluster_enum` — DFS-поиск связных кластеров; опции `--max-edges`, `--bound`, `--prefix`.
- `artifacts/` — результаты; ключевые подкаталоги:
  - `q2_2_ht_cpp/` — JSON с counts, библиотеки префиксов (`prefix_states_len{3,4}.json`), чанки.
  - `tools/` — сохранённые трассы idea_lab.

## 2. Рабочий протокол
1. **Pre-flight**: прочитать `TASK.md`, `experiments.json`, все `engrams/`.
2. **IdeaLab**: `python3 scripts/idea_lab.py --task "<описание>" --json-out artifacts/tools/tricks_<ts>.json --text-out ...`.
3. **План**: один эксперимент = одна гипотеза с конкретной метрикой/чекером.
4. **DOE**: не более 5 факторов; для нескольких факторов — мини-план (2^k, A/B).
5. **Execution**: фиксировать seed/версии (Python 3.12.3, CUDA=none). Сохранять артефакты в `artifacts/`, считать SHA‑256.
6. **Report**: структура из AGENTS.md (`Дайджест`, `План`, `Форум`, `Исполнение`, `Результаты`, `Голоса и решение`).
7. **Log**: добавить запись в `experiments.json` по заданной схеме (env, compute, metrics, operators_trace, voices, status).

## 3. Текущий контекст Q2.2
- Получены χ(K) до K^10 (round 30) с помощью префиксного батчинга.
- Орбитальный анализ (round 32) доказал несостоятельность симметрии по направлениям.
- Созданы библиотеки канонических состояний префиксов длины 3 и 4 (rounds 33–34). Распределение орбит: len=3 → {6:4, 24:6, 48:1}; len=4 → {6:8, 24:28, 48:12}.
- Диспетчер (`q2_2_run_prefix_batches.py`) теперь принимает `--state-library` и передаёт `--prefix-state-id` драйверу; это база для state-level симметризации и последующих мостовых прунингов.

## 4. Полезные команды
```bash
# генерация канонической библиотеки префиксов
python3 scripts/q2_2_prefix_state_lib.py --prefix-length=3 --output artifacts/q2_2_ht_cpp/prefix_states_len3.json

# запуск state-level батчинга (пример)
python3 scripts/q2_2_run_prefix_batches.py \
  --max-edges=6 \
  --prefix-length=3 \
  --state-library=artifacts/q2_2_ht_cpp/prefix_states_len3.json \
  --binary=build/cluster_enum \
  --base-counts=artifacts/q2_2_ht_cpp/results_max9.json \
  --chunk-dir=artifacts/q2_2_ht_cpp/chunks_len3 \
  --output=artifacts/q2_2_ht_cpp/results_len3_state.json
```

## 5. Следующие шаги
- Встроить канонические состояния (len=3/4) в C++ backend (подстановка рёбер, bridge_prune).
- Расширить библиотеку на len=5 и определить стратегию гибридного батчинга.
- После state-level симметризации вернуться к экспериментам Q2.2 (K^11) и фиксировать прирост метрик.
