# Эмерджентный движок (MVP)

Минимальный движок для обнаружения эмерджентности в динамике спиновых систем (заточен под 3D Изинг 3×3×3). Реализовано на C++20 без внешних зависимостей: симулятор, coarse-graining, информ. метрики и CLI с JSON‑выводом.

## Сборка
```bash
g++ -std=c++20 -O3 cpp/emergence_engine.cpp cpp/emergence_cli.cpp -o build/emergence_cli
```

## Запуск (пример)
```bash
./build/emergence_cli --dims 3x3x3 --beta 0.3 --steps 8 --trajectories 4 --burn 50
```
Вывод — компактный JSON с микрометриками и набором макро‑карт. Полный набор флагов: `--help`.
Новый флаг для выбора набора макросов:
```bash
--macro-set baseline|full   # baseline (по умолчанию): magnetization + layer_z + block_majority; full: все макро (добавляет energy, domain_walls)
```

## Что внутри
- **Sampler**: Метрополис для куба `dims`, параметры `beta, J, h`, PBC по умолчанию. Отдаёт траектории `steps+1` состояний (после `burn_in`), с интервалом `sample_interval`.
- **MacroMap** (готовые):
  - `magnetization` — суммарный спин.
  - `energy` — округлённая энергия решётки.
  - `block_majority` — знаки сумм в блоках `2×2×2` (обрезает края).
  - `layer_z` — магнетизации слоёв вдоль z.
  - `domain_walls` — число доменных стенок (соседи с разными спинами, без двойного счёта).
- **Partition**: по умолчанию разделение по оси z пополам (AxisHalf). Для макро — пополам по компонентам.

## Метрики
Все логарифмы — натуральные.
- `entropy` H(X_t)
- `mi` I(X_t; X_{t+1})
- `ei` Effective Information с равномерным p(X_t) (MI с переоценкой по уникальным X_t)
- `phi` = MI(whole) − MI(partA) − MI(partB)
- `synergy` ≈ I([A,B]; future) − I(A; future) − I(B; future) (interaction-information стиль)
- `ppmi` — MI с лагом `ppmi_lag` (по умолчанию 5)
- `emergence_score` для макро: (EI_macro−EI_micro) + (synergy_macro−synergy_micro) + (phi_macro−phi_micro)

## Рекомендованное coarse‑graining (на основе прогонов β∈[0.1,0.30])
- Итоговая сводка: `artifacts/emergence/summary_beta_macros.json` (layer_z, block_majority, magnetization и др.).
- Наблюдение: `layer_z` устойчиво лидер или наравне с `block_majority`, заметно превосходит магнетизацию в HT/около критики; при β>0.3 различия стираются.
- Бейзлайн: использовать `layer_z` как основную макрокарту для расчётов χ(K) и связанных HT/critical проверок; `block_majority` — дополнительная опция.
- CLI по умолчанию (macro-set=baseline) считает макро: magnetization, layer_z, block_majority. Для полного набора (добавить energy/domain_walls) используйте `--macro-set full`.

## Интеграция с LLM‑агентом
- CLI выводит JSON, его можно читать напрямую.
- При необходимости можно добавлять свои макрокарты в `cpp/emergence_cli.cpp` либо вызывать `EmergenceAnalyzer` из другого C++ кода, передавая свои `MacroSpec`.
