# -*- coding: utf-8 -*-
"""
idea_lab.py — минимальный фреймворк «операторов изобретательности»
для научного/инженерного агента.

Содержимое:
- IdeaGraph: простая графовая модель идеи
- Operators: набор «операторов изобретательности» (TRIZ-подобные, морфология, абдукция и т.д.)
- Метрики новизны/комбинирования (uzzi_zscore, recombination_metrics)
- IdeaEngine: двухфазный цикл дивергенция → конвергенция
- CLI: генерация JSON-трассы под текстовую задачу

Зависимости: numpy (только линал и KL-дивергенция).
Лицензия: MIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import argparse
import itertools
import json
import math
import random
from fractions import Fraction
from math import gcd as _gcd

import numpy as np

__all__ = [
    "Node",
    "Edge",
    "IdeaGraph",
    "Operators",
    "IdeaEngine",
    "uzzi_zscore",
    "recombination_metrics",
    "_toy_example",
    "generate_task_trace",
]


# -------------------------
# Базовые структуры графа
# -------------------------


@dataclass
class Node:
    id: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IdeaGraph:
    """Простая графовая модель идеи.

    nodes: словарь id -> Node
    edges: список Edge
    aspects: оси/координаты представления (масштаб, абстракция, роль, время, ресурсы ...)
    constraints: явные ограничения (могут сниматься оператором re_represent)
    goals: цели / метрики пригодности (произвольная структура)
    """

    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    aspects: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    goals: Dict[str, Any] = field(default_factory=dict)

    # --- базовые операции ---

    def add_node(self, node_id: str, **attrs) -> None:
        """Добавить узел, если его ещё нет, и обновить атрибуты."""
        if node_id in self.nodes:
            self.nodes[node_id].attrs.update(attrs)
        else:
            self.nodes[node_id] = Node(node_id, dict(attrs))

    def add_edge(self, src: str, dst: str, relation: str, **attrs) -> None:
        """Добавить ребро, при необходимости создавая узлы."""
        if src not in self.nodes:
            self.add_node(src)
        if dst not in self.nodes:
            self.add_node(dst)
        self.edges.append(Edge(src, dst, relation, dict(attrs)))

    def neighbors(self, node_id: str) -> List[str]:
        """Список соседей (без направленности)."""
        return [
            e.dst for e in self.edges if e.src == node_id
        ] + [
            e.src for e in self.edges if e.dst == node_id
        ]

    def copy(self) -> "IdeaGraph":
        """Глубокая копия графа (без shared-структур)."""
        ig = IdeaGraph()
        ig.nodes = {k: Node(v.id, dict(v.attrs)) for k, v in self.nodes.items()}
        ig.edges = [Edge(e.src, e.dst, e.relation, dict(e.attrs)) for e in self.edges]
        ig.aspects = dict(self.aspects)
        ig.constraints = dict(self.constraints)
        ig.goals = dict(self.goals)
        return ig

    # --- сериализация ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: n.attrs for nid, n in self.nodes.items()},
            "edges": [
                {"src": e.src, "dst": e.dst, "relation": e.relation, "attrs": e.attrs}
                for e in self.edges
            ],
            "aspects": dict(self.aspects),
            "constraints": dict(self.constraints),
            "goals": dict(self.goals),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "IdeaGraph":
        ig = IdeaGraph()
        for nid, attrs in data.get("nodes", {}).items():
            ig.add_node(nid, **attrs)
        for e in data.get("edges", []):
            ig.add_edge(e["src"], e["dst"], e["relation"], **e.get("attrs", {}))
        ig.aspects.update(data.get("aspects", {}))
        ig.constraints.update(data.get("constraints", {}))
        ig.goals.update(data.get("goals", {}))
        return ig

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @staticmethod
    def from_json(text: str) -> "IdeaGraph":
        data = json.loads(text)
        return IdeaGraph.from_dict(data)


# -------------------------
# Вспомогательные утилиты
# -------------------------


def _normalize(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Distribution must sum to > 0")
    return p / s


def _lcm(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    if a == 0 or b == 0:
        return 0
    return a // _gcd(a, b) * b


def _to_small_ints(vec: np.ndarray, max_den: int = 64) -> np.ndarray:
    """Преобразовать вещественный вектор в «малые» целые через рационализацию."""
    fracs = [Fraction(float(x)).limit_denominator(max_den) for x in vec]
    den_lcm = 1
    for f in fracs:
        den_lcm = _lcm(den_lcm, f.denominator) if den_lcm else f.denominator
    ints = np.array(
        [int(f.numerator * (den_lcm // f.denominator)) for f in fracs], dtype=int
    )
    # сократить по gcd
    g = 0
    for v in ints:
        g = _gcd(g, abs(int(v)))
    if g > 0:
        ints //= g
    if np.all(ints == 0):
        return ints
    # нормировка знака: сделать первый ненулевой положительным
    idxs = np.where(ints != 0)[0]
    if idxs.size:
        first = idxs[0]
        if ints[first] < 0:
            ints = -ints
    return ints


# --- вспомогательные штуки для TRIZ ---


_TRIZ_PRINCIPLES: List[str] = [
    "Segmentation",
    "Taking out",
    "Local quality",
    "Asymmetry",
    "Merging",
    "Universality",
    "Nested doll",
    "Counterweight",
    "Prior counteraction",
    "Preliminary action",
    "Prior cushioning",
    "Equipotentiality",
    "Inversion",
    "Spheroidality/Curvature",
    "Dynamics",
    "Partial/Excessive action",
    "Another dimension",
    "Mechanical vibration",
    "Periodic action",
    "Continuity of useful action",
    "Rushing through",
    "Blessing in disguise",
    "Feedback",
    "Intermediary",
    "Self-service",
    "Copying",
    "Cheap short-living",
    "Replacement of mechanical system",
    "Pneumatics/Hydraulics",
    "Flexible shells/Thin films",
    "Porous materials",
    "Color changes",
    "Homogeneity",
    "Discarding and recovering",
    "Parameter changes",
    "Phase transitions",
    "Thermal expansion",
    "Accelerated oxidation",
    "Inert atmosphere",
    "Composite materials",
]


def _canonical_triz_param(name: str) -> str:
    """Грубая нормализация описания улучшения/ухудшения к TRIZ-подобным параметрам."""
    s = name.lower().strip()
    t = s.replace(" ", "_")

    if any(k in s for k in ("energy", "power", "kwh", "efficien")):
        return "energy_efficiency"
    if any(k in s for k in ("runtime", "latency", "speed", "throughput", "time")):
        return "speed"
    if any(k in s for k in ("accuracy", "precision", "error", "fidelity")):
        return "accuracy"
    if any(k in s for k in ("weight", "mass", "load")):
        return "weight"
    if any(k in s for k in ("strength", "robust", "durab", "reliab")):
        return "strength"
    if any(k in s for k in ("memory", "ram", "footprint")):
        return "memory"
    if any(k in s for k in ("compute", "flop", "cpu", "gpu", "runtime_budget")):
        return "performance"
    if "order_depth" in t or "series_order" in t:
        return "order_depth"
    if any(k in s for k in ("novelty", "explore", "diversity")):
        return "novelty"
    if any(k in s for k in ("cost", "budget", "price")):
        return "cost"
    return t


_TRIZ_HEURISTICS: Dict[Tuple[str, str], List[str]] = {
    # инженерные
    ("energy_efficiency", "performance"): [
        "Local quality",
        "Preliminary action",
        "Inversion",
        "Segmentation",
    ],
    ("weight", "strength"): [
        "Composite materials",
        "Another dimension",
        "Asymmetry",
        "Porous materials",
    ],
    ("speed", "accuracy"): [
        "Feedback",
        "Partial/Excessive action",
        "Dynamics",
        "Intermediary",
    ],
    ("memory", "performance"): [
        "Segmentation",
        "Nested doll",
        "Parameter changes",
        "Discarding and recovering",
    ],
    ("novelty", "reliability"): [
        "Copying",
        "Blessing in disguise",
        "Feedback",
        "Partial/Excessive action",
    ],
    # специфично под generate_task_trace
    ("order_depth", "runtime"): [
        "Segmentation",
        "Another dimension",
        "Parameter changes",
        "Discarding and recovering",
    ],
}


# -------------------------
# Реализация операторов
# -------------------------


class Operators:
    """Коллекция статических операторов («операторы изобретательности»)."""

    # ---- геометрия/репрезентация ----

    @staticmethod
    def aspect_rotate(
        ig: IdeaGraph, axis: str, delta: float, *, rephrase_goal: bool = False
    ) -> Tuple[IdeaGraph, Dict[str, Any]]:
        """Повернуть идею по оси аспектов (масштаб/абстракция/...)."""
        out = ig.copy()
        prev = float(out.aspects.get(axis, 0.0))
        new_val = prev + float(delta)
        out.aspects[axis] = new_val
        log: Dict[str, Any] = {
            "op": "ASPECT_ROTATE",
            "axis": axis,
            "delta": float(delta),
            "prev": prev,
            "new": new_val,
        }
        if rephrase_goal:
            new_goals = {}
            for k, v in out.goals.items():
                if isinstance(v, (int, float)):
                    new_goals[k] = v * (1.0 + 0.1 * float(delta))
                else:
                    new_goals[k] = v
            out.goals = new_goals
            log["rephrased_goals"] = True
        return out, log

    @staticmethod
    def re_represent(
        ig: IdeaGraph,
        remove_constraints: Optional[List[str]] = None,
        decompose_chunks: Optional[List[str]] = None,
    ) -> Tuple[IdeaGraph, Dict[str, Any]]:
        """Сменить репрезентацию/снять ограничения/разукрупнить узлы."""
        out = ig.copy()
        removed: List[str] = []
        if remove_constraints:
            for k in remove_constraints:
                if k in out.constraints:
                    out.constraints.pop(k, None)
                    removed.append(k)

        decomposed: Dict[str, Any] = {}
        if decompose_chunks:
            for nid in decompose_chunks:
                node = out.nodes.get(nid)
                if node is not None and "chunks" in node.attrs:
                    chunks = node.attrs.pop("chunks")
                    for i, ch in enumerate(chunks):
                        new_id = f"{nid}_part{i+1}"
                        payload = (
                            ch if isinstance(ch, dict) else {"value": ch}
                        )
                        payload = {"from_chunk": nid, **payload}
                        out.add_node(new_id, **payload)
                        out.add_edge(nid, new_id, "part_of")
                    decomposed[nid] = chunks

        log = {
            "op": "RE_REPRESENT",
            "removed_constraints": removed,
            "decomposed": list(decomposed.keys()),
        }
        return out, log

    # ---- аналогии / бленды ----

    @staticmethod
    def analogize(
        target: IdeaGraph,
        base: IdeaGraph,
        preserve_relations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Структурная аналогия: выравниваем типы связей и предлагаем «предсказанные» ребра."""
        preserve_relations = preserve_relations or list(
            {e.relation for e in base.edges}
        )

        def index_rel(ig: IdeaGraph) -> Dict[str, List[Tuple[str, str]]]:
            idx: Dict[str, List[Tuple[str, str]]] = {}
            for e in ig.edges:
                idx.setdefault(e.relation, []).append((e.src, e.dst))
            return idx

        base_idx = index_rel(base)
        targ_idx = index_rel(target)

        predictions: List[Dict[str, Any]] = []
        seen_pred: set = set()

        for r in preserve_relations:
            b_pairs = set(base_idx.get(r, []))
            t_pairs = set(targ_idx.get(r, []))
            for (s, d) in b_pairs - t_pairs:
                # простая эвристика сопоставления по подстроке (безопасная)
                if s in target.nodes:
                    s_ = s
                else:
                    s_ = next(
                        (
                            n
                            for n in target.nodes
                            if s.lower() in n.lower() or n.lower() in s.lower()
                        ),
                        None,
                    )
                if d in target.nodes:
                    d_ = d
                else:
                    d_ = next(
                        (
                            n
                            for n in target.nodes
                            if d.lower() in n.lower() or n.lower() in d.lower()
                        ),
                        None,
                    )
                if s_ and d_ and (s_, d_, r) not in seen_pred:
                    seen_pred.add((s_, d_, r))
                    predictions.append(
                        {
                            "relation": r,
                            "src": s_,
                            "dst": d_,
                            "why": "structural_analogy",
                        }
                    )

        mapping = {r: r for r in preserve_relations}
        return {"op": "ANALOGIZE", "mapping": mapping, "predictions": predictions}

    @staticmethod
    def blend(
        space_A: IdeaGraph,
        space_B: IdeaGraph,
        alignment_hints: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[IdeaGraph, Dict[str, Any]]:
        """Концептуальный бленд: объединяем два графа с частичной унификацией узлов.

        alignment_hints: список пар (node_id_B, node_id_A), которые следует считать «одним и тем же».
        Дедуплирует ребра по (src, dst, relation, attrs_json).
        """
        out = space_A.copy()
        log: Dict[str, Any] = {
            "op": "BLEND",
            "merged_nodes": [],
            "merged_edges": 0,
        }
        mapping = dict(alignment_hints or [])

        # Слияние узлов
        for nid, node in space_B.nodes.items():
            target_id = mapping.get(nid, nid)
            if target_id in out.nodes:
                out.nodes[target_id].attrs.update(
                    {f"B::{k}": v for k, v in node.attrs.items()}
                )
                log["merged_nodes"].append((nid, target_id))
            else:
                out.add_node(target_id, **node.attrs)

        # Слияние ребер с дедупом
        existing = {
            (e.src, e.dst, e.relation, json.dumps(e.attrs, sort_keys=True))
            for e in out.edges
        }
        added = 0
        for e in space_B.edges:
            s = mapping.get(e.src, e.src)
            d = mapping.get(e.dst, e.dst)
            key = (s, d, e.relation, json.dumps(e.attrs, sort_keys=True))
            if key not in existing:
                out.add_edge(s, d, e.relation, **e.attrs)
                existing.add(key)
                added += 1
        log["merged_edges"] = added
        return out, log

    # ---- комбинаторика / морфология ----

    @staticmethod
    def morphology(
        axes: Dict[str, List[Any]],
        cca_rule: Optional[Callable[[Dict[str, Any]], bool]] = None,
        top_k: Optional[int] = None,
        score_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        shuffle: bool = False,
    ) -> Dict[str, Any]:
        """Морфологический анализ: декартово произведение осей.

        axes: ось -> список значений
        cca_rule(cfg) -> bool: фильтр совместимости (можно использовать для CCA/ограничений)
        score_fn(cfg) -> float: если задан, комбинации сортируются по убыванию скоров
        top_k: если задан, отдаём только top_k комбинаций после сортировки/перемешивания
        shuffle: если True, комбинации перемешиваются перед score_fn (для случайного поиска)
        """
        keys = list(axes.keys())
        values = [axes[k] for k in keys]

        combos: List[Dict[str, Any]] = []
        for prod in itertools.product(*values):
            candidate = dict(zip(keys, prod))
            if cca_rule is None or bool(cca_rule(candidate)):
                combos.append(candidate)

        if shuffle:
            random.shuffle(combos)

        if score_fn is not None:
            scored = [(c, float(score_fn(c))) for c in combos]
            scored.sort(key=lambda x: x[1], reverse=True)
            combos = [c for c, _ in scored]

        if top_k is not None:
            combos = combos[: int(top_k)]

        return {"op": "MORPHOLOGY", "count": len(combos), "candidates": combos}

    # ---- TRIZ-подобная работа с противоречиями ----

    @staticmethod
    def triz_contradiction(improve: str, worsen: str) -> Dict[str, Any]:
        """Подбор TRIZ-принципов для пары (улучшаем, не хотим ухудшить)."""
        ci = _canonical_triz_param(improve)
        cw = _canonical_triz_param(worsen)
        key = (ci, cw)
        if key not in _TRIZ_HEURISTICS:
            # попробуем наоборот (на случай перепутанной формулировки)
            key = (cw, ci)
        if key in _TRIZ_HEURISTICS:
            picks = list(_TRIZ_HEURISTICS[key])
        else:
            # резерв: случайная выборка подмножеств из 40 принципов
            picks = random.sample(_TRIZ_PRINCIPLES, k=5)

        design_moves = [
            f"Apply principle '{p}' to improve '{improve}' without degrading '{worsen}'."
            for p in picks
        ]
        return {
            "op": "TRIZ_CONTRADICTION",
            "improve": improve,
            "worsen": worsen,
            "principles": picks,
            "design_moves": design_moves,
        }

    # ---- размерностный анализ ----

    @staticmethod
    def dimensionalize(
        dim_matrix: np.ndarray, var_names: List[str]
    ) -> Dict[str, Any]:
        """Buckingham Π: базис ядра A = dim_matrix^T (размерности x переменные).

        dim_matrix: shape = (n_vars, n_dims), каждая строка — экспоненты для [M, L, T, ...]
        var_names: имена переменных, длина n_vars.
        """
        if dim_matrix.ndim != 2:
            raise ValueError("dim_matrix must be 2D (n_vars x n_dims)")
        if dim_matrix.shape[0] != len(var_names):
            raise ValueError("dim_matrix rows must match len(var_names)")

        # Нам нужно ядро A^T: A = dim_matrix^T имеет форму (n_dims x n_vars)
        A = dim_matrix.T.astype(float)  # n_dims x n_vars
        U, S, Vt = np.linalg.svd(A, full_matrices=True)
        tol = 1e-10
        rank = int((S > tol).sum())
        V = Vt.T  # n_vars x n_vars
        null = V[:, rank:]  # n_vars x (n_vars - rank)

        groups: List[Dict[str, int]] = []
        for i in range(null.shape[1]):
            coeffs = null[:, i]
            if np.all(np.abs(coeffs) < 1e-12):
                continue
            ints = _to_small_ints(coeffs)
            group = {
                var_names[j]: int(ints[j])
                for j in range(len(var_names))
                if ints[j] != 0
            }
            if group and group not in groups:
                groups.append(group)
        return {"op": "DIMENSIONALIZE", "pi_groups": groups}

    # ---- симметрии / экстремальные режимы ----

    @staticmethod
    def invariant_symmetry(
        candidate_symmetries: List[str], action: str = "enforce"
    ) -> Dict[str, Any]:
        """Фиксировать или нарушать симметрию (текстовая заглушка)."""
        if action not in {"enforce", "break"}:
            raise ValueError("action must be 'enforce' or 'break'")
        consequences = []
        for s in candidate_symmetries:
            if action == "enforce":
                consequences.append(
                    f"Conservation law implied by symmetry '{s}' (Noether-style reasoning)."
                )
            else:
                consequences.append(
                    f"Potential new effect by breaking symmetry '{s}' (evaluate measurables)."
                )
        return {"op": "INVARIANT_SYMMETRY", "action": action, "consequences": consequences}

    @staticmethod
    def extremize(
        ig: IdeaGraph, param: str, limit: str = "zero"
    ) -> Dict[str, Any]:
        """Экстремальные случаи параметров (→0 или →∞) — выявление инвариантов и ломких связей."""
        if limit not in {"zero", "inf"}:
            raise ValueError("limit must be 'zero' or 'inf'")
        failures: List[Tuple[str, str, str]] = []
        for e in ig.edges:
            if param in e.attrs.get("depends_on", []):
                failures.append((e.src, e.dst, e.relation))
        invariants = [
            n
            for n in ig.nodes
            if param not in ig.nodes[n].attrs.get("depends_on", [])
        ]
        return {
            "op": "EXTREMIZE",
            "param": param,
            "limit": limit,
            "invariants": invariants,
            "failure_modes": failures,
        }

    # ---- логико-вероятностные операторы ----

    @staticmethod
    def abduct(
        observation: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Абдукция: ранжируем гипотезы по апостериорной вероятности."""
        ranked: List[Tuple[str, float]] = []
        total = 0.0
        n = max(1, len(candidates))
        for h in candidates:
            prior = float(h.get("prior", 1.0 / n))
            like = float(h.get("likelihood", 0.5))
            score = prior * like
            ranked.append((h.get("name", "hyp"), score))
            total += score
        ranked.sort(key=lambda x: x[1], reverse=True)
        posterior = [
            (name, (s / total) if total > 0 else 0.0) for name, s in ranked
        ]
        return {"op": "ABDUCT", "observation": observation, "posterior": posterior}

    @staticmethod
    def thought_experiment(
        assumptions: List[str],
        scenario: str,
        hypotheses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Мысленный эксперимент: помечает гипотезы, конфликтующие с набором предпосылок."""
        hypotheses = hypotheses or []
        refuted: List[str] = []
        survivors = list(hypotheses)
        for h in list(hypotheses):
            for a in assumptions:
                if f"not({a})" in h or f"violates({a})" in h:
                    refuted.append(h)
                    if h in survivors:
                        survivors.remove(h)
                    break
        return {
            "op": "THOUGHT_EXPERIMENT",
            "scenario": scenario,
            "refuted": refuted,
            "survivors": survivors,
        }

    # ---- перспективы / точки зрения ----

    @staticmethod
    def perspective_swap(
        ig: IdeaGraph, role_A: str, role_B: str
    ) -> Tuple[IdeaGraph, Dict[str, Any]]:
        """Смена ролей/перспектив (фиксируем в aspects["role"])."""
        out = ig.copy()
        out.aspects["role"] = {"prev": role_A, "now": role_B}
        log = {"op": "PERSPECTIVE_SWAP", "from": role_A, "to": role_B}
        return out, log

    # ---- управляемая случайность / удивление ----

    @staticmethod
    def serendipity_hook(
        sources: List[str],
        k: int = 5,
        scorer: Optional[Callable[[str], Tuple[float, float, float]]] = None,
    ) -> Dict[str, Any]:
        """Управляемая случайность: выбираем k источников и оцениваем U/I/V."""
        k = max(0, min(k, len(sources)))
        if k == 0:
            picks: List[str] = []
        else:
            picks = random.sample(sources, k)
        results = []
        for s in picks:
            if scorer is None:
                U, I, V = random.random(), random.random(), random.random()
            else:
                U, I, V = scorer(s)
            results.append(
                {
                    "source": s,
                    "U": U,
                    "I": I,
                    "V": V,
                    "UIV": (U + I + V) / 3.0,
                }
            )
        results.sort(key=lambda x: x["UIV"], reverse=True)
        return {"op": "SERENDIPITY_HOOK", "ranked": results}

    @staticmethod
    def evaluate_surprise(
        prior: np.ndarray, posterior: np.ndarray
    ) -> Dict[str, Any]:
        """Баесовское удивление S = KL(posterior || prior) для категориального распределения."""
        p = _normalize(np.array(prior, dtype=float))
        q = _normalize(np.array(posterior, dtype=float))
        eps = 1e-12
        S = float(np.sum(q * (np.log(q + eps) - np.log(p + eps))))
        return {
            "op": "EVALUATE_SURPRISE",
            "surprise_nats": S,
            "surprise_bits": S / math.log(2.0),
        }


# -------------------------
# Метрики комбинирования/новизны
# -------------------------


def uzzi_zscore(
    pair_counts: Dict[Tuple[str, str], int],
    marginals: Dict[str, int],
    pair: Tuple[str, str],
) -> float:
    """Приближенный z-score атипичности пары (Uzzi et al.-style)."""
    a, b = pair
    if a > b:
        a, b = b, a
    N = max(1, sum(marginals.values()))
    f_i = marginals.get(a, 0)
    f_j = marginals.get(b, 0)
    E = (f_i * f_j) / max(1, N)
    obs = pair_counts.get((a, b), pair_counts.get((b, a), 0))
    if E <= 0:
        return 0.0
    return (obs - E) / math.sqrt(E)


def recombination_metrics(hist_blocks: List[List[str]]) -> Dict[str, Any]:
    """Оценивает долю новых комбинаций и темп 'explore' по истории наборов блоков."""
    seen_pairs = set()
    new_pairs = 0
    total_pairs = 0
    for blocks in hist_blocks:
        blocks = sorted(set(blocks))
        for a, b in itertools.combinations(blocks, 2):
            total_pairs += 1
            if (a, b) not in seen_pairs:
                new_pairs += 1
                seen_pairs.add((a, b))
    share = new_pairs / total_pairs if total_pairs else 0.0
    explore_rate = share  # простая эвристика
    return {"new_combo_share": share, "explore_rate": explore_rate}


# -------------------------
# Оркестратор
# -------------------------


class IdeaEngine:
    """Двухфазный цикл: дивергенция → конвергенция."""

    def __init__(
        self,
        divergence_ops: Optional[List[Callable]] = None,
        convergence_ops: Optional[List[Callable]] = None,
        random_seed: Optional[int] = None,
    ):
        self.divergence_ops = divergence_ops or [
            Operators.morphology,
            Operators.triz_contradiction,
            Operators.analogize,
            Operators.blend,
            Operators.serendipity_hook,
            Operators.abduct,
        ]
        self.convergence_ops = convergence_ops or [
            Operators.re_represent,
            Operators.aspect_rotate,
            Operators.dimensionalize,
            Operators.invariant_symmetry,
            Operators.extremize,
            Operators.thought_experiment,
            Operators.evaluate_surprise,
        ]
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def run_divergence(self, ig: IdeaGraph, **kwargs) -> Tuple[IdeaGraph, List[Dict[str, Any]]]:
        """Запускает набор дивергентных операторов."""
        logs: List[Dict[str, Any]] = []
        ig_out = ig.copy()

        if "morph_axes" in kwargs:
            logs.append(
                Operators.morphology(
                    kwargs["morph_axes"],
                    kwargs.get("cca_rule"),
                    kwargs.get("top_k"),
                    kwargs.get("score_fn"),
                    kwargs.get("shuffle", False),
                )
            )

        if "triz" in kwargs:
            imp, wor = kwargs["triz"]
            logs.append(Operators.triz_contradiction(imp, wor))

        if "analogize" in kwargs:
            targ, base, pres = kwargs["analogize"]
            logs.append(Operators.analogize(targ, base, pres))

        if "blend" in kwargs:
            A, B, hints = kwargs["blend"]
            ig_out, b_log = Operators.blend(A, B, hints)
            logs.append(b_log)

        if "serendipity_sources" in kwargs:
            logs.append(
                Operators.serendipity_hook(
                    kwargs["serendipity_sources"], kwargs.get("k", 5)
                )
            )

        if "abduct" in kwargs:
            obs, cands = kwargs["abduct"]
            logs.append(Operators.abduct(obs, cands))

        return ig_out, logs

    def run_convergence(self, ig: IdeaGraph, **kwargs) -> Tuple[IdeaGraph, List[Dict[str, Any]]]:
        """Запускает набор конвергентных операторов."""
        logs: List[Dict[str, Any]] = []
        ig_out = ig.copy()

        if "aspect" in kwargs:
            axis, delta = kwargs["aspect"]
            ig_out, log = Operators.aspect_rotate(
                ig_out, axis, float(delta), rephrase_goal=True
            )
            logs.append(log)

        if "re_represent" in kwargs:
            conf = kwargs["re_represent"]
            ig_out, log = Operators.re_represent(
                ig_out, conf.get("remove"), conf.get("decompose")
            )
            logs.append(log)

        if "dimensionalize" in kwargs:
            dimM, names = kwargs["dimensionalize"]
            logs.append(Operators.dimensionalize(dimM, names))

        if "invariant" in kwargs:
            syms, act = kwargs["invariant"]
            logs.append(Operators.invariant_symmetry(syms, act))

        if "extreme" in kwargs:
            param, lim = kwargs["extreme"]
            logs.append(Operators.extremize(ig_out, param, lim))

        if "thought" in kwargs:
            assump, scen, hyps = kwargs["thought"]
            logs.append(Operators.thought_experiment(assump, scen, hyps))

        if "surprise" in kwargs:
            prior, post = kwargs["surprise"]
            logs.append(Operators.evaluate_surprise(prior, post))

        return ig_out, logs


# -------------------------
# Примеры использования
# -------------------------


def _toy_example() -> Dict[str, Any]:
    """Демонстрация минимальной трассы на игрушечном примере (охлаждение ЦОДов)."""
    # Строим целевой граф (data center cooling)
    target = IdeaGraph(
        goals={"energy_use_kwh": 1000, "uptime": 0.999},
        constraints={"no_hardware_change": False},
    )
    for n in ["Rack", "HotAir", "ColdAir", "Vent", "HeatSink"]:
        target.add_node(n)
    target.add_edge("Rack", "HotAir", "produces", depends_on=["load"])
    target.add_edge("ColdAir", "Rack", "cools", depends_on=["flow"])
    target.add_edge("Vent", "HotAir", "extracts", depends_on=["flow"])

    # Базовый домен: "термитник" (пассивация + конвекция)
    base = IdeaGraph()
    for n in ["Nest", "WarmAir", "CoolAir", "Shafts", "Soil"]:
        base.add_node(n)
    base.add_edge("Nest", "WarmAir", "produces")
    base.add_edge("CoolAir", "Nest", "cools")
    base.add_edge("Shafts", "WarmAir", "extracts")

    engine = IdeaEngine(random_seed=42)

    # Дивергенция
    morph_axes = {
        "airflow": ["forced", "passive"],
        "heat_exchange": ["plate", "phase_change", "sorption"],
        "layout": ["open_hot_cold", "chimneys", "porous_panels"],
        "control": ["static", "feedback"],
    }
    ig_after_div, div_logs = engine.run_divergence(
        target,
        morph_axes=morph_axes,
        cca_rule=lambda c: not (
            c["airflow"] == "forced" and c["layout"] == "chimneys"
        ),
        top_k=6,
        triz=("energy_efficiency", "performance"),
        analogize=(target, base, ["produces", "cools", "extracts"]),
        blend=(target, base, [("HotAir", "WarmAir"), ("ColdAir", "CoolAir")]),
        serendipity_sources=[
            "paper: termite mounds",
            "paper: radiative cooling",
            "paper: MOF sorbents",
            "paper: heat pipes",
            "paper: phase-change materials",
        ],
        k=3,
    )

    # Конвергенция
    dimM = np.array(
        [
            #   M  L  T
            [0, 1, -1],  # velocity v: L / T
            [1, 2, -3],  # power P: M L^2 / T^3
            [0, 1, 0],  # length L: L
        ],
        dtype=float,
    )
    ig_after_conv, conv_logs = engine.run_convergence(
        ig_after_div,
        aspect=("abstraction", +1.0),
        re_represent={"remove": ["no_hardware_change"], "decompose": []},
        dimensionalize=(dimM, ["v", "P", "L"]),
        invariant=(["time_translation"], "enforce"),
        extreme=("flow", "zero"),
        thought=(
            ["adiabatic"],
            "no_external_cooling",
            ["violates(adiabatic):radiative_cooling", "passive_convection"],
        ),
        surprise=(
            np.array([0.4, 0.3, 0.3]),
            np.array([0.2, 0.6, 0.2]),
        ),
    )

    # Метрики комбинирования
    recomb = recombination_metrics(
        [
            ["forced", "plate", "open_hot_cold"],
            ["passive", "sorption", "porous_panels"],
            ["passive", "phase_change", "chimneys"],
        ]
    )
    uzzi = uzzi_zscore(
        pair_counts={("passive", "chimneys"): 1, ("forced", "plate"): 5},
        marginals={"passive": 10, "chimneys": 2, "forced": 20, "plate": 8},
        pair=("passive", "chimneys"),
    )

    return {
        "divergence": div_logs,
        "convergence": conv_logs,
        "recombination": recomb,
        "uzzi_z": uzzi,
        "graph_after": ig_after_conv.to_json(),
    }


def generate_task_trace(
    task_description: str, random_seed: int = 42
) -> Dict[str, Any]:
    """Сборка трассы операторов под конкретную текстовую формулировку задачи."""
    target = IdeaGraph(
        goals={
            "task": task_description,
            "order_target": "K^10",
            "runtime_budget_min": 10,
        },
        constraints={"only_python": True, "max_patch": "2x3x3"},
    )
    target.add_node("Task", description=task_description)
    target.add_node("Enumerators", role="algorithm", focus="cluster_series")
    target.add_node("Series", role="artifact", metric="χ(K)")
    target.add_node("Validation", role="check", method="finite_patch")
    target.add_edge("Task", "Enumerators", "requires")
    target.add_edge("Enumerators", "Series", "produces")
    target.add_edge("Series", "Validation", "checked_by")

    base = IdeaGraph()
    base.add_node("SAW", domain="self_avoiding_walks")
    base.add_node("Counting", domain="transfer_matrix")
    base.add_node("Benchmarks", domain="2D_series")
    base.add_edge("SAW", "Counting", "requires")
    base.add_edge("Counting", "Benchmarks", "checked_by")
    base.add_edge("SAW", "Benchmarks", "produces")

    engine = IdeaEngine(random_seed=random_seed)

    morph_axes = {
        "lattice_patch": ["2x3x3", "3x3x3", "2x3x4"],
        "state_repr": ["python_dfs", "frontier_dp", "polymer_basis"],
        "pruning": ["none", "connectivity", "symmetry_orbits"],
        "validation": ["small_lattice", "HT_series_crosscheck", "autocheck"],
        "acceleration": ["single_core", "multiprocessing", "gpu_like"],
    }
    ig_after_div, divergence = engine.run_divergence(
        target,
        morph_axes=morph_axes,
        cca_rule=lambda cfg: not (
            cfg["lattice_patch"] == "3x3x3" and cfg["state_repr"] == "python_dfs"
        ),
        top_k=8,
        triz=("order_depth", "runtime"),
        analogize=(target, base, ["requires", "produces", "checked_by"]),
        blend=(
            target,
            base,
            [("Enumerators", "Counting"), ("Validation", "Benchmarks")],
        ),
        serendipity_sources=[
            "paper: 3D Ising HT series",
            "paper: SAW frontier DP",
            "code: Jensen transfer matrices",
            "paper: diagrammatic Monte Carlo",
            "paper: graph homology pruning",
        ],
        k=4,
        abduct=(
            "runtime_explosion",
            [
                {"name": "lack_pruning", "prior": 0.4, "likelihood": 0.7},
                {"name": "python_overhead", "prior": 0.3, "likelihood": 0.5},
                {"name": "memory_pressure", "prior": 0.3, "likelihood": 0.6},
            ],
        ),
    )

    ig_after_conv, convergence = engine.run_convergence(
        ig_after_div,
        aspect=("abstraction", +0.5),
        re_represent={"remove": ["only_python"], "decompose": []},
        invariant=(["translation", "parity"], "enforce"),
        extreme=("max_edges", "inf"),
        thought=(
            ["connected_only"],
            "wraparound_edges",
            ["cycle_basis_filter", "bridge_prune"],
        ),
        surprise=(
            np.array([0.5, 0.3, 0.2]),
            np.array([0.2, 0.5, 0.3]),
        ),
    )

    recomb = recombination_metrics(
        [
            ["frontier_dp", "pbc_wrap", "symmetry_orbits"],
            ["dfs_walks", "loop_hash", "autocheck"],
            ["frontier_dp", "polymer_basis", "reweighting"],
        ]
    )
    uzzi = uzzi_zscore(
        pair_counts={
            ("frontier_dp", "symmetry_orbits"): 2,
            ("dfs_walks", "loop_hash"): 5,
        },
        marginals={
            "frontier_dp": 10,
            "symmetry_orbits": 2,
            "dfs_walks": 15,
            "loop_hash": 8,
        },
        pair=("frontier_dp", "symmetry_orbits"),
    )

    return {
        "task": task_description,
        "divergence": divergence,
        "convergence": convergence,
        "recombination": recomb,
        "uzzi_z": uzzi,
        "graph_after": ig_after_conv.to_json(),
    }


# -------------------------
# CLI
# -------------------------


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IdeaLab operator trace generator"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Краткое текстовое описание эксперимента/вопроса",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Путь для сохранения JSON-трассы",
    )
    parser.add_argument(
        "--text-out",
        type=Path,
        help="Путь для сохранения текстового представления (тот же JSON).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Сид генератора случайностей для IdeaEngine",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    trace = generate_task_trace(args.task, args.seed)
    formatted = json.dumps(trace, indent=2, ensure_ascii=False)
    print(formatted)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(formatted + "\n", encoding="utf-8")
    if args.text_out:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
