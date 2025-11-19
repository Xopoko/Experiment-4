# -*- coding: utf-8 -*-
"""
idea_lab.py — минимальный фреймворк «операторов изобретательности»
для научного агента.

Включает:
- Структуры данных (IdeaGraph)
- Набор операторов (aspect_rotate, re_represent, analogize, blend, morphology,
  triz_contradiction, dimensionalize, invariant_symmetry, extremize, abduct,
  thought_experiment, perspective_swap, serendipity_hook, evaluate_surprise)
- Оркестратор (IdeaEngine) с фазами дивергенции/конвергенции
- Метрики (uzzi_zscore, recombination_metrics)

Зависимости: numpy (только для линал/дивергенции КЛ).
Автор: сгенерировано GPT-5 Pro для интеграции в R&D-агента.
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

import numpy as np


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
    aspects: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    goals: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node_id: str, **attrs):
        self.nodes.setdefault(node_id, Node(node_id, attrs))

    def add_edge(self, src: str, dst: str, relation: str, **attrs):
        if src not in self.nodes:
            self.add_node(src)
        if dst not in self.nodes:
            self.add_node(dst)
        self.edges.append(Edge(src, dst, relation, attrs))

    def neighbors(self, node_id: str) -> List[str]:
        return [e.dst for e in self.edges if e.src == node_id] + [e.src for e in self.edges if e.dst == node_id]

    def copy(self) -> "IdeaGraph":
        ig = IdeaGraph()
        ig.nodes = {k: Node(v.id, dict(v.attrs)) for k, v in self.nodes.items()}
        ig.edges = [Edge(e.src, e.dst, e.relation, dict(e.attrs)) for e in self.edges]
        ig.aspects = dict(self.aspects)
        ig.constraints = dict(self.constraints)
        ig.goals = dict(self.goals)
        return ig


# -------------------------
# Вспомогательные утилиты
# -------------------------

def _safe_log(x: float) -> float:
    return math.log(max(1e-12, x))


def _normalize(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Distribution must sum to > 0")
    return p / s


# -------------------------
# Реализация операторов
# -------------------------

class Operators:
    """Коллекция статических операторов."""

    @staticmethod
    def aspect_rotate(ig: IdeaGraph, axis: str, delta: float, rephrase_goal: bool = False) -> Tuple[IdeaGraph, Dict[str, Any]]:
        """Повернуть идею по оси аспектов.
        Возвращает обновлённый граф и лог.
        """
        out = ig.copy()
        out.aspects[axis] = out.aspects.get(axis, 0.0) + float(delta)
        log = {"op": "ASPECT_ROTATE", "axis": axis, "delta": delta, "new_aspect": out.aspects[axis]}
        if rephrase_goal:
            # Примитивная переформулировка: меняем масштабы/единицы в целях, если заданы числом.
            new_goals = {}
            for k, v in out.goals.items():
                if isinstance(v, (int, float)):
                    new_goals[k] = v * (1.0 + 0.1 * delta)
                else:
                    new_goals[k] = v
            out.goals = new_goals
            log["rephrased_goals"] = True
        return out, log

    @staticmethod
    def re_represent(ig: IdeaGraph, remove_constraints: Optional[List[str]] = None, decompose_chunks: Optional[List[str]] = None) -> Tuple[IdeaGraph, Dict[str, Any]]:
        """Сменить репрезентацию/снять ограничения/разукрупнить узлы."""
        out = ig.copy()
        removed = []
        if remove_constraints:
            for k in remove_constraints:
                if k in out.constraints:
                    out.constraints.pop(k, None)
                    removed.append(k)

        decomposed = {}
        if decompose_chunks:
            for nid in decompose_chunks:
                if nid in out.nodes and "chunks" in out.nodes[nid].attrs:
                    chunks = out.nodes[nid].attrs.pop("chunks")
                    # Создаём подузлы
                    for i, ch in enumerate(chunks):
                        new_id = f"{nid}_part{i+1}"
                        out.add_node(new_id, **({"from_chunk": nid, **(ch if isinstance(ch, dict) else {"value": ch})}))
                        out.add_edge(nid, new_id, "part_of")
                    decomposed[nid] = chunks
        log = {"op": "RE_REPRESENT", "removed_constraints": removed, "decomposed": list(decomposed.keys())}
        return out, log

    @staticmethod
    def analogize(target: IdeaGraph, base: IdeaGraph, preserve_relations: Optional[List[str]] = None) -> Dict[str, Any]:
        """Структурная аналогия (упрощённо): пытаемся выровнять отношения по типам.
        Возвращает mapping и возможные 'предсказания'.
        """
        preserve_relations = preserve_relations or list({e.relation for e in base.edges})
        # Строим индексы связей
        def index_rel(ig: IdeaGraph) -> Dict[str, List[Tuple[str, str]]]:
            idx = {}
            for e in ig.edges:
                idx.setdefault(e.relation, []).append((e.src, e.dst))
            return idx

        base_idx = index_rel(base)
        targ_idx = index_rel(target)

        mapping = {}
        predictions = []
        for r in preserve_relations:
            b_pairs = set(base_idx.get(r, []))
            t_pairs = set(targ_idx.get(r, []))
            # если в базе есть связь, а в цели ещё нет — предлагаем 'предсказание' подобной связи
            for (s, d) in b_pairs - t_pairs:
                # эвристика: ищем узлы в target с схожими именами
                s_ = s if s in target.nodes else next((n for n in target.nodes if n.lower() in s.lower() or s.lower() in n.lower()), None)
                d_ = d if d in target.nodes else next((n for n in target.nodes if n.lower() in d.lower() or d.lower() in n.lower()), None)
                if s_ and d_:
                    predictions.append({"relation": r, "src": s_, "dst": d_, "why": "structural_analogy"})
            mapping[r] = r  # тождественное сохранение типа связи
        return {"op": "ANALOGIZE", "mapping": mapping, "predictions": predictions}

    @staticmethod
    def blend(space_A: IdeaGraph, space_B: IdeaGraph, alignment_hints: Optional[List[Tuple[str, str]]] = None) -> Tuple[IdeaGraph, Dict[str, Any]]:
        """Концептуальный бленд: объединяем графы с частичной унификацией узлов по подсказкам alignment_hints."""
        out = space_A.copy()
        log = {"op": "BLEND", "merged_nodes": [], "merged_edges": 0}
        mapping = dict(alignment_hints or [])
        # Слияние узлов
        for nid, node in space_B.nodes.items():
            target_id = mapping.get(nid, nid)
            if target_id in out.nodes:
                # сливаем атрибуты
                out.nodes[target_id].attrs.update({f"B::{k}": v for k, v in node.attrs.items()})
                log["merged_nodes"].append((nid, target_id))
            else:
                out.add_node(target_id, **node.attrs)
        # Слияние ребер
        for e in space_B.edges:
            s = mapping.get(e.src, e.src)
            d = mapping.get(e.dst, e.dst)
            out.add_edge(s, d, e.relation, **e.attrs)
            log["merged_edges"] += 1
        return out, log

    @staticmethod
    def morphology(axes: Dict[str, List[Any]], cca_rule: Optional[Callable[[Dict[str, Any]], bool]] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Морфологический анализ: декартово произведение осей с опциональным отсеиванием по CCA."""
        keys = list(axes.keys())
        values = [axes[k] for k in keys]
        combos = []
        for prod in itertools.product(*values):
            candidate = dict(zip(keys, prod))
            if cca_rule is None or bool(cca_rule(candidate)):
                combos.append(candidate)
        if top_k is not None:
            combos = combos[:top_k]
        return {"op": "MORPHOLOGY", "count": len(combos), "candidates": combos}

    @staticmethod
    def triz_contradiction(improve: str, worsen: str) -> Dict[str, Any]:
        """Подбор ТРИЗ-принципов (упрощённый). Возвращает подмножество из 40 принципов + эскизы ходов."""
        principles40 = [
            "Segmentation", "Taking out", "Local quality", "Asymmetry", "Merging",
            "Universality", "Nested doll", "Counterweight", "Prior counteraction",
            "Preliminary action", "Prior cushioning", "Equipotentiality", "Inversion",
            "Spheroidality/Curvature", "Dynamics", "Partial/Excessive action",
            "Another dimension", "Mechanical vibration", "Periodic action", "Continuity of useful action",
            "Rushing through", "Blessing in disguise", "Feedback", "Intermediary", "Self-service",
            "Copying", "Cheap short-living", "Replacement of mechanical system", "Pneumatics/Hydraulics",
            "Flexible shells/Thin films", "Porous materials", "Color changes", "Homogeneity",
            "Discarding and recovering", "Parameter changes", "Phase transitions", "Thermal expansion",
            "Accelerated oxidation", "Inert atmosphere", "Composite materials"
        ]
        # Небольшая эвристика для типичных пар (улучшить энергоэффективность, но не ухудшить производительность и т.п.)
        heur = {
            ("energy_efficiency", "performance"): ["Local quality", "Preliminary action", "Inversion", "Segmentation"],
            ("weight", "strength"): ["Composite materials", "Another dimension", "Asymmetry"],
            ("speed", "accuracy"): ["Feedback", "Partial/Excessive action", "Dynamics"],
        }
        key = (improve, worsen)
        picks = heur.get(key, random.sample(principles40, k=5))
        design_moves = [f"Apply principle '{p}' to address ({improve}↑, {worsen}≈)" for p in picks]
        return {"op": "TRIZ_CONTRADICTION", "principles": picks, "design_moves": design_moves}

    @staticmethod
    def dimensionalize(dim_matrix: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
        """Buckingham Π: находит базис ядра матрицы размерностей (переменные x базовые единицы).
        dim_matrix: shape = (n_vars, n_dims), каждая строка — экспоненты для [M, L, T, ...]
        Возвращает коэффициенты групп Π в виде списков.
        """
        if dim_matrix.ndim != 2:
            raise ValueError("dim_matrix must be 2D (n_vars x n_dims)")
        # Находим правое ядро (nullspace) с помощью SVD
        U, S, Vt = np.linalg.svd(dim_matrix.astype(float), full_matrices=True)
        tol = 1e-10
        rank = (S > tol).sum()
        nullspace = Vt[rank:].T  # n_dims x (n_dims-rank)
        # Переводим базис nullspace из пространства базовых ед. в коэффициенты по переменным:
        # Мы хотим найти w такие, что dim_matrix^T * w = 0 → w в ядре dim_matrix^T
        # Эквивалентно ядру dim_matrix.T
        Ut, St, Vtt = np.linalg.svd(dim_matrix.T.astype(float), full_matrices=True)
        rank_t = (St > tol).sum()
        null_vars = Vtt[rank_t:].T  # n_vars x (n_vars-rank_t)
        groups = []
        for i in range(null_vars.shape[1]):
            coeffs = null_vars[:, i]
            # Нормируем до целых-рациональных (по возможности)
            if np.all(np.abs(coeffs) < 1e-12):
                continue
            coeffs = coeffs / (np.min(np.abs(coeffs[np.abs(coeffs) > 1e-12])))
            coeffs = np.round(coeffs).astype(int)
            group = {var_names[j]: int(coeffs[j]) for j in range(len(var_names)) if coeffs[j] != 0}
            groups.append(group)
        return {"op": "DIMENSIONALIZE", "pi_groups": groups}

    @staticmethod
    def invariant_symmetry(candidate_symmetries: List[str], action: str = "enforce") -> Dict[str, Any]:
        """Фиксирует или нарушает симметрию (текстовая заглушка для прототипа)."""
        if action not in {"enforce", "break"}:
            raise ValueError("action must be 'enforce' or 'break'")
        consequences = []
        for s in candidate_symmetries:
            if action == "enforce":
                consequences.append(f"Conservation law implied by symmetry '{s}' (Noether-style reasoning).")
            else:
                consequences.append(f"Potential new effect by breaking symmetry '{s}' (evaluate measurables).")
        return {"op": "INVARIANT_SYMMETRY", "action": action, "consequences": consequences}

    @staticmethod
    def extremize(ig: IdeaGraph, param: str, limit: str = "zero") -> Dict[str, Any]:
        """Экстремальные случаи параметров (→0 или →∞) — выявление инвариантов/ломающихся связей."""
        if limit not in {"zero", "inf"}:
            raise ValueError("limit must be 'zero' or 'inf'")
        invariants = []
        failures = []
        # В прототипе просто отмечаем, какие связи зависят от param
        for e in ig.edges:
            if param in e.attrs.get("depends_on", []):
                failures.append((e.src, e.dst, e.relation))
        invariants = [n for n in ig.nodes if param not in ig.nodes[n].attrs.get("depends_on", [])]
        return {"op": "EXTREMIZE", "param": param, "limit": limit, "invariants": invariants, "failure_modes": failures}

    @staticmethod
    def abduct(observation: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Абдукция: ранжируем гипотезы по апостериорной вероятности (если есть priors/likelihood)."""
        ranked = []
        total = 0.0
        for h in candidates:
            prior = float(h.get("prior", 1.0 / max(1, len(candidates))))
            like = float(h.get("likelihood", 0.5))
            score = prior * like
            ranked.append((h.get("name", "hyp"), score))
            total += score
        ranked.sort(key=lambda x: x[1], reverse=True)
        posterior = [(name, (s / total) if total > 0 else 0.0) for name, s in ranked]
        return {"op": "ABDUCT", "observation": observation, "posterior": posterior}

    @staticmethod
    def thought_experiment(assumptions: List[str], scenario: str, hypotheses: Optional[List[str]] = None) -> Dict[str, Any]:
        """Мысленный эксперимент: просто помечает гипотезы, конфликтующие с набором предпосылок."""
        hypotheses = hypotheses or []
        refuted = []
        survivors = list(hypotheses)
        for h in hypotheses or []:
            for a in assumptions:
                if f"not({a})" in h or f"violates({a})" in h:
                    refuted.append(h)
                    if h in survivors:
                        survivors.remove(h)
                    break
        return {"op": "THOUGHT_EXPERIMENT", "scenario": scenario, "refuted": refuted, "survivors": survivors}

    @staticmethod
    def perspective_swap(ig: IdeaGraph, role_A: str, role_B: str) -> Tuple[IdeaGraph, Dict[str, Any]]:
        """Смена ролей/перспектив: переопределяем цели/ограничения с точки зрения другой роли."""
        out = ig.copy()
        out.aspects["role"] = {"prev": role_A, "now": role_B}
        # Примитивно меняем знак требований по роли
        new_goals = {}
        for k, v in out.goals.items():
            new_goals[k] = v
        return out, {"op": "PERSPECTIVE_SWAP", "from": role_A, "to": role_B}

    @staticmethod
    def serendipity_hook(sources: List[str], k: int = 5, scorer: Optional[Callable[[str], Tuple[float, float, float]]] = None) -> Dict[str, Any]:
        """Управляемая случайность: выбираем k источников и оцениваем U/I/V."""
        k = min(k, len(sources))
        picks = random.sample(sources, k) if k > 0 else []
        results = []
        for s in picks:
            if scorer is None:
                # Заглушка: случайная оценка (замените на модельную функцию)
                U, I, V = random.random(), random.random(), random.random()
            else:
                U, I, V = scorer(s)
            results.append({"source": s, "U": U, "I": I, "V": V, "UIV": (U + I + V) / 3.0})
        results.sort(key=lambda x: x["UIV"], reverse=True)
        return {"op": "SERENDIPITY_HOOK", "ranked": results}

    @staticmethod
    def evaluate_surprise(prior: np.ndarray, posterior: np.ndarray) -> Dict[str, Any]:
        """Баесовское удивление S = KL(posterior || prior) для категориального распределения."""
        p = _normalize(np.array(prior, dtype=float))
        q = _normalize(np.array(posterior, dtype=float))
        # KL(q || p)
        eps = 1e-12
        S = float(np.sum(q * (np.log(q + eps) - np.log(p + eps))))
        return {"op": "EVALUATE_SURPRISE", "surprise": S}


# -------------------------
# Метрики комбинирования/новизны
# -------------------------

def uzzi_zscore(pair_counts: Dict[Tuple[str, str], int],
                marginals: Dict[str, int],
                pair: Tuple[str, str]) -> float:
    """Приближенный z-score атипичности пары (Uzzi et al.-style), через независимую модель.
    z = (obs - E) / sqrt(E), где E = f_i * f_j / N, N = сумма всех маргиналий.
    Примечание: в оригинале используются более тонкие ядровые оценки; здесь инженерный скелет.
    """
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
    """Оценивает долю новых комбинаций и темп 'explore' по истории наборов блоков.
    hist_blocks: список наборов (перечни строк), каждый — один проект/работа.
    """
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
    explore_rate = share  # эвристика
    return {"new_combo_share": share, "explore_rate": explore_rate}


# -------------------------
# Оркестратор
# -------------------------

class IdeaEngine:
    """Двухфазный цикл: дивергенция → конвергенция.
    Вы можете задать собственные политики выбора операторов и метрики отбора.
    """
    def __init__(self,
                 divergence_ops: Optional[List[Callable]] = None,
                 convergence_ops: Optional[List[Callable]] = None,
                 random_seed: Optional[int] = None):
        self.divergence_ops = divergence_ops or [
            Operators.morphology,
            Operators.triz_contradiction,
            Operators.analogize,
            Operators.blend,
            Operators.serendipity_hook,
            Operators.abduct
        ]
        self.convergence_ops = convergence_ops or [
            Operators.re_represent,
            Operators.aspect_rotate,
            Operators.dimensionalize,
            Operators.invariant_symmetry,
            Operators.extremize,
            Operators.thought_experiment,
            Operators.evaluate_surprise
        ]
        if random_seed is not None:
            random.seed(random_seed)

    def run_divergence(self, ig: IdeaGraph, **kwargs) -> List[Dict[str, Any]]:
        """Выполняет набор дивергентных операторов, возвращает список артефактов/гипотез."""
        out = []
        # Примеры вызовов — вы можете передавать свои аргументы через kwargs
        if "morph_axes" in kwargs:
            out.append(Operators.morphology(kwargs["morph_axes"], kwargs.get("cca_rule"), kwargs.get("top_k")))
        if "triz" in kwargs:
            imp, wor = kwargs["triz"]
            out.append(Operators.triz_contradiction(imp, wor))
        if "analogize" in kwargs:
            targ, base, pres = kwargs["analogize"]
            out.append(Operators.analogize(targ, base, pres))
        if "blend" in kwargs:
            A, B, hints = kwargs["blend"]
            b_ig, b_log = Operators.blend(A, B, hints)
            out.append(b_log)
        if "serendipity_sources" in kwargs:
            out.append(Operators.serendipity_hook(kwargs["serendipity_sources"], kwargs.get("k", 5)))
        if "abduct" in kwargs:
            obs, cands = kwargs["abduct"]
            out.append(Operators.abduct(obs, cands))
        return out

    def run_convergence(self, ig: IdeaGraph, **kwargs) -> List[Dict[str, Any]]:
        """Выполняет конвергентные проверки/фильтры."""
        out = []
        if "aspect" in kwargs:
            axis, delta = kwargs["aspect"]
            ig, log = Operators.aspect_rotate(ig, axis, delta, rephrase_goal=True)
            out.append(log)
        if "re_represent" in kwargs:
            ig, log = Operators.re_represent(ig, kwargs["re_represent"].get("remove"), kwargs["re_represent"].get("decompose"))
            out.append(log)
        if "dimensionalize" in kwargs:
            dimM, names = kwargs["dimensionalize"]
            out.append(Operators.dimensionalize(dimM, names))
        if "invariant" in kwargs:
            syms, act = kwargs["invariant"]
            out.append(Operators.invariant_symmetry(syms, act))
        if "extreme" in kwargs:
            param, lim = kwargs["extreme"]
            out.append(Operators.extremize(ig, param, lim))
        if "thought" in kwargs:
            assump, scen, hyps = kwargs["thought"]
            out.append(Operators.thought_experiment(assump, scen, hyps))
        if "surprise" in kwargs:
            prior, post = kwargs["surprise"]
            out.append(Operators.evaluate_surprise(prior, post))
        return out


# -------------------------
# Пример использования
# -------------------------

def _toy_example() -> Dict[str, Any]:
    """Демонстрация минимальной трассы на игрушечном примере (охлаждение ЦОДов)."""
    # Строим целевой граф (data center cooling)
    target = IdeaGraph(
        goals={"energy_use_kwh": 1000, "uptime": 0.999},
        constraints={"no_hardware_change": False}
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
    div = engine.run_divergence(
        target,
        morph_axes=morph_axes,
        cca_rule=lambda c: not (c["airflow"] == "forced" and c["layout"] == "chimneys"),
        top_k=6,
        triz=("energy_efficiency", "performance"),
        analogize=(target, base, ["produces", "cools", "extracts"]),
        blend=(target, base, [("HotAir", "WarmAir"), ("ColdAir", "CoolAir")]),
        serendipity_sources=["paper: termite mounds", "paper: radiative cooling", "paper: MOF sorbents", "paper: heat pipes", "paper: phase-change materials"],
        k=3,
        abduct=("observed_hotspots", [{"name": "insufficient_flow", "prior": 0.5, "likelihood": 0.7},
                                      {"name": "poor_contact", "prior": 0.3, "likelihood": 0.6},
                                      {"name": "sensor_bias", "prior": 0.2, "likelihood": 0.1}])
    )

    # Конвергенция
    dimM = np.array([
        #   M  L  T
        [ 0, 1, -1],  # velocity L/T
        [ 1, 0, -3],  # power M/T^3 (примерно)
        [ 0, 1,  0],  # length L
    ])
    conv = engine.run_convergence(
        target,
        aspect=("abstraction", +1.0),
        re_represent={"remove": ["no_hardware_change"], "decompose": []},
        dimensionalize=(dimM, ["v", "P", "L"]),
        invariant=(["time_translation"], "enforce"),
        extreme=("flow", "zero"),
        thought=(["adiabatic"], "no_external_cooling", ["violates(adiabatic):radiative_cooling", "passive_convection"]),
        surprise=(np.array([0.4, 0.3, 0.3]), np.array([0.2, 0.6, 0.2]))
    )

    # Метрики комбинирования
    recomb = recombination_metrics([
        ["forced", "plate", "open_hot_cold"],
        ["passive", "sorption", "porous_panels"],
        ["passive", "phase_change", "chimneys"],
    ])
    uzzi = uzzi_zscore(
        pair_counts={("passive", "chimneys"): 1, ("forced", "plate"): 5},
        marginals={"passive": 10, "chimneys": 2, "forced": 20, "plate": 8},
        pair=("passive", "chimneys")
    )

    return {"divergence": div, "convergence": conv, "recombination": recomb, "uzzi_z": uzzi}


def generate_task_trace(task_description: str, random_seed: int = 42) -> Dict[str, Any]:
    """Сборка трассы операторов под конкретную формулировку задачи."""
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
    divergence = engine.run_divergence(
        target,
        morph_axes=morph_axes,
        cca_rule=lambda cfg: not (
            cfg["lattice_patch"] == "3x3x3" and cfg["state_repr"] == "python_dfs"
        ),
        top_k=8,
        triz=("order_depth", "runtime"),
        analogize=(target, base, ["requires", "produces", "checked_by"]),
        blend=(target, base, [("Enumerators", "Counting"), ("Validation", "Benchmarks")]),
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

    convergence = engine.run_convergence(
        target,
        aspect=("abstraction", +0.5),
        re_represent={"remove": ["only_python"], "decompose": []},
        invariant=(["translation", "parity"], "enforce"),
        extreme=("max_edges", "inf"),
        thought=(
            ["connected_only"],
            "wraparound_edges",
            ["cycle_basis_filter", "bridge_prune"],
        ),
        surprise=(np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.5, 0.3])),
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
        marginals={"frontier_dp": 10, "symmetry_orbits": 2, "dfs_walks": 15, "loop_hash": 8},
        pair=("frontier_dp", "symmetry_orbits"),
    )

    return {
        "task": task_description,
        "divergence": divergence,
        "convergence": convergence,
        "recombination": recomb,
        "uzzi_z": uzzi,
    }


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IdeaLab operator trace generator")
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
        help="Путь для сохранения текстового представления",
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
