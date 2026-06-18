# -*- coding: utf-8 -*-
"""
============================================================================
 solver.py — Matheurística híbrida VNS + Gurobi (v5 / vns_v2)
============================================================================

A v5 mantém TODO o GVNS da v1 (busca + verificação integral das 43
restrições) e ACRESCENTA uma camada matheurística: a metaheurística recorta
o problema em subproblemas pequenos e o Gurobi os reotimiza até a otimalidade
(fix-and-optimize / LNS-exato / POPMUSIC). Ver plano_metaheuristica_v5.md.

Organization (the MODEL semantics live in constraints.py; the tunable
PARAMETERS live in config.py — this file holds only the SEARCH):

  1. DebugLog            — console + debug_log.txt logging
  2. Repairs             — repair_topologico, iterative_repair,
                           repair_feasibility_fast, repair_final_cap
  3. Initial solution    — greedy_construction (B_apr-aware, pessimistic)
                           + chain/monolith template variants (multi-start)
  4. Neighborhoods N1–N10 — shaking operators
  5. VND (L1–L5)         — local search with internal-iteration logging
  6. GVNS loop           — INTEGRAL verification of the 43 constraints at
                           every incumbent transition: a candidate that
                           would violate any ENFORCE constraint is REJECTED
                           and the reason is logged.
  6b. MATHEURISTIC       — free-set selectors + mip_reoptimize (sub-MIP do
                           Gurobi) + fix_and_optimize_sweep (POPMUSIC). Todo
                           candidato do sub-MIP passa pelo MESMO gate das 43
                           restrições; o Tmax aceito é SEMPRE o do evaluate()
                           (nunca o do Gurobi — ver E-024 no changes.md).
  7. Output              — CSVs, convergence logs (VNS and VND) and the
                           final report over the 43 constraints.

Solution representation:
  - sa[j] ∈ {1..S_max}; fm[j] sums to 1.0 with fractions from VALID_COMBOS.
"""

import os
import glob
import time
import math
import random
import itertools

import pandas as pd

from . import config as cfg
from . import constraints as vc
from .constraints import evaluate, penalized_objective, EPS

# Re-exported for convenience to callers that import from this module.
VALID_COMBOS = cfg.VALID_COMBOS
VALID_COMBOS_SET = cfg.VALID_COMBOS_SET
VNS_GAP_TARGET = cfg.VNS_GAP_TARGET


def _twelfths(frac_row):
    """Frações decimais não-nulas → tupla ordenada de doze avos."""
    return tuple(sorted(round(f * 12) for f in frac_row if f > EPS))


def is_valid_frac(frac_row):
    return _twelfths(frac_row) in VALID_COMBOS_SET


def compute_time_limit(N: int, M: int, S: int) -> float:
    """Tempo limite dinâmico escalado por cfg.T_MULT.
    Referência (T_MULT=1): Cenário 1 (75×5×15=5625) → ~420s."""
    t = cfg.T_MULT * cfg.T_BASE_REF * math.sqrt((N * M * S) / cfg.NVAR_REF)
    return max(cfg.T_CLAMP_MIN * cfg.T_MULT,
               min(cfg.T_CLAMP_MAX * cfg.T_MULT, t))


# ============================================================================
# 1. LOG DE DEBUG (console + arquivo)
# ============================================================================

class DebugLog:
    """Registra todo o detalhe da busca em debug_log.txt e controla o que
    aparece no terminal. Métodos:
      milestone() — marco importante: SEMPRE no terminal e no arquivo;
      console()   — detalhe: no arquivo sempre, no terminal só se show_console;
      section()   — banner: arquivo sempre, terminal só se show_console;
      to_file()   — só no arquivo;
      verbose()   — só no arquivo e só se DEBUG_LEVEL >= 2.
    O pipeline chama o solver com show_console=False (terminal limpo); o
    debug_log.txt guarda a transparência completa independentemente disso."""

    def __init__(self, log_path, show_console=True):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._f = open(log_path, "w", encoding="utf-8")
        self._t0 = time.time()
        self.show_console = show_console

    def _stamp(self):
        return f"[{time.time() - self._t0:8.1f}s]"

    def section(self, title):
        line = f"\n{'=' * 70}\n  {title}\n{'=' * 70}"
        if self.show_console:
            print(line)
        self._f.write(line + "\n")
        self._f.flush()

    def milestone(self, msg):
        print(msg)
        self._f.write(f"{self._stamp()} {msg}\n")
        self._f.flush()

    def console(self, msg):
        if self.show_console:
            print(msg)
        self._f.write(f"{self._stamp()} {msg}\n")
        self._f.flush()

    def to_file(self, msg):
        self._f.write(f"{self._stamp()} {msg}\n")

    def verbose(self, msg):
        if cfg.DEBUG_LEVEL >= 2:
            self._f.write(f"{self._stamp()} {msg}\n")

    def block_to_file(self, texto):
        self._f.write(texto + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ============================================================================
# 2. REPAROS
# ============================================================================

def repair_topologico(inst, sa, Dur_approx, fm=None):
    """Garante precedências temporais; quando fm é fornecida, considera
    conflitos de dev e a duração estimada para avançar a task de sprint
    quando ela não cabe na janela. Modifica e retorna sa."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    H = cfg.H_SPAN
    Inic_r = [0.0] * N
    dev_busy = [[0.0] * (S_max + 2) for _ in range(M)]

    for j in inst["topo"]:
        sprint_min = 1
        for k in inst["Pred"][j]:
            end_k = Inic_r[k] + Dur_approx[k] + inst["Gap_prec"]
            s_need = math.ceil(end_k / H) if end_k > 0 else 1
            sprint_min = max(sprint_min, s_need)
        sa[j] = min(max(sa[j], sprint_min), S_max)

        team = ([i for i in range(M) if fm[j][i] > EPS] if fm is not None
                  else [])
        preds_end = max((Inic_r[k] + Dur_approx[k] + inst["Gap_prec"]
                         for k in inst["Pred"][j]), default=0.0)
        s = sa[j]
        conflict = max((dev_busy[i][s] for i in team), default=0.0)
        Inic_r[j] = max((s - 1) * H, preds_end, conflict)

        while Inic_r[j] + Dur_approx[j] > s * H and s < S_max:
            s += 1
            sa[j] = s
            conflict = max((dev_busy[i][s] for i in team), default=0.0)
            Inic_r[j] = max((s - 1) * H, preds_end, conflict)

        for i in team:
            dev_busy[i][s] = max(dev_busy[i][s], Inic_r[j] + Dur_approx[j])

    return sa


def iterative_repair(inst, sa, fm, log=None, max_iters=None):
    """Reparo com multiplicadores REAIS (via evaluate) até zerar
    pen_cap, pen_window e pen_tasks — ou esgotar attempts.
    É o mesmo repair da fase 2 da gulosa, exposto para reuso no gate
    de aceitação. Retorna (sa, fm, av)."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    if max_iters is None:
        max_iters = N * S_max * 3

    av = evaluate(inst, sa, fm)
    for _ in range(max_iters):
        if av.pen_cap < EPS and av.pen_window < EPS and av.pen_tasks < EPS:
            break

        repaired = False

        # Reparo 1: pen_window — task estoura a janela do sprint
        if av.pen_window > EPS:
            for j in inst["topo"]:
                s = sa[j]
                if av.Inic[j] + av.Dur[j] > s * cfg.H_SPAN + EPS:
                    sa[j] = min(s + 1, S_max)
                    sa = repair_topologico(inst, sa, list(av.Dur), fm)
                    repaired = True
                    break

        # Reparo 2: pen_cap — dev sobrecarregado numa sprint
        if not repaired and av.pen_cap > EPS:
            for s in range(1, S_max + 1):
                for i in range(M):
                    tasks_is = [j for j in range(N)
                                  if sa[j] == s and fm[j][i] > EPS]
                    load = sum(av.T_din[i][j] for j in tasks_is)
                    if load > cfg.CAP_UTIL + EPS and tasks_is:
                        j_mv = tasks_is[-1]
                        sa[j_mv] = min(s + 1, S_max)
                        sa = repair_topologico(inst, sa, list(av.Dur), fm)
                        repaired = True
                        break
                if repaired:
                    break

        # Reparo 3: pen_tasks — dev com mais de 5 tasks numa sprint
        if not repaired and av.pen_tasks > EPS:
            for s in range(1, S_max + 1):
                for i in range(M):
                    tasks_is = [j for j in range(N)
                                  if sa[j] == s and fm[j][i] > EPS]
                    if len(tasks_is) > cfg.MAX_TASKS_PER_DEV_SPRINT:
                        j_mv = tasks_is[-1]
                        sa[j_mv] = min(s + 1, S_max)
                        sa = repair_topologico(inst, sa, list(av.Dur), fm)
                        repaired = True
                        break
                if repaired:
                    break

        if not repaired:
            if log:
                log.to_file("[iterative_repair] sem mais repairs possiveis "
                            f"(pen_cap={av.pen_cap:.3f} "
                            f"pen_jan={av.pen_window:.3f} "
                            f"pen_tsk={av.pen_tasks:.3f})")
            break
        av = evaluate(inst, sa, fm)

    return sa, fm, av


def repair_feasibility_fast(inst, sa_in, fm_in):
    """Reparo rápido pós-shaking (sem chamadas a evaluate()): move tasks
    de slots (dev, sprint) sobrecarregados para os mais folgados usando
    estimativa pessimistic F_PESS_REPAIR. Retorna cópias (sa, fm)."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    F_PESS = cfg.F_PESS_REPAIR
    T_sta = inst["T_sta"]
    sa = list(sa_in)
    fm = [row[:] for row in fm_in]

    for _ in range(cfg.REPAIR_FAST_PASSES):
        load = [[0.0] * (S_max + 2) for _ in range(M)]
        for j in range(N):
            s = sa[j]
            for i in range(M):
                if fm[j][i] > EPS:
                    load[i][s] += T_sta[i][j] * fm[j][i] * F_PESS + cfg.SETUP_HOURS

        mudou = False
        for i in range(M):
            for s in range(1, S_max + 1):
                if load[i][s] <= cfg.CAP_UTIL * 1.001:
                    continue
                tasks_is = [(j, T_sta[i][j] * fm[j][i] * F_PESS + cfg.SETUP_HOURS)
                              for j in range(N)
                              if sa[j] == s and fm[j][i] > EPS]
                tasks_is.sort(key=lambda x: -x[1])
                for j, t_j in tasks_is:
                    if load[i][s] <= cfg.CAP_UTIL * 1.001:
                        break
                    best_s, best_slack = None, t_j
                    for s_c in range(1, S_max + 1):
                        if s_c == s:
                            continue
                        folga = cfg.CAP_UTIL - load[i][s_c]
                        if folga >= t_j and folga > best_slack:
                            best_slack, best_s = folga, s_c
                    if best_s is not None:
                        load[i][s] -= t_j
                        load[i][best_s] += t_j
                        sa[j] = best_s
                        mudou = True

        sa = repair_topologico(inst, sa, [cfg.H_SPAN / 3.0] * N, fm)
        if not mudou:
            break

    return sa, fm


def repair_final_cap(inst, sa_in, fm_in, log=None):
    """Reparo direcionado de pen_cap residual com avaliações EXATAS.
    Estratégias por task do slot mais violated:
      1. mover de sprint;  2. dividir frações (task > CapUtil sozinha);
      3. boost de B_apr (adiciona o dev violated a task do axis em sprint
         anterior — sub-estratégias in-place e mover-axis).
    Aceita qualquer mudança que REDUZA pen_cap. Retorna (sa, fm)."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    sa = list(sa_in)
    fm = [row[:] for row in fm_in]

    for _ in range(N * 3):
        av = evaluate(inst, sa, fm)
        if av.pen_cap < 1e-6:
            break
        pen_atual = av.pen_cap

        worst_exc, worst_i, worst_s = 0.0, -1, -1
        for i in range(M):
            for s in range(1, S_max + 1):
                load = sum(av.T_din[i][j] for j in range(N)
                            if sa[j] == s and fm[j][i] > EPS)
                exc = load - cfg.CAP_UTIL
                if exc > worst_exc:
                    worst_exc, worst_i, worst_s = exc, i, s
        if worst_i < 0:
            break

        tasks_vs = sorted(
            [(j, av.T_din[worst_i][j]) for j in range(N)
             if sa[j] == worst_s and fm[j][worst_i] > EPS],
            key=lambda x: -x[1])

        moveu = False
        for j, t_j in tasks_vs:
            if moveu:
                break

            # Estratégia 1: mover de sprint
            for s_novo in range(1, S_max + 1):
                if s_novo == worst_s:
                    continue
                sa_t = list(sa)
                sa_t[j] = s_novo
                sa_t = repair_topologico(inst, sa_t, list(av.Dur), fm)
                if evaluate(inst, sa_t, fm).pen_cap < pen_atual - 1e-6:
                    sa = sa_t
                    moveu = True
                    break

            # Estratégia 2: dividir frações (task não cabe sozinha)
            if not moveu and t_j > cfg.CAP_UTIL - cfg.SETUP_HOURS:
                for combo in [c for c in VALID_COMBOS if len(c) >= 2]:
                    if moveu or len(combo) > M:
                        continue
                    if len(combo) <= 3:
                        dev_iters = itertools.permutations(range(M), len(combo))
                    else:
                        dev_iters = [random.sample(range(M), len(combo))
                                     for _ in range(8)]
                    for devs in dev_iters:
                        fm_t = [row[:] for row in fm]
                        fm_t[j] = [0.0] * M
                        for d_idx, f12 in zip(devs, combo):
                            fm_t[j][d_idx] = f12 / 12.0
                        if not is_valid_frac(fm_t[j]):
                            continue
                        if evaluate(inst, sa, fm_t).pen_cap < pen_atual - 1e-6:
                            fm = fm_t
                            moveu = True
                            break

            # Estratégia 3: boost de B_apr para task solo marginal
            team_j = [i for i in range(M) if fm[j][i] > EPS]
            if not moveu and len(team_j) == 1 and team_j[0] == worst_i:
                for axis_k in inst["axis"][j]:
                    if moveu or fm[axis_k][worst_i] > EPS:
                        continue
                    team_k = [d for d in range(M) if fm[axis_k][d] > EPS]
                    if len(team_k) != 1:
                        continue
                    dev_orig = team_k[0]
                    for s_k in range(1, worst_s):
                        if moveu:
                            break
                        for f_novo in (3, 4):
                            fm_t = [row[:] for row in fm]
                            fm_t[axis_k] = [0.0] * M
                            fm_t[axis_k][worst_i] = f_novo / 12.0
                            fm_t[axis_k][dev_orig] = (12 - f_novo) / 12.0
                            if not is_valid_frac(fm_t[axis_k]):
                                continue
                            sa_t = list(sa)
                            sa_t[axis_k] = s_k
                            sa_t = repair_topologico(inst, sa_t,
                                                     list(av.Dur), fm_t)
                            if evaluate(inst, sa_t, fm_t).pen_cap < pen_atual - 1e-6:
                                sa, fm = sa_t, fm_t
                                moveu = True
                                break

            # Estratégia 3c: PARES de tasks axis — quando adicionar o dev a
            # UMA task axis não basta para o B_apr cruzar o threshold (caso
            # E-018: B_apr=0.144 < 0.146), tenta adicionar a DUAS tasks axis
            # solo já em sprints anteriores, avaliando o efeito cumulativo.
            if not moveu and len(team_j) == 1 and team_j[0] == worst_i:
                axis_cands = [k for k in inst["axis"][j]
                              if sa[k] < worst_s and fm[k][worst_i] < EPS
                              and sum(1 for d in range(M) if fm[k][d] > EPS) == 1]
                for a in range(len(axis_cands)):
                    if moveu:
                        break
                    for b in range(a + 1, len(axis_cands)):
                        k1, k2 = axis_cands[a], axis_cands[b]
                        fm_t = [row[:] for row in fm]
                        ok_par = True
                        for kk in (k1, k2):
                            dev_orig = next(d for d in range(M)
                                            if fm[kk][d] > EPS)
                            fm_t[kk] = [0.0] * M
                            fm_t[kk][worst_i] = 3 / 12.0
                            fm_t[kk][dev_orig] = 9 / 12.0
                            if not is_valid_frac(fm_t[kk]):
                                ok_par = False
                                break
                        if not ok_par:
                            continue
                        if evaluate(inst, sa, fm_t).pen_cap < pen_atual - 1e-6:
                            fm = fm_t
                            moveu = True
                            break

        if not moveu:
            break

    return sa, fm


def full_repair(inst, sa_in, fm_in, log=None):
    """Pipeline de repair usado antes do gate de aceitação:
    repair iterativo (janela/cap/tasks) + repair direcionado de cap residual.
    O repair exato (caro) só roda quando o excesso de capacity é pequeno
    (<= REPAIR_CAP_LIMIT_H horas) — candidatos muito infactíveis não valem
    o cost computacional."""
    sa, fm, av = iterative_repair(inst, list(sa_in),
                                  [row[:] for row in fm_in], log)
    excesso_h = av.pen_cap / cfg.CAP_UTIL   # pen_cap = horas_excesso * CapUtil
    if 1e-6 < av.pen_cap and excesso_h <= cfg.REPAIR_CAP_LIMIT_H:
        sa, fm = repair_final_cap(inst, sa, fm, log)
    return sa, fm


# ============================================================================
# 3. SOLUÇÃO INICIAL — HEURÍSTICA GULOSA (B_apr-aware + pessimistic)
# ============================================================================

def greedy_construction(inst, log):
    """Fase 1: para cada task (order topológica), escolhe (dev, sprint)
    minimizando o T_din ESTIMADO com multiplicadores (P_ctx, P_phi, B_apr).
    A estimativa de load usa F_TOTAL_PESS (pessimistic) para não criar
    sprints irrealistas. NUNCA reverter para um greedy "natural-match"
    sem B_apr — ver changes.md v2.2.
    Fase 2: repair iterativo com multiplicadores reais até factibilidade."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    T_sta, F_beh = inst["T_sta"], inst["F_beh"]
    F_PESS = cfg.F_TOTAL_PESS

    sa = [1] * N
    fm = [[0.0] * M for _ in range(N)]
    dev_of = [-1] * N

    load_pess = [[0.0] * (S_max + 2) for _ in range(M)]
    n_tasks = [[0] * (S_max + 2) for _ in range(M)]
    sum_fbeh = [[0.0] * (S_max + 2) for _ in range(M)]

    def estimate_b_apr(j, i, s_j):
        b_axis = sum(cfg.LAMBDA_LEARN ** (s_j - sa[k])
                     for k in inst["axis"][j]
                     if dev_of[k] == i and sa[k] < s_j)
        b_dep = sum(cfg.LAMBDA_DEP ** (dist - 1)
                    for k, dist in inst["pred_dist"][j].items()
                    if dev_of[k] == i)
        return cfg.ALPHA_AXIS * b_axis + cfg.ALPHA_DEP * b_dep

    def earliest_sprint(i, s_min, tb):
        t_pess = tb * F_PESS + cfg.SETUP_HOURS
        for s in range(s_min, S_max + 1):
            if (load_pess[i][s] + t_pess <= cfg.CAP_UTIL
                    and n_tasks[i][s] < cfg.MAX_TASKS_PER_DEV_SPRINT):
                return s
        cands = [s for s in range(s_min, S_max + 1)
                 if n_tasks[i][s] < cfg.MAX_TASKS_PER_DEV_SPRINT]
        return min(cands, key=lambda s: load_pess[i][s]) if cands else s_min

    for j in inst["topo"]:
        s_min = max((sa[k] for k in inst["Pred"][j]), default=1)

        best_dev, best_sprint, best_cost = 0, S_max + 1, float("inf")
        for i in range(M):
            tb = T_sta[i][j]
            s_i = earliest_sprint(i, s_min, tb)
            b_apr = estimate_b_apr(j, i, s_i)
            p_ctx = cfg.CONTEXT_WEIGHT * n_tasks[i][s_i]
            fbeh_novo = sum_fbeh[i][s_i] + F_beh[i][j]
            p_phi = cfg.FATIGUE_WEIGHT * max(0.0, fbeh_novo / cfg.COG_LOAD_DIVISOR - cfg.FATIGUE_THRESHOLD)
            f_tot = max(0.1, 1.0 + p_ctx + p_phi - b_apr)
            cost = tb * f_tot + cfg.SETUP_HOURS
            if (s_i < best_sprint
                    or (s_i == best_sprint and cost < best_cost)):
                best_dev, best_sprint, best_cost = i, s_i, cost

        sa[j] = best_sprint
        fm[j] = [0.0] * M
        fm[j][best_dev] = 1.0
        dev_of[j] = best_dev
        tb = T_sta[best_dev][j]
        load_pess[best_dev][best_sprint] += tb * F_PESS + cfg.SETUP_HOURS
        n_tasks[best_dev][best_sprint] += 1
        sum_fbeh[best_dev][best_sprint] += F_beh[best_dev][j]

    log.to_file("[gulosa] fase 1 concluida — iniciando repair iterativo")
    sa, fm, av = iterative_repair(inst, sa, fm, log)
    return sa, fm, av


def greedy_construction_chains(inst, log):
    """Construtiva alternativa para instâncias COM precedências (template do
    Gurobi no Cenário 3): decompõe o grafo em CADEIAS (caminhos seguindo
    sucessores), atribui cada chain inteira ao dev de menor T_sta total
    (specialist — acumula b_dep=0.2 por predecessora direta) e empacota
    ~N10_PER_SPRINT tasks de chain por sprint. Tarefas fora de chains
    são distribuídas nos buracos pelo critério guloso padrão."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    T_sta = inst["T_sta"]

    # --- Decomposição em chains: começa nos nós sem predecessoras ---
    visitado = [False] * N
    chains = []
    for j0 in inst["topo"]:
        if visitado[j0] or inst["Pred"][j0]:
            continue
        chain = []
        cur = j0
        while cur is not None and not visitado[cur]:
            visitado[cur] = True
            chain.append(cur)
            prox = [s2 for s2 in inst["succs"][cur] if not visitado[s2]]
            cur = prox[0] if prox else None
        chains.append(chain)
    soltas = [j for j in range(N) if not visitado[j]]
    chains.sort(key=len, reverse=True)

    sa = [1] * N
    fm = [[0.0] * M for _ in range(N)]
    load = [[0.0] * (S_max + 2) for _ in range(M)]
    n_tasks = [[0] * (S_max + 2) for _ in range(M)]

    for chain in chains:
        if len(chain) >= 3:
            i_target = min(range(M),
                         key=lambda d: sum(T_sta[d][j] for j in chain))
        else:
            i_target = None  # chains curtas seguem o critério padrão abaixo

        s_cursor = 1
        no_sprint = 0
        for j in chain:
            s_min = max((sa[k] for k in inst["Pred"][j]), default=1)
            if i_target is not None:
                i = i_target
            else:
                i = min(range(M), key=lambda d: T_sta[d][j])
            t_est = T_sta[i][j] * cfg.CHAIN_GREEDY_F_PESS + cfg.SETUP_HOURS
            s = max(s_cursor, s_min)
            while (s <= S_max
                   and (no_sprint >= cfg.N10_PER_SPRINT
                        or load[i][s] + t_est > cfg.CAP_UTIL
                        or n_tasks[i][s] >= cfg.MAX_TASKS_PER_DEV_SPRINT)):
                if s == s_cursor and no_sprint == 0:
                    break  # sprint vazio para esta chain: usa mesmo assim
                s += 1
                no_sprint = 0
            s = min(s, S_max)
            if s != s_cursor:
                no_sprint = 0
                s_cursor = s
            sa[j] = s
            fm[j] = [0.0] * M
            fm[j][i] = 1.0
            load[i][s] += t_est
            n_tasks[i][s] += 1
            no_sprint += 1

    # --- Tarefas soltas: dev/sprint mais cedo com folga (padrão guloso) ---
    for j in soltas:
        s_min = max((sa[k] for k in inst["Pred"][j]), default=1)
        best = None
        for i in range(M):
            t_est = T_sta[i][j] * cfg.F_TOTAL_PESS + cfg.SETUP_HOURS
            for s in range(s_min, S_max + 1):
                if (load[i][s] + t_est <= cfg.CAP_UTIL
                        and n_tasks[i][s] < cfg.MAX_TASKS_PER_DEV_SPRINT):
                    if best is None or (s, t_est) < best[:2]:
                        best = (s, t_est, i)
                    break
        if best is None:
            i = min(range(M), key=lambda d: T_sta[d][j])
            s = min(range(s_min, S_max + 1), key=lambda x: load[i][x])
            t_est = T_sta[i][j] * cfg.F_TOTAL_PESS + cfg.SETUP_HOURS
        else:
            s, t_est, i = best
        sa[j] = s
        fm[j] = [0.0] * M
        fm[j][i] = 1.0
        load[i][s] += t_est
        n_tasks[i][s] += 1

    log.to_file(f"[gulosa-chains] {len(chains)} chains "
                f"(maior: {max((len(c) for c in chains), default=0)}), "
                f"{len(soltas)} tasks soltas — iniciando repair")
    sa, fm, av = iterative_repair(inst, sa, fm, log)
    return sa, fm, av


def greedy_construction_monoliths(inst, log):
    """Construtiva DO ZERO para instâncias com monoliths — layout exato do
    padrão Gurobi C2:
      - tasks pequenas dos axes dos monoliths viram EQUIPES (warmup) nas
        sprints iniciais; a team de cada warmup contém os devs cujos
        monoliths ainda não começaram (todos acumulam B_apr juntos);
      - o 1º monolith entra na sprint 3 (dev exclusivo, SOZINHO na sprint);
        os demais na sprint 4 em diante, um dev distinto cada;
      - tasks fora dos axes: solo com o best dev, onde houver folga;
      - fixup final: movimentos unitários que reduzam pen_cap (exatos).
    """
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    T_sta = inst["T_sta"]

    monoliths = sorted(
        [j for j in range(N)
         if min(T_sta[i][j] for i in range(M)) + cfg.SETUP_HOURS > cfg.CAP_UTIL],
        key=lambda j: -inst["T_nom"][j])
    if not monoliths or S_max < 4:
        return None

    # Devs specialists (distintos enquanto possível)
    devs_mono = []
    for j in monoliths:
        rank = sorted(range(M), key=lambda d: T_sta[d][j])
        escolhido = next((d for d in rank if d not in devs_mono), rank[0])
        devs_mono.append(escolhido)

    # Sprint de cada monolith: 1º na sprint 3; demais na 4+, M-1 por sprint
    s_mono = {}
    s_mono[monoliths[0]] = 3
    for idx, j in enumerate(monoliths[1:]):
        s_mono[j] = min(S_max, 4 + idx // max(1, M - 1))

    # Pool de warmup: pequenas dos axes dos monoliths; demais ficam free_tasks
    axes_mono = set()
    for j in monoliths:
        axes_mono.update(inst["axis"][j])
    warmup_pool = sorted([k for k in axes_mono if k not in s_mono],
                         key=lambda k: inst["T_nom"][k])
    free_tasks = [j for j in range(N)
              if j not in s_mono and j not in set(warmup_pool)]

    sa = [1] * N
    fm = [[0.0] * M for _ in range(N)]
    n_por_dev = [[0] * (S_max + 2) for _ in range(M)]

    # Monolitos: solo, dev exclusivo
    for j, i in zip(monoliths, devs_mono):
        sa[j] = s_mono[j]
        fm[j][i] = 1.0

    # Warmups: round-robin nas sprints 1..(maior s_mono - 1); a team são
    # os devs de monoliths AINDA NÃO iniciados naquela sprint + fillers
    s_warm_max = max(s_mono.values()) - 1
    s_rr = 1
    for k in warmup_pool:
        for _ in range(S_max):
            devs_ainda = [d for j2, d in zip(monoliths, devs_mono)
                          if s_mono[j2] > s_rr]
            cheio = any(n_por_dev[d][s_rr] >= 3 for d in devs_ainda)
            if devs_ainda and not cheio:
                break
            s_rr = s_rr + 1 if s_rr < s_warm_max else 1
        fillers = [d for d in range(M) if d not in devs_ainda]
        team = (devs_ainda + fillers)[:min(4, M)]
        combo = {4: (3, 3, 3, 3), 3: (3, 3, 6), 2: (6, 6),
                 1: (12,)}[len(team)]
        sa[k] = s_rr
        for d, f12 in zip(team, combo):
            fm[k][d] = f12 / 12.0
            n_por_dev[d][s_rr] += 1
        s_rr = s_rr + 1 if s_rr < s_warm_max else 1

    # Livres: solo, best dev, sprint com menos tasks desse dev
    for j in free_tasks:
        i = min(range(M), key=lambda d: T_sta[d][j])
        s = min(range(1, S_max + 1),
                key=lambda x: (n_por_dev[i][x] + (10 if any(
                    j2 for j2, d2 in zip(monoliths, devs_mono)
                    if d2 == i and s_mono[j2] == x) else 0)))
        sa[j] = s
        fm[j][i] = 1.0
        n_por_dev[i][s] += 1

    # Fixup exato: movimentos unitários que reduzem pen_cap
    av = evaluate(inst, sa, fm)
    for _ in range(3 * N):
        if av.pen_cap < EPS and av.pen_window < EPS:
            break
        improved_fix = False
        for j in sorted(range(N), key=lambda x: -inst["T_nom"][x]):
            if j in s_mono:
                continue
            for s_c in range(1, S_max + 1):
                if s_c == sa[j]:
                    continue
                sa_t = list(sa)
                sa_t[j] = s_c
                av_t = evaluate(inst, sa_t, fm)
                if (av_t.pen_cap + av_t.pen_window
                        < av.pen_cap + av.pen_window - EPS):
                    sa, av = sa_t, av_t
                    improved_fix = True
                    break
            if improved_fix:
                break
        if not improved_fix:
            break

    monos_str = ", ".join(f"T{j}->Dev_{i}@s{s_mono[j]}"
                          for j, i in zip(monoliths, devs_mono))
    log.to_file(f"[gulosa-monoliths v2] monos=[{monos_str}] "
                f"warmups={len(warmup_pool)} free_tasks={len(free_tasks)} "
                f"pen_cap={av.pen_cap:.1f} pen_jan={av.pen_window:.1f}")
    sa, fm, av = iterative_repair(inst, sa, fm, log)
    return sa, fm, av


# ============================================================================
# 4. OPERADORES DE VIZINHANÇA N1–N8 (shaking)
# ============================================================================

def op_sprint_move(inst, sa, fm, Dur_curr):
    """N1: move 1 task aleatória para outra sprint aleatória."""
    N, S_max = inst["N"], inst["S_max"]
    j = random.randrange(N)
    cands = [s for s in range(1, S_max + 1) if s != sa[j]]
    if not cands:
        return None
    sa_n = list(sa)
    sa_n[j] = random.choice(cands)
    sa_n = repair_topologico(inst, sa_n, Dur_curr, fm)
    return sa_n, [row[:] for row in fm]


def op_dev_swap(inst, sa, fm):
    """N2: troca devs exclusivos entre 2 tasks da mesma sprint."""
    N, M = inst["N"], inst["M"]
    por_sprint = {}
    for j in range(N):
        por_sprint.setdefault(sa[j], []).append(j)
    cands = [t for t in por_sprint.values() if len(t) >= 2]
    if not cands:
        return None
    j1, j2 = random.sample(random.choice(cands), 2)
    t1 = [i for i in range(M) if fm[j1][i] > EPS]
    t2 = [i for i in range(M) if fm[j2][i] > EPS]
    ex1 = [i for i in t1 if i not in t2]
    ex2 = [i for i in t2 if i not in t1]
    if not ex1 or not ex2:
        return None
    i1, i2 = random.choice(ex1), random.choice(ex2)
    f1, f2 = fm[j1][i1], fm[j2][i2]
    fm_n = [row[:] for row in fm]
    fm_n[j1][i1] = 0.0; fm_n[j1][i2] = f1
    fm_n[j2][i2] = 0.0; fm_n[j2][i1] = f2
    if not (is_valid_frac(fm_n[j1]) and is_valid_frac(fm_n[j2])):
        return None
    return list(sa), fm_n


def op_frac_shift(inst, sa, fm):
    """N3: muda a combinação de frações de 1 task para outra válida."""
    N, M = inst["N"], inst["M"]
    js = list(range(N))
    random.shuffle(js)
    for j in js:
        atual = _twelfths(fm[j])
        alts = [c for c in VALID_COMBOS
                if tuple(sorted(c)) != atual and len(c) <= M]
        random.shuffle(alts)
        for alt in alts:
            devs = random.sample(range(M), len(alt))
            fm_n = [row[:] for row in fm]
            fm_n[j] = [0.0] * M
            combo = list(alt)
            random.shuffle(combo)
            for d_idx, f12 in zip(devs, combo):
                fm_n[j][d_idx] = f12 / 12.0
            if is_valid_frac(fm_n[j]):
                return list(sa), fm_n
    return None


def op_dev_reassign(inst, sa, fm):
    """N4: remove 1 dev de uma task multi-dev e o adiciona a outra
    task da mesma sprint, redistribuindo frações válidas em ambas."""
    N, M = inst["N"], inst["M"]
    por_sprint = {}
    for j in range(N):
        por_sprint.setdefault(sa[j], []).append(j)
    js = list(range(N))
    random.shuffle(js)
    for j1 in js:
        t1 = [i for i in range(M) if fm[j1][i] > EPS]
        if len(t1) < 2:
            continue
        outros = [j for j in por_sprint.get(sa[j1], []) if j != j1]
        if not outros:
            continue
        i_rem = random.choice(t1)
        resto = [i for i in t1 if i != i_rem]
        combos1 = [c for c in VALID_COMBOS if len(c) == len(resto)]
        if not combos1:
            continue
        fm_n = [row[:] for row in fm]
        fm_n[j1] = [0.0] * M
        c1 = list(random.choice(combos1))
        random.shuffle(c1)
        for d_idx, f12 in zip(resto, c1):
            fm_n[j1][d_idx] = f12 / 12.0
        if not is_valid_frac(fm_n[j1]):
            continue
        j2 = random.choice(outros)
        t2 = [i for i in range(M) if fm_n[j2][i] > EPS]
        if i_rem in t2:
            continue
        novo_time = t2 + [i_rem]
        combos2 = [c for c in VALID_COMBOS if len(c) == len(novo_time)]
        if not combos2:
            continue
        fm_n[j2] = [0.0] * M
        c2 = list(random.choice(combos2))
        random.shuffle(c2)
        for d_idx, f12 in zip(novo_time, c2):
            fm_n[j2][d_idx] = f12 / 12.0
        if is_valid_frac(fm_n[j2]):
            return list(sa), fm_n
    return None


def op_sprint_swap(inst, sa, fm, Dur_curr):
    """N5: troca as sprints de 2 tasks."""
    N = inst["N"]
    if N < 2:
        return None
    j1, j2 = random.sample(range(N), 2)
    if sa[j1] == sa[j2]:
        return None
    sa_n = list(sa)
    sa_n[j1], sa_n[j2] = sa_n[j2], sa_n[j1]
    sa_n = repair_topologico(inst, sa_n, Dur_curr, fm)
    return sa_n, [row[:] for row in fm]


def N6_lns_scatter(inst, sa, fm, Dur_curr, p):
    """N6/N8: LNS — destrói p% das tasks (sprint aleatório >= mínimo
    topológico + dev solo aleatório) e reconstrói por repair."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    n_destroy = max(2, int(N * p))
    if inst["comp_weights"] is not inst["rem_path"]:
        # sem precedências: destruição uniforme
        target = set(random.sample(range(N), n_destroy))
    else:
        # com precedências: destruição PONDERADA por criticidade — os ganhos
        # do C3 vêm de reconstruir a chain crítica e seus fillers, não
        # de re-sortear tasks folgadas (ciclo 9: 414 draws uniformes sem
        # nenhum acerto; o breakthrough k=6 é loteria de qual conjunto cai)
        weights = [inst["rem_path"][j] for j in range(N)]
        target = set()
        attempts = 0
        while len(target) < n_destroy and attempts < 20 * n_destroy:
            target.add(random.choices(range(N), weights=weights, k=1)[0])
            attempts += 1
    sa_n = list(sa)
    fm_n = [row[:] for row in fm]
    for j in inst["topo"]:
        if j not in target:
            continue
        s_min = max((sa_n[k] for k in inst["Pred"][j]), default=1)
        # janela ±3 em torno da sprint atual: diversifica sem despedaçar
        # chains de precedência (sprints aleatórias free_tasks geravam
        # candidatos irreparáveis no Cenário 3)
        lo = max(s_min, sa_n[j] - 3)
        hi = min(S_max, sa_n[j] + 3)
        sa_n[j] = random.randint(lo, max(lo, hi))
        dev = random.randrange(M)
        fm_n[j] = [0.0] * M
        fm_n[j][dev] = 1.0
    sa_n = repair_topologico(inst, sa_n, Dur_curr, fm_n)
    return sa_n, fm_n


def N7_block_move(inst, sa, fm, Dur_curr):
    """N7: move todas as tasks de uma sprint para sprint vizinha (±1/±2)."""
    N, S_max = inst["N"], inst["S_max"]
    ocupadas = list({sa[j] for j in range(N)})
    if not ocupadas:
        return None
    s_orig = random.choice(ocupadas)
    s_dest = max(1, min(S_max, s_orig + random.choice([-2, -1, 1, 2])))
    if s_dest == s_orig:
        return None
    sa_n = list(sa)
    for j in range(N):
        if sa_n[j] == s_orig:
            sa_n[j] = s_dest
    sa_n = repair_topologico(inst, sa_n, Dur_curr, fm)
    return sa_n, [row[:] for row in fm]


def N8_team_reshuffle(inst, sa, fm):
    """N8 (parte 2): re-sorteia combinações válidas de todas as tasks
    de uma sprint aleatória."""
    N, M = inst["N"], inst["M"]
    ocupadas = list({sa[j] for j in range(N)})
    if not ocupadas:
        return None
    s_target = random.choice(ocupadas)
    combos = [c for c in VALID_COMBOS if len(c) <= M]
    fm_n = [row[:] for row in fm]
    for j in range(N):
        if sa[j] != s_target:
            continue
        combo = list(random.choice(combos))
        devs = random.sample(range(M), len(combo))
        random.shuffle(combo)
        fm_n[j] = [0.0] * M
        for d_idx, f12 in zip(devs, combo):
            fm_n[j][d_idx] = f12 / 12.0
    return list(sa), fm_n


def N9_axis_warmup(inst, sa, fm, Dur_curr, j_mono=None, i_target=None,
                   s_mono=None, devs_warmup=None, n_warmup=None):
    """N9: template "aquecimento de axis + monolith tardio" (padrão Gurobi).
    Sorteia uma task grande (top N9_TOP_FRAC por T_nom), um dev com bom
    T_sta para ela, move-a para uma sprint tardia SOLO com esse dev e
    insere o dev (fração 3/12) em até n_warmup tasks do mesmo axis
    em sprints anteriores — acumulando B_apr para o monolith caber.
    Os parâmetros opcionais permitem aplicação DIRIGIDA (usada pela
    greedy_construction_monoliths): target, dev, sprint e team de warmup
    fixados em vez de sorteados."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    T_nom = inst["T_nom"]
    if n_warmup is None:
        n_warmup = cfg.N9_WARMUP_N

    if j_mono is None:
        n_top = max(1, int(N * cfg.N9_TOP_FRAC))
        top_tasks = sorted(range(N), key=lambda j: -T_nom[j])[:n_top]
        j_mono = random.choice(top_tasks)
    if not inst["axis"][j_mono]:
        return None

    if i_target is None:
        # Dev: um dos 2 best_ones T_sta para o monolith
        devs_rank = sorted(range(M), key=lambda d: inst["T_sta"][d][j_mono])
        i_target = random.choice(devs_rank[:min(2, M)])

    s_min = max((sa[k] for k in inst["Pred"][j_mono]), default=1)
    if s_mono is None:
        if max(s_min + 1, 3) > S_max:
            return None
        s_mono = random.randint(max(s_min + 1, 3), S_max)
    s_mono = max(s_mono, s_min + 1)
    if s_mono > S_max:
        return None

    sa_n = list(sa)
    fm_n = [row[:] for row in fm]
    sa_n[j_mono] = s_mono
    fm_n[j_mono] = [0.0] * M
    fm_n[j_mono][i_target] = 1.0

    # Aquecimento: tasks axis viram teams de 4 devs (frações 3/12 cada,
    # como o Gurobi faz no Cenário 2) incluindo i_target, em sprints
    # anteriores ao monolith — cada dev carrega só ~25% do esforço, então
    # as sprints iniciais não estouram e i_target acumula B_apr.
    n_warm_devs = min(4, M)
    combo_warm = {4: (3, 3, 3, 3), 3: (3, 3, 6), 2: (6, 6)}.get(n_warm_devs)
    if combo_warm is None:
        return None
    cands = [k for k in inst["axis"][j_mono]
             if k != j_mono and fm_n[k][i_target] < EPS]
    random.shuffle(cands)
    inseridos = 0
    for k in cands:
        if inseridos >= n_warmup:
            break
        if inst["T_nom"][k] >= inst["T_nom"][j_mono]:
            continue  # warmup só com tasks menores que o monolith
        s_k_min = max((sa_n[p] for p in inst["Pred"][k]), default=1)
        if s_k_min >= s_mono:
            continue
        if devs_warmup is not None:
            devs_k = list(devs_warmup)[:n_warm_devs]
            if i_target not in devs_k:
                devs_k = [i_target] + devs_k[:n_warm_devs - 1]
        else:
            outros = [d for d in range(M) if d != i_target]
            random.shuffle(outros)
            devs_k = [i_target] + outros[:n_warm_devs - 1]
        fm_n[k] = [0.0] * M
        for d_idx, f12 in zip(devs_k, combo_warm):
            fm_n[k][d_idx] = f12 / 12.0
        sa_n[k] = random.randint(s_k_min, s_mono - 1)
        inseridos += 1

    if inseridos == 0:
        return None
    sa_n = repair_topologico(inst, sa_n, Dur_curr, fm_n)
    return sa_n, fm_n


def N10_chain_specialist(inst, sa, fm, Dur_curr):
    """N10: template "specialist de chain" (padrão Gurobi do Cenário 3).
    Sorteia um caminho no grafo de precedência (3..N10_PATH_MAX tasks),
    atribui TODAS as tasks do caminho SOLO ao dev de menor T_sta total e
    compacta ~N10_PER_SPRINT tasks por sprint — o dev acumula b_dep (0.2
    por predecessora direta) e vira specialist (F_total ~0.7-0.9)."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    com_arestas = [j for j in range(N) if inst["Pred"][j] or inst["succs"][j]]
    if not com_arestas:
        return None
    j0 = random.choice(com_arestas)
    caminho = [j0]
    cur = j0
    while inst["Pred"][cur] and len(caminho) < cfg.N10_PATH_MAX:
        cur = random.choice(inst["Pred"][cur])
        caminho.insert(0, cur)
    cur = j0
    while inst["succs"][cur] and len(caminho) < cfg.N10_PATH_MAX:
        cur = random.choice(inst["succs"][cur])
        caminho.append(cur)
    if len(caminho) < 3:
        return None

    devs_rank = sorted(range(M),
                       key=lambda d: sum(inst["T_sta"][d][j] for j in caminho))
    i_target = random.choice(devs_rank[:min(2, M)])

    sa_n = list(sa)
    fm_n = [row[:] for row in fm]
    s_base = max(1, min(sa_n[j] for j in caminho))
    for idx, j in enumerate(caminho):
        fm_n[j] = [0.0] * M
        fm_n[j][i_target] = 1.0
        sa_n[j] = min(S_max, s_base + idx // cfg.N10_PER_SPRINT)
    sa_n = repair_topologico(inst, sa_n, Dur_curr, fm_n)
    return sa_n, fm_n


def shaking(inst, sa, fm, k, Dur_curr):
    """k=1..5: (k-1) sprint-moves aleatórios + operador Nk.
    k=6: LNS 25%. k=7: block-move. k=8: LNS 40% + team-reshuffle.
    k=9: aquecimento de axis + monolith tardio (template Gurobi C2).
    k=10: specialist de chain (template Gurobi C3)."""
    N, S_max = inst["N"], inst["S_max"]
    sa_n = list(sa)
    fm_n = [row[:] for row in fm]

    if k <= 5:
        if k > 1:
            for j in random.sample(range(N), min(k - 1, N)):
                sa_n[j] = random.randint(1, S_max)
            sa_n = repair_topologico(inst, sa_n, Dur_curr, fm_n)
        ops = {
            1: lambda: op_sprint_move(inst, sa_n, fm_n, Dur_curr),
            2: lambda: op_dev_swap(inst, sa_n, fm_n),
            3: lambda: op_frac_shift(inst, sa_n, fm_n),
            4: lambda: op_dev_reassign(inst, sa_n, fm_n),
            5: lambda: op_sprint_swap(inst, sa_n, fm_n, Dur_curr),
        }
        r = ops[k]()
        return r if r else (sa_n, fm_n)
    elif k == 6:
        r = N6_lns_scatter(inst, sa_n, fm_n, Dur_curr, cfg.LNS_P_K6)
        return r if r else (sa_n, fm_n)
    elif k == 7:
        r = N7_block_move(inst, sa_n, fm_n, Dur_curr)
        return r if r else (sa_n, fm_n)
    elif k == 8:
        r6 = N6_lns_scatter(inst, sa_n, fm_n, Dur_curr, cfg.LNS_P_K8)
        if r6:
            sa_n, fm_n = r6
        r8 = N8_team_reshuffle(inst, sa_n, fm_n)
        return r8 if r8 else (sa_n, fm_n)
    elif k == 9:
        r = N9_axis_warmup(inst, sa_n, fm_n, Dur_curr)
        return r if r else (sa_n, fm_n)
    else:  # k == 10
        r = N10_chain_specialist(inst, sa_n, fm_n, Dur_curr)
        return r if r else (sa_n, fm_n)


# ============================================================================
# 5. VND — L1 sprint-move, L2 dev-swap, L3 frac-shift, L4 team-reshuffle
# ============================================================================

def vnd(inst, sa_in, fm_in, deadline, vnd_log, vns_iter, log):
    """Variable Neighborhood Descent com first-improvement.
    Registra a convergência INTERNA em vnd_log (lista de dicts):
      evento 'entrada'/'melhoria'/'saida', l, iter_interna, fp, tmax,
      penalidades e tempo. Interrompe se time.time() >= deadline."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    sa = list(sa_in)
    fm = [row[:] for row in fm_in]
    av = evaluate(inst, sa, fm)
    fp = penalized_objective(av)
    Dur_curr = list(av.Dur)
    t0 = time.time()

    iter_interna = 0
    improvements_by_l = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    def record(evento, l):
        vnd_log.append({
            "vns_iter": vns_iter, "evento": evento, "l": l,
            "iter_interna": iter_interna,
            "fp": round(fp, 4), "tmax": round(av.Tmax, 4),
            "pen_cap": round(av.pen_cap, 4),
            "pen_jan": round(av.pen_window, 4),
            "pen_tsk": round(av.pen_tasks, 4),
            "tempo_s": round(time.time() - t0, 3),
        })

    record("entrada", 0)

    l = 1
    while l <= 5:
        if time.time() >= deadline:
            record("timeout", l)
            break
        improved = False
        iter_interna += 1

        if l == 1:
            # L1: mover 1 task de sprint (first-improvement). Para cortar o
            # cost da prova de ótimo local, só tenta sprints a ±L1_WINDOW da
            # atual + 1 sprint aleatória distante (cobre compressões longas).
            js = list(range(N))
            random.shuffle(js)
            for j in js:
                if improved:
                    break
                s_atual = sa[j]
                sprints = [s for s in range(max(1, s_atual - cfg.L1_WINDOW),
                                            min(S_max, s_atual + cfg.L1_WINDOW) + 1)
                           if s != s_atual]
                s_rand = random.randint(1, S_max)
                if s_rand != s_atual and s_rand not in sprints:
                    sprints.append(s_rand)
                random.shuffle(sprints)
                for s_novo in sprints:
                    sa_t = list(sa)
                    sa_t[j] = s_novo
                    sa_t = repair_topologico(inst, sa_t, Dur_curr, fm)
                    av_t = evaluate(inst, sa_t, fm)
                    fp_t = penalized_objective(av_t)
                    if fp_t < fp - EPS:
                        sa, av, fp = sa_t, av_t, fp_t
                        Dur_curr = list(av.Dur)
                        improved = True
                        break

        elif l == 2:
            # L2: trocar devs exclusivos entre pares da mesma sprint
            por_sprint = {}
            for j in range(N):
                por_sprint.setdefault(sa[j], []).append(j)
            pares = [(t[a], t[b]) for t in por_sprint.values()
                     for a in range(len(t)) for b in range(a + 1, len(t))]
            random.shuffle(pares)
            for (j1, j2) in pares:
                if improved:
                    break
                t1 = [i for i in range(M) if fm[j1][i] > EPS]
                t2 = [i for i in range(M) if fm[j2][i] > EPS]
                ex1 = [i for i in t1 if i not in t2]
                ex2 = [i for i in t2 if i not in t1]
                if not ex1 or not ex2:
                    continue
                for i1 in ex1:
                    if improved:
                        break
                    for i2 in ex2:
                        fm_t = [row[:] for row in fm]
                        f1, f2 = fm_t[j1][i1], fm_t[j2][i2]
                        fm_t[j1][i1] = 0.0; fm_t[j1][i2] = f1
                        fm_t[j2][i2] = 0.0; fm_t[j2][i1] = f2
                        if not (is_valid_frac(fm_t[j1])
                                and is_valid_frac(fm_t[j2])):
                            continue
                        av_t = evaluate(inst, sa, fm_t)
                        fp_t = penalized_objective(av_t)
                        if fp_t < fp - EPS:
                            fm, av, fp = fm_t, av_t, fp_t
                            Dur_curr = list(av.Dur)
                            improved = True
                            break

        elif l == 3:
            # L3: trocar combinação de frações de tasks sampled
            sample = random.sample(
                range(N), min(N, max(cfg.L3_MIN_SAMPLE,
                                     int(N * cfg.L3_SAMPLE_FRAC))))
            for j in sample:
                if improved:
                    break
                atual = _twelfths(fm[j])
                alts = [c for c in VALID_COMBOS
                        if tuple(sorted(c)) != atual and len(c) <= M]
                random.shuffle(alts)
                for alt in alts:
                    if improved:
                        break
                    for _ in range(min(8, math.factorial(min(len(alt), 4)))):
                        devs = random.sample(range(M), len(alt))
                        fm_t = [row[:] for row in fm]
                        fm_t[j] = [0.0] * M
                        combo = list(alt)
                        random.shuffle(combo)
                        for d_idx, f12 in zip(devs, combo):
                            fm_t[j][d_idx] = f12 / 12.0
                        if not is_valid_frac(fm_t[j]):
                            continue
                        av_t = evaluate(inst, sa, fm_t)
                        fp_t = penalized_objective(av_t)
                        if fp_t < fp - EPS:
                            fm, av, fp = fm_t, av_t, fp_t
                            Dur_curr = list(av.Dur)
                            improved = True
                            break

        elif l == 4:
            # L4: reshuffle ALEATÓRIO das teams de uma sprint inteira
            # (não-greedy de propósito — escapa de mínimos locais)
            sprints = list(range(1, S_max + 1))
            random.shuffle(sprints)
            combos = [c for c in VALID_COMBOS if len(c) <= M]
            for s_target in sprints:
                if improved:
                    break
                tasks_s = [j for j in range(N) if sa[j] == s_target]
                if not tasks_s:
                    continue
                for _ in range(cfg.L4_ATTEMPTS):
                    if improved:
                        break
                    fm_t = [row[:] for row in fm]
                    for j in tasks_s:
                        combo = list(random.choice(combos))
                        devs = random.sample(range(M), len(combo))
                        random.shuffle(combo)
                        fm_t[j] = [0.0] * M
                        for d_idx, f12 in zip(devs, combo):
                            fm_t[j][d_idx] = f12 / 12.0
                    av_t = evaluate(inst, sa, fm_t)
                    fp_t = penalized_objective(av_t)
                    if fp_t < fp - EPS:
                        fm, av, fp = fm_t, av_t, fp_t
                        Dur_curr = list(av.Dur)
                        improved = True

        elif l == 5:
            # L5: "des-equipar" — tenta converter cada task multi-dev em
            # SOLO (testa todos os devs). Barato e direto: teams parasitas
            # custam P_com + P_match (~2.5) e raramente compensam fora dos
            # casos estruturais (ver Gurobi C3: 49 solo de 50 tasks).
            # DESLIGADA em instâncias com monoliths: lá as teams de
            # aquecimento são necessárias (C2) e o L5 as desmontava no meio
            # da reconstrução, antes de o benefício do B_apr aparecer.
            multi = ([] if inst["has_monoliths"] else
                     [j for j in range(N)
                      if sum(1 for i in range(M) if fm[j][i] > EPS) >= 2])
            random.shuffle(multi)
            for j in multi:
                if improved:
                    break
                devs = list(range(M))
                random.shuffle(devs)
                for i_solo in devs:
                    fm_t = [row[:] for row in fm]
                    fm_t[j] = [0.0] * M
                    fm_t[j][i_solo] = 1.0
                    av_t = evaluate(inst, sa, fm_t)
                    fp_t = penalized_objective(av_t)
                    if fp_t < fp - EPS:
                        fm, av, fp = fm_t, av_t, fp_t
                        Dur_curr = list(av.Dur)
                        improved = True
                        break

        if improved:
            improvements_by_l[l] += 1
            record("melhoria", l)
            log.verbose(f"[VND it{vns_iter}] melhoria L{l}: fp={fp:.2f} "
                        f"tmax={av.Tmax:.1f} pen_cap={av.pen_cap:.2f}")
            l = 1
        else:
            l += 1

    record("saida", 0)
    return sa, fm, av, fp, improvements_by_l


# ============================================================================
# 6. GATE DE ACEITAÇÃO — verificação integral das 43 restrições
# ============================================================================

def incumbent_improves(av_novo, av_atual):
    """Critério LEXICOGRÁFICO de aceitação do incumbent (ambos factíveis):
    1º Tmax real menor; 2º mesmo Tmax mas cronograma mais comprimido (comp).
    O desempate por compressão permite atravessar platôs (mover tasks para
    mais cedo mesmo sem reduzir o Tmax ainda) sem NUNCA piorar o Tmax."""
    if av_novo.Tmax < av_atual.Tmax - 1e-6:
        return True
    return (av_novo.Tmax < av_atual.Tmax + 1e-6
            and av_novo.comp < av_atual.comp - 1e-6)


def acceptance_gate(inst, sa_cand, fm_cand, av_cand, log, vns_iter):
    """Submete a solução candidata à verificação INTEGRAL das 43 restrições.
    Se infactível, tenta o repair complete e re-verifica. Retorna
    (accepts_feasible, sa, fm, av, chk)."""
    chk = vc.verify_all(inst, sa_cand, fm_cand, av_cand)
    if chk["feasible"]:
        return True, sa_cand, fm_cand, av_cand, chk

    # Tenta reparar e re-verificar
    sa_r, fm_r = full_repair(inst, sa_cand, fm_cand, log)
    av_r = evaluate(inst, sa_r, fm_r)
    chk_r = vc.verify_all(inst, sa_r, fm_r, av_r)
    if chk_r["feasible"]:
        log.to_file(f"[gate it{vns_iter}] candidato repaired com sucesso "
                    f"(antes: {chk['summary_str']})")
        return True, sa_r, fm_r, av_r, chk_r

    return False, sa_r, fm_r, av_r, chk_r


# ============================================================================
# 6b. MATHEURÍSTICA — recorte de subproblemas + reotimização exata (Gurobi)
# ============================================================================

def _finish(av, j):
    return av.Inic[j] + av.Dur[j]


def _pad_free_set(inst, sa, av, base, size):
    """Completa/limita a lista `base` até `size` tasks únicas, preenchendo com
    as de término mais tardio ainda não incluídas (mantém o subproblema com
    um tamanho útil)."""
    seen = []
    seset = set()
    for j in base:
        if j not in seset:
            seset.add(j); seen.append(j)
    if len(seen) >= size:
        return seen[:size]
    extra = sorted((j for j in range(inst["N"]) if j not in seset),
                   key=lambda j: -_finish(av, j))
    for j in extra:
        if len(seen) >= size:
            break
        seen.append(j)
    return seen


def select_free_set(inst, sa, fm, av, strategy, size):
    """Escolhe as tasks LIVRES do subproblema (as demais ficam fixas).
    Cada estratégia mapeia uma decomposição (Seção 5.2 do plano v5)."""
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    size = max(2, min(size, N))

    if strategy == "makespan_tail":
        base = sorted(range(N), key=lambda j: -_finish(av, j))

    elif strategy == "bottleneck_dev":
        load = [0.0] * M
        for j in range(N):
            for i in range(M):
                if fm[j][i] > EPS:
                    load[i] += av.T_din[i][j]
        i_bn = max(range(M), key=lambda i: load[i])
        base = sorted((j for j in range(N) if fm[j][i_bn] > EPS),
                      key=lambda j: -_finish(av, j))

    elif strategy == "critical_chain":
        j0 = max(range(N), key=lambda j: _finish(av, j))
        chain = [j0]
        cur = j0
        while inst["Pred"][cur] and len(chain) < size:
            cur = max(inst["Pred"][cur], key=lambda p: inst["rem_path"][p])
            if cur in chain:
                break
            chain.insert(0, cur)
        cur = j0
        while inst["succs"][cur] and len(chain) < size:
            cur = max(inst["succs"][cur], key=lambda s2: inst["rem_path"][s2])
            if cur in chain:
                break
            chain.append(cur)
        base = chain

    elif strategy == "monolith_axis":
        monos = [j for j in range(N)
                 if min(inst["T_sta"][i][j] for i in range(M)) + cfg.SETUP_HOURS
                 > cfg.CAP_UTIL]
        if not monos:
            monos = sorted(range(N), key=lambda j: -inst["T_nom"][j])[:1]
        seed = max(monos, key=lambda j: inst["T_nom"][j])
        base = [seed] + list(inst["axis"][seed])

    elif strategy == "window":
        # C-104: mira a sprint que DEFINE o makespan (a da tarefa que termina por
        # ÚLTIMO), não a de maior carga (que costuma ser uma sprint INICIAL) — é a
        # janela cuja compressão de fato reduz o Tmax.
        j_last = max(range(N), key=lambda j: _finish(av, j))
        s_hot = sa[j_last]
        win = {s_hot - 1, s_hot, s_hot + 1}
        base = sorted((j for j in range(N) if sa[j] in win),
                      key=lambda j: -_finish(av, j))
        if not base:
            base = sorted(range(N), key=lambda j: -_finish(av, j))

    elif strategy == "related":
        seed = max(range(N), key=lambda j: _finish(av, j))
        team_seed = [i for i in range(M) if fm[seed][i] > EPS]
        related = set([seed]) | set(inst["axis"][seed]) | set(inst["Pred"][seed]) \
            | set(inst["succs"][seed])
        for j in range(N):
            if sa[j] == sa[seed] or any(fm[j][i] > EPS for i in team_seed):
                related.add(j)
        base = sorted(related, key=lambda j: -_finish(av, j))

    elif strategy == "random_scatter":
        # C-102: destroy LNS ALEATÓRIO disperso (Shaw 1998; Pisinger&Ropke 2010)
        # — diversifica a camada exata (as outras 6 estratégias são gulosas p/
        # cauda/gargalo). RNG LOCAL seedado pelo incumbente: reprodutível e NÃO
        # consome o stream global (não perturba a trajetória do GVNS).
        _rng = random.Random(int(round(av.Tmax * 1000)) * 131 + size)
        base = _rng.sample(range(N), min(max(2, size), N))

    elif strategy == "precedence_chain_repair":
        # C-107: "joelhos" da cadeia (Fase 1) — tarefas da cadeia crítica cujo dev
        # atribuído tem B_apr BAIXO (não herdou aprendizado das predecessoras →
        # F_total alto). Liberá-las (ordenadas por B_apr crescente) deixa o
        # sub-MIP reatribuí-las ao especialista da cadeia, baixando o F_total.
        j0 = max(range(N), key=lambda j: _finish(av, j))
        chain = [j0]
        cur = j0
        while inst["Pred"][cur] and len(chain) < N:
            cur = max(inst["Pred"][cur], key=lambda p: inst["rem_path"][p])
            if cur in chain:
                break
            chain.insert(0, cur)

        def _bapr_min(j):
            team = [i for i in range(M) if fm[j][i] > EPS]
            return min((av.B_apr[i][j] for i in team), default=0.0)
        base = sorted(chain, key=_bapr_min)

    elif strategy == "dev_rolling":
        # C-108: decomposição por recurso COMPLEMENTAR — libera as tarefas do 2º
        # dev mais carregado (bottleneck_dev pega o 1º); ao longo das passadas
        # cobre mais recursos. Aditiva.
        load = [0.0] * M
        for j in range(N):
            for i in range(M):
                if fm[j][i] > EPS:
                    load[i] += av.T_din[i][j]
        devs_by_load = sorted(range(M), key=lambda i: -load[i])
        i_sel = devs_by_load[1] if M > 1 else devs_by_load[0]
        base = sorted((j for j in range(N) if fm[j][i_sel] > EPS),
                      key=lambda j: -_finish(av, j))

    elif strategy == "rolling_horizon":
        # C-109: janela de sprints DESLIZANTE (Palpant et al. 2004) — libera um
        # bloco contíguo de 3 sprints cuja posição rotaciona com o incumbente.
        s_start = (int(round(av.Tmax)) % max(1, S_max - 2)) + 1
        win = {s_start, s_start + 1, s_start + 2}
        base = sorted((j for j in range(N) if sa[j] in win),
                      key=lambda j: -_finish(av, j))
        if not base:
            base = sorted(range(N), key=lambda j: -_finish(av, j))

    else:
        base = sorted(range(N), key=lambda j: -_finish(av, j))

    return _pad_free_set(inst, sa, av, base, size)


def mip_reoptimize(inst, sa, fm, av, strategy, time_limit, log, tag,
                   free_size=None, radius=None, dev_cand=None):
    """Recorta um subproblema (free-set da `strategy`) e o resolve com o
    Gurobi até a otimalidade (no contexto global). Devolve
    (sa2, fm2, free, obj_gurobi, elapsed, n_triples, n_v) ou None.
    Os knobs (free_size/radius/dev_cand) são PRECEDENCE-AWARE (passados pelo
    solve_vns). NÃO aplica o gate — quem chama é responsável por isso."""
    free_size = cfg.MH_FREE_SET_SIZE if free_size is None else free_size
    radius = cfg.MH_SPRINT_RADIUS if radius is None else radius
    dev_cand = cfg.MH_DEV_CANDIDATES if dev_cand is None else dev_cand
    free = select_free_set(inst, sa, fm, av, strategy, free_size)
    if not free:
        return None
    t0 = time.time()
    model, refs = vc.build_reduced_model(
        inst, sa, fm, free, radius, dev_cand,
        time_limit, cfg.MH_SUBMIP_GAP, cfg.MH_GUROBI_THREADS, cfg.MH_GUROBI_SEED)
    model.optimize()
    status, ncount = model.Status, model.SolCount
    obj = model.ObjVal if ncount > 0 else None
    n_tr, n_v = refs["n_triples"], refs["n_v"]
    sa2 = fm2 = None
    if ncount > 0:
        sa2, fm2 = vc.extract_solution(inst, sa, fm, model, refs)
    model.dispose()
    refs["env"].dispose()
    elapsed = time.time() - t0
    log.to_file(f"[mip {tag}] strat={strategy} free={len(free)} "
                f"triples={n_tr} v={n_v} status={status} "
                f"objGurobi={obj if obj is None else round(obj, 1)} "
                f"t={elapsed:.1f}s")
    if sa2 is None:
        return None
    return sa2, fm2, free, obj, elapsed, n_tr, n_v


def fix_and_optimize_sweep(inst, sa, fm, av, t_deadline, log, on_improve, tag,
                           free_size=None, radius=None, dev_cand=None,
                           submip_time=None):
    """Varredura POPMUSIC: roda cada estratégia em sequência sobre o
    incumbente; aceita o resultado SÓ se passar no gate das 43 restrições e
    melhorar (Tmax do evaluate()). Repete até uma passada inteira sem melhora
    ou o orçamento (t_deadline) acabar. Devolve (sa, fm, av, n_improvements)."""
    submip_time = cfg.MH_SUBMIP_TIME if submip_time is None else submip_time
    cur_sa, cur_fm, cur_av = list(sa), [r[:] for r in fm], av
    n_imp = 0
    for _pass in range(cfg.MH_SWEEP_MAX_PASSES):
        improved = False
        for strat in cfg.MH_STRATEGIES:
            remaining = t_deadline - time.time()
            if remaining < 1.0:
                return cur_sa, cur_fm, cur_av, n_imp
            tl = min(submip_time, remaining)
            r = mip_reoptimize(inst, cur_sa, cur_fm, cur_av, strat, tl, log,
                               f"{tag}-sweep", free_size, radius, dev_cand)
            if r is None:
                continue
            sa2, fm2 = r[0], r[1]
            av2 = evaluate(inst, sa2, fm2)
            ok, sa_a, fm_a, av_a, chk = acceptance_gate(
                inst, sa2, fm2, av2, log, tag)
            if ok and incumbent_improves(av_a, cur_av):
                cur_sa, cur_fm, cur_av = sa_a, fm_a, av_a
                improved = True
                n_imp += 1
                on_improve(strat, cur_av)
            elif not ok:
                log.to_file(f"[mip {tag}-sweep] {strat} REJEITADO pelo gate: "
                            f"{chk['summary_str']}")
        if not improved:
            break
    return cur_sa, cur_fm, cur_av, n_imp


# ============================================================================
# 7. FUNÇÃO PRINCIPAL
# ============================================================================

def solve_vns(file_path, t_limit_override=None, show_console=True):
    if cfg.RANDOM_SEED is not None:
        random.seed(cfg.RANDOM_SEED)

    inst = vc.load_instance(file_path)
    name = inst["name"]
    N, M, S_max = inst["N"], inst["M"], inst["S_max"]
    LB = inst["LB"]
    T_limit = (t_limit_override if t_limit_override is not None
               else compute_time_limit(N, M, S_max))
    # Orçamento PRECEDENCE-AWARE (E-038): instâncias com precedências (cadeias)
    # são as mais difíceis e recebem um boost de tempo (≈ orçamento da v1), pois
    # o GVNS precisa de mais tempo para achar a bacia de 7 sprints. Não se aplica
    # quando o tempo é forçado externamente (t_limit_override).
    if (cfg.MH_ENABLED and t_limit_override is None
            and any(inst["Pred"][j] for j in range(N))):
        T_limit *= cfg.MH_PRED_TIME_BOOST

    out_dir = os.path.join("data", "results", cfg.RESULTS_SUBDIR, name)
    log = DebugLog(os.path.join(out_dir, "debug_log.txt"),
                   show_console=show_console)

    def gap_de(tmax):
        return (tmax - LB) / LB * 100.0 if LB > 0 else float("inf")

    # ------------------------------------------------------------------
    # Cabeçalho: instância + TODOS os parâmetros em uso
    # ------------------------------------------------------------------
    log.section(f"VNS — {name}")
    log.console(f"[VNS] N={N} M={M} S_max={S_max} | LB={LB:.1f}h | "
                f"T_limit={T_limit:.0f}s | seed={cfg.RANDOM_SEED}")
    log.block_to_file(
        "PARAMETROS DO MODELO: "
        f"alpha_axis={cfg.ALPHA_AXIS} lmbda_apr={cfg.LAMBDA_LEARN} "
        f"alpha_dep={cfg.ALPHA_DEP} lmbda_dep={cfg.LAMBDA_DEP} "
        f"l_fat={cfg.FATIGUE_THRESHOLD} phi_fat={cfg.FATIGUE_WEIGHT} "
        f"CapUtil={cfg.CAP_UTIL} H_span={cfg.H_SPAN} Setup={cfg.SETUP_HOURS}\n"
        "PARAMETROS DE BUSCA: "
        f"T_MULT={cfg.T_MULT} K_MAX={cfg.K_MAX} "
        f"ILS_AFTER={cfg.ILS_RESTART_AFTER} VND_FRAC={cfg.VND_TIME_FRAC} "
        f"F_PESS={cfg.F_TOTAL_PESS} "
        f"LAM=({cfg.LAM_CAP},{cfg.LAM_WINDOW},{cfg.LAM_TASKS})")

    t_start = time.time()
    vnd_log = []

    # ------------------------------------------------------------------
    # Solução inicial (multi-start) + verificação integral
    # ------------------------------------------------------------------
    starts = [("gulosa", greedy_construction(inst, log))]
    if any(inst["Pred"][j] for j in range(N)):
        starts.append(("gulosa-chains", greedy_construction_chains(inst, log)))
    if inst["has_monoliths"]:
        st_mono = greedy_construction_monoliths(inst, log)
        if st_mono is not None:
            starts.append(("gulosa-monoliths", st_mono))

    if len(starts) > 1:
        # VND curto em cada start; o best (factibilidade, fp) vence
        refinados = []
        for start_name, (sa_s, fm_s, av_s) in starts:
            dl = time.time() + cfg.MULTISTART_VND_S
            sa_s, fm_s, av_s, fp_s, _ = vnd(inst, sa_s, fm_s, dl,
                                            vnd_log, 0, log)
            pen_tot = av_s.pen_cap + av_s.pen_window + av_s.pen_tasks
            if pen_tot > EPS:
                # resíduos pequenos (ex.: 0.2h de capacity) são exatamente
                # o que o repair complete do gate resolve — starts quase
                # factíveis viram factíveis antes de competir
                sa_s, fm_s = full_repair(inst, sa_s, fm_s, log)
                av_s = evaluate(inst, sa_s, fm_s)
                fp_s = penalized_objective(av_s)
                pen_tot = av_s.pen_cap + av_s.pen_window + av_s.pen_tasks
            refinados.append((pen_tot > EPS, fp_s, start_name, sa_s, fm_s, av_s))
            log.console(f"   Start '{start_name}': Tmax={av_s.Tmax:.1f}h "
                        f"fp={fp_s:.1f} "
                        f"{'(infeasible)' if pen_tot > EPS else ''}")
        refinados.sort(key=lambda r: (r[0], r[1]))
        _, _, winner_name, sa_best, fm_best, av_best = refinados[0]
        log.console(f"   Multi-start: vencedor = '{winner_name}'")
    else:
        sa_best, fm_best, av_best = starts[0][1]

    log.console(f"   Inicial: Tmax={av_best.Tmax:.1f}h "
                f"gap={gap_de(av_best.Tmax):.1f}% "
                f"pen_cap={av_best.pen_cap:.3f} "
                f"pen_jan={av_best.pen_window:.3f} "
                f"pen_tsk={av_best.pen_tasks:.3f}")

    dist = {}
    for j in range(N):
        dist[sa_best[j]] = dist.get(sa_best[j], 0) + 1
    log.console("   Distribuicao por sprint: "
                + " ".join(f"s{s}:{n}" for s, n in sorted(dist.items())))
    n_bapr = sum(1 for j in range(N) for i in range(M)
                 if fm_best[j][i] > EPS and av_best.B_apr[i][j] > EPS)
    n_ftec = sum(1 for j in range(N) for i in range(M)
                 if fm_best[j][i] > EPS and inst["F_tech"][i][j] < 0.99)
    log.console(f"   Diagnostico: {n_bapr} alocacoes c/ B_apr>0, "
                f"{n_ftec} c/ F_tech<1")

    chk0 = vc.verify_all(inst, sa_best, fm_best, av_best)
    log.console(f"   Verificacao integral (43 restricoes) da gulosa: "
                + ("FACTIVEL" if chk0["feasible"]
                   else f"INFACTIVEL — {chk0['summary_str']}"))
    log.block_to_file(vc.detailed_report(chk0))

    if not chk0["feasible"]:
        log.console("   [AVISO] Gulosa infeasible — aplicando repair complete")
        sa_best, fm_best = full_repair(inst, sa_best, fm_best, log)
        av_best = evaluate(inst, sa_best, fm_best)
        chk0 = vc.verify_all(inst, sa_best, fm_best, av_best)
        log.console("   Apos repair: "
                    + ("FACTIVEL" if chk0["feasible"]
                       else f"AINDA INFACTIVEL — {chk0['summary_str']}"))

    incumbent_feasible = chk0["feasible"]

    # ------------------------------------------------------------------
    # VND inicial
    # ------------------------------------------------------------------
    deadline0 = t_start + min(cfg.VND_MAX_TIME,
                               max(cfg.VND_MIN_TIME,
                                   T_limit * cfg.VND_TIME_FRAC))
    sa_c, fm_c, av_c, fp_c, _ = vnd(inst, sa_best, fm_best, deadline0,
                                    vnd_log, 0, log)
    ok, sa_c, fm_c, av_c, chk_c = acceptance_gate(inst, sa_c, fm_c, av_c,
                                                 log, 0)
    if ok and (not incumbent_feasible or incumbent_improves(av_c, av_best)):
        sa_best, fm_best, av_best = sa_c, fm_c, av_c
        incumbent_feasible = True
    elif not ok and not incumbent_feasible and penalized_objective(av_c) < penalized_objective(av_best):
        sa_best, fm_best, av_best = sa_c, fm_c, av_c

    Tmax_best = av_best.Tmax
    fp_best = penalized_objective(av_best)
    log.console(f"   VND inicial: Tmax={Tmax_best:.1f}h "
                f"gap={gap_de(Tmax_best):.1f}% "
                f"feasible={'SIM' if incumbent_feasible else 'NAO'}")

    convergence_log = [{
        "iteracao": 0, "tempo_s": round(time.time() - t_start, 3),
        "tmax_best_h": Tmax_best, "fp_best": fp_best, "fp_iter": fp_best,
        "gap_pct": gap_de(Tmax_best), "k": 0, "melhoria": 1,
        "rejeitada_infactivel": 0,
    }]

    # ------------------------------------------------------------------
    # Estado da MATHEURÍSTICA (camada Gurobi)
    # ------------------------------------------------------------------
    mh_on = cfg.MH_ENABLED
    has_precedence = any(inst["Pred"][j] for j in range(N))
    # Reserva PRECEDENCE-AWARE (E-036): C3 reserva muito p/ a varredura; C1/C2
    # reservam o mínimo (o GVNS precisa do orçamento — o C2 crava 425.2h em ~123s).
    mh_reserve = (cfg.MH_FINAL_RESERVE if has_precedence
                  else cfg.MH_FINAL_RESERVE_NOPRED)
    # Injeção de sub-MIP no loop só em instâncias COM precedências (E-036).
    mh_loop_inject = (mh_on and cfg.MH_ON_STAGNATION
                      and (has_precedence or not cfg.MH_LOOP_ONLY_PRECEDENCE))
    loop_limit = (T_limit * (1 - mh_reserve)
                  if (mh_on and cfg.MH_FINAL_SWEEP) else T_limit)
    # Knobs do sub-MIP PRECEDENCE-AWARE (E-037): C3 usa recortes maiores/largos/
    # longos para compactar a cadeia; os demais usam os valores padrão.
    mh_free = cfg.MH_FREE_SET_SIZE_PRED if has_precedence else cfg.MH_FREE_SET_SIZE
    mh_radius = cfg.MH_SPRINT_RADIUS_PRED if has_precedence else cfg.MH_SPRINT_RADIUS
    mh_submip_time = cfg.MH_SUBMIP_TIME_PRED if has_precedence else cfg.MH_SUBMIP_TIME
    mh_scores = {s: 1.0 for s in cfg.MH_STRATEGIES}   # adaptativo (ALNS)
    mh_calls = 0
    mh_improves = 0
    mip_since = 0
    mh_consec_fail = 0       # back-off: sub-MIPs de loop seguidos sem melhora
    mh_loop_off = False      # quando True, para de injetar sub-MIP no loop
    submip_log = []

    # RNG ISOLADO da matheurística: as escolhas de estratégia NÃO consomem o
    # stream global do GVNS, para que a trajetória do GVNS fique idêntica à da
    # v1 com a mesma seed (ver E-035). Assim o sub-MIP só agrega; nunca desvia.
    mh_rng = random.Random((cfg.RANDOM_SEED or 0) * 7919 + 13)

    def _pick_strategy():
        if not cfg.MH_STRATEGY_ADAPTIVE:
            return mh_rng.choice(cfg.MH_STRATEGIES)
        ws = [mh_scores.get(s, 1.0) for s in cfg.MH_STRATEGIES]
        return mh_rng.choices(cfg.MH_STRATEGIES, weights=ws, k=1)[0]

    # ------------------------------------------------------------------
    # Loop GVNS
    # ------------------------------------------------------------------
    k = 1
    deck_k = []   # order aleatória das vizinhanças (cfg.K_RANDOM_ORDER)
    vns_iter = 0
    no_improve = 0
    n_rejected = 0
    ils_contador = 0   # local: o contador NÃO pode vazar entre instâncias
                       # (os sub-seeds das trajetórias novas dependem dele)

    while time.time() - t_start < loop_limit:
        if cfg.K_RANDOM_ORDER:
            if not deck_k:
                deck_k = list(range(1, cfg.K_MAX + 1))
                if inst["has_monoliths"]:
                    # bilhetes vencedores do C2: wrecks de frac-shift /
                    # dev-reassign / LNS reconstruídos pelo VND — dobra a
                    # frequência desses operadores
                    deck_k += [3, 4, 6, 8]
                random.shuffle(deck_k)
            k = deck_k.pop(0)
        k_usado = k
        sa_sh, fm_sh = shaking(inst, sa_best, fm_best, k, list(av_best.Dur))
        sa_sh, fm_sh = repair_feasibility_fast(inst, sa_sh, fm_sh)
        # Faixas de tratamento do candidato pós-shaking (por fp/fp_best):
        #   < REPAIR_PRE_VND_THRESHOLD  -> VND recebe o wreck CRU (o gradiente
        #     de penalidade guia a reconstrução criativa — caminho do ótimo
        #     no C2);
        #   [LIMIAR, DESCARTE)       -> repair iterativo limitado antes do
        #     VND (wrecks médios do C3: precedência×janela);
        #   >= SHAKE_DISCARD_FACTOR  -> descartado sem VND (irreparável).
        av_sh = evaluate(inst, sa_sh, fm_sh)
        if penalized_objective(av_sh) >= fp_best * cfg.REPAIR_PRE_VND_THRESHOLD:
            sa_sh, fm_sh, av_sh = iterative_repair(inst, sa_sh, fm_sh, log,
                                                   max_iters=inst["N"])
        # Triagem: wreck irreparável não merece orçamento de VND
        if penalized_objective(av_sh) > fp_best * cfg.SHAKE_DISCARD_FACTOR:
            vns_iter += 1
            log.verbose(f"[descarte it{vns_iter}] candidato pos-repair com "
                        f"fp={penalized_objective(av_sh):.0f} > "
                        f"{cfg.SHAKE_DISCARD_FACTOR}x fp_best — sem VND")
            no_improve += 1
            convergence_log.append({
                "iteracao": vns_iter,
                "tempo_s": round(time.time() - t_start, 3),
                "tmax_best_h": Tmax_best, "fp_best": fp_best,
                "fp_iter": penalized_objective(av_sh),
                "gap_pct": gap_de(Tmax_best), "k": k_usado,
                "melhoria": 0, "rejeitada_infactivel": 0,
            })
            continue

        restante = T_limit - (time.time() - t_start)
        deadline = time.time() + min(cfg.VND_MAX_TIME,
                                     max(cfg.VND_MIN_TIME,
                                         restante * cfg.VND_TIME_FRAC))
        fp_stage = penalized_objective(av_sh)
        sa_l, fm_l, av_l, fp_l, mel_l = vnd(inst, sa_sh, fm_sh, deadline,
                                            vnd_log, vns_iter + 1, log)
        # VND multi-estágio: ganha mais um estágio enquanto o candidato
        # está perto do incumbent OU a reconstrução progride forte
        # (fp caiu para <50% no último estágio). Wrecks recuperáveis
        # precisam de ~25-30s contíguos (caminho do ótimo no C2).
        stages = 0
        while (stages < cfg.VND_EXT_MAX_STAGES
               and time.time() - t_start < T_limit
               and fp_l > fp_best - EPS
               and (fp_l < fp_best * cfg.VND_EXT_FACTOR
                    or fp_l < fp_stage * cfg.VND_EXT_PROGRESS)):
            fp_stage = fp_l
            deadline2 = time.time() + cfg.VND_MAX_TIME
            sa_l, fm_l, av_l, fp_l, _ = vnd(inst, sa_l, fm_l, deadline2,
                                            vnd_log, vns_iter + 1, log)
            stages += 1
        vns_iter += 1

        melhoria = 0
        rejeitada = 0
        # Candidato é promissor se a função penalizada bate o incumbent
        if fp_l < fp_best - EPS:
            ok, sa_a, fm_a, av_a, chk_a = acceptance_gate(
                inst, sa_l, fm_l, av_l, log, vns_iter)
            if ok and (not incumbent_feasible
                       or incumbent_improves(av_a, av_best)):
                tmax_caiu = av_a.Tmax < Tmax_best - EPS
                sa_best, fm_best, av_best = sa_a, fm_a, av_a
                Tmax_best = av_best.Tmax
                fp_best = penalized_objective(av_best)
                incumbent_feasible = True
                melhoria = 1
                k = 1
                no_improve = 0
                rotulo = "melhoria" if tmax_caiu else "compressao"
                log.milestone(f"      [{rotulo}] it={vns_iter} k{k_usado}  "
                              f"Tmax={Tmax_best:.1f}h  "
                              f"gap={gap_de(Tmax_best):.1f}%  "
                              f"t={time.time()-t_start:.0f}s")
            elif not ok and not incumbent_feasible:
                # Incumbente ainda infactível: aceita candidato infactível
                # com fp menor para a busca progredir rumo à factibilidade
                fp_a = penalized_objective(av_a)
                if fp_a < fp_best - EPS:
                    sa_best, fm_best, av_best = sa_a, fm_a, av_a
                    Tmax_best = av_best.Tmax
                    fp_best = fp_a
                    melhoria = 1
                    k = 1
                    no_improve = 0
                    log.console(
                        f"   [PROGRESSO-INFACTIVEL] it={vns_iter} "
                        f"k={k_usado} fp={fp_best:.1f} (incumbent ainda "
                        f"infeasible: {chk_a['summary_str']})")
                else:
                    k = 1 if k >= cfg.K_MAX else k + 1
                    no_improve += 1
            elif not ok:
                rejeitada = 1
                n_rejected += 1
                log.console(
                    f"   [REJEITADA] iteracao {vns_iter} (k={k_usado}): o "
                    f"algoritmo ia pular para uma nova solucao "
                    f"(fp={fp_l:.1f} < best={fp_best:.1f}), mas mesmo "
                    f"apos repair ela estava INFACTIVEL devido a: "
                    f"{chk_a['summary_str']}")
                k = 1 if k >= cfg.K_MAX else k + 1
                no_improve += 1
            else:
                # Factível mas não melhora o Tmax real (penalidade enganosa)
                log.to_file(f"[gate it{vns_iter}] candidato feasible mas "
                            f"Tmax={av_a.Tmax:.1f} >= best={Tmax_best:.1f}")
                k = 1 if k >= cfg.K_MAX else k + 1
                no_improve += 1
        else:
            k = 1 if k >= cfg.K_MAX else k + 1
            no_improve += 1

        # ILS restart após estagnação prolongada. Alterna entre:
        #   (a) perturbação do incumbent (intensificação clássica);
        #   (b) TRAJETÓRIA NOVA a partir da gulosa — o experimento de
        #       sementes (C2: 1/5 acha o ótimo) mostrou que a bacia do
        #       incumbent captura as perturbações; só uma trajetória
        #       independente decorrelaciona a busca (changes.md E-028).
        if no_improve >= cfg.ILS_RESTART_AFTER:
            ils_contador += 1
            if ils_contador % 2 == 0:
                # Re-semeia o RNG: sem isso o stream continua o mesmo e as
                # "trajetórias novas" são correlacionadas — a loteria do C2
                # (E-029) precisa de trajetórias INDEPENDENTES por execução
                random.seed((cfg.RANDOM_SEED or 0) * 100003 + ils_contador)
                if inst["comp_weights"] is inst["rem_path"]:
                    # com precedências (C3): os breakthroughs vêm de wrecks
                    # LNS do incumbent, não da gulosa — o re-seed dá um
                    # draw de destruição INDEPENDENTE
                    log.to_file(f"[ILS it{vns_iter}] restart WRECK re-semeado"
                                f" (LNS do incumbent, sub-seed {ils_contador})")
                    r6 = N6_lns_scatter(inst, sa_best, fm_best,
                                        list(av_best.Dur), cfg.LNS_P_K6)
                    sa_r, fm_r = r6 if r6 else (list(sa_best),
                                                [row[:] for row in fm_best])
                    sa_r, fm_r = repair_feasibility_fast(inst, sa_r, fm_r)
                else:
                    log.to_file(f"[ILS it{vns_iter}] restart TRAJETORIA NOVA "
                                f"(gulosa + VND fresco, sub-seed {ils_contador})")
                    sa_r, fm_r, _ = greedy_construction(inst, log)
            else:
                log.to_file(f"[ILS it{vns_iter}] restart por estagnacao "
                            f"({no_improve} iteracoes sem melhora)")
                n_pert = max(1, int(N * cfg.ILS_PERTURB_FRAC))
                sa_r = list(sa_best)
                fm_r = [row[:] for row in fm_best]
                for j in random.sample(range(N), n_pert):
                    sa_r[j] = random.randint(1, S_max)
                sa_r = repair_topologico(inst, sa_r, list(av_best.Dur), fm_r)
                sa_r, fm_r = repair_feasibility_fast(inst, sa_r, fm_r)
                sa_r, fm_r, _ = iterative_repair(inst, sa_r, fm_r, log,
                                                 max_iters=inst["N"])
            restante = T_limit - (time.time() - t_start)
            deadline = time.time() + min(cfg.VND_MAX_TIME,
                                         max(cfg.VND_MIN_TIME,
                                             restante * cfg.VND_TIME_FRAC))
            sa_i, fm_i, av_i, fp_i, _ = vnd(inst, sa_r, fm_r, deadline,
                                            vnd_log, vns_iter, log)
            if fp_i < fp_best - EPS:
                ok, sa_a, fm_a, av_a, chk_a = acceptance_gate(
                    inst, sa_i, fm_i, av_i, log, vns_iter)
                if ok and (not incumbent_feasible
                           or incumbent_improves(av_a, av_best)):
                    sa_best, fm_best, av_best = sa_a, fm_a, av_a
                    Tmax_best = av_best.Tmax
                    fp_best = penalized_objective(av_best)
                    incumbent_feasible = True
                    log.milestone(f"      [melhoria-ILS] it={vns_iter}  "
                                  f"Tmax={Tmax_best:.1f}h  "
                                  f"gap={gap_de(Tmax_best):.1f}%")
                elif not ok:
                    n_rejected += 1
                    log.console(
                        f"   [REJEITADA-ILS] iteracao {vns_iter}: candidato "
                        f"do restart infeasible: {chk_a['summary_str']}")
            no_improve = 0
            k = 1

        convergence_log.append({
            "iteracao": vns_iter,
            "tempo_s": round(time.time() - t_start, 3),
            "tmax_best_h": Tmax_best,
            "fp_best": fp_best,
            "fp_iter": fp_l,
            "gap_pct": gap_de(Tmax_best),
            "k": k_usado,
            "melhoria": melhoria,
            "rejeitada_infactivel": rejeitada,
        })

        # --- Injeção de sub-MIP (matheurística, Modo A) ao estagnar ---
        # A cada MH_STAGNATION_AFTER iterações sem melhora, recorta um
        # subproblema e o Gurobi o reotimiza; o candidato passa pelo MESMO
        # gate das 43 restrições e só é aceito se melhorar o Tmax REAL.
        if mh_loop_inject and not mh_loop_off:
            mip_since = 0 if melhoria else mip_since + 1
            if mip_since >= cfg.MH_STAGNATION_AFTER and incumbent_feasible:
                mip_since = 0
                strat = _pick_strategy()
                remaining = loop_limit - (time.time() - t_start)
                tl = min(mh_submip_time, remaining * cfg.MH_SUBMIP_FRAC)
                if tl >= 1.0:
                    r = mip_reoptimize(inst, sa_best, fm_best, av_best, strat,
                                       tl, log, f"it{vns_iter}",
                                       mh_free, mh_radius)
                    mh_calls += 1
                    accepted = 0
                    if r is not None:
                        sa2, fm2, free_mip, obj_g = r[0], r[1], r[2], r[3]
                        av2 = evaluate(inst, sa2, fm2)
                        ok, sa_a, fm_a, av_a, chk_a = acceptance_gate(
                            inst, sa2, fm2, av2, log, vns_iter)
                        if ok and incumbent_improves(av_a, av_best):
                            tmax_caiu = av_a.Tmax < Tmax_best - EPS
                            sa_best, fm_best, av_best = sa_a, fm_a, av_a
                            Tmax_best = av_best.Tmax
                            fp_best = penalized_objective(av_best)
                            incumbent_feasible = True
                            mh_improves += 1
                            accepted = 1
                            mh_consec_fail = 0
                            mh_scores[strat] = (              # C-110 (ALNS decay)
                                cfg.MH_ALNS_DECAY * mh_scores.get(strat, 1.0)
                                + (1 - cfg.MH_ALNS_DECAY) * cfg.MH_ALNS_REWARD_BEST)
                            no_improve = 0
                            k = 1
                            rotulo = "MIP" if tmax_caiu else "MIP-compr"
                            log.milestone(
                                f"      [{rotulo} {strat}] it={vns_iter}  "
                                f"Tmax={Tmax_best:.1f}h  "
                                f"gap={gap_de(Tmax_best):.1f}%  "
                                f"t={time.time()-t_start:.0f}s")
                        else:
                            mh_consec_fail += 1
                            mh_scores[strat] = (              # C-110 (ALNS decay)
                                cfg.MH_ALNS_DECAY * mh_scores.get(strat, 1.0)
                                + (1 - cfg.MH_ALNS_DECAY) * cfg.MH_ALNS_REWARD_FAIL)
                            if not ok:
                                n_rejected += 1
                                log.to_file(
                                    f"   [MIP-REJEITADA] it{vns_iter} {strat}: "
                                    f"{chk_a['summary_str']}")
                        submip_log.append({
                            "iteracao": vns_iter, "fase": "loop",
                            "strategy": strat, "free": len(free_mip),
                            "obj_gurobi": round(obj_g, 2) if obj_g else None,
                            "tmax_best_h": round(Tmax_best, 2),
                            "aceito": accepted,
                            "tempo_s": round(time.time() - t_start, 2),
                        })
                    else:
                        mh_consec_fail += 1
                    # Back-off: muitos sub-MIPs de loop seguidos sem melhora →
                    # desliga a injeção e devolve o orçamento ao GVNS (E-032).
                    if mh_consec_fail >= cfg.MH_MAX_CONSEC_FAILS:
                        mh_loop_off = True
                        log.to_file(f"   [MIP back-off] {mh_consec_fail} sub-MIPs "
                                    f"de loop sem melhora — injecao desligada; "
                                    f"orcamento devolvido ao GVNS")

    # ----------------------------------------------------------------------
    # Varredura final fix-and-optimize / POPMUSIC sobre o melhor incumbente
    # ----------------------------------------------------------------------
    if mh_on and cfg.MH_FINAL_SWEEP and incumbent_feasible:
        t_deadline = t_start + T_limit
        if t_deadline - time.time() > 1.0:
            log.console("   [MIP] varredura final fix-and-optimize (POPMUSIC)...")

            def _on_sweep_improve(strat, av_new):
                log.milestone(f"      [MIP-sweep {strat}]  "
                              f"Tmax={av_new.Tmax:.1f}h  "
                              f"gap={gap_de(av_new.Tmax):.1f}%  "
                              f"t={time.time()-t_start:.0f}s")
                submip_log.append({
                    "iteracao": vns_iter, "fase": "sweep", "strategy": strat,
                    "free": mh_free, "obj_gurobi": None,
                    "tmax_best_h": round(av_new.Tmax, 2), "aceito": 1,
                    "tempo_s": round(time.time() - t_start, 2)})

            sa_best, fm_best, av_best, n_sweep = fix_and_optimize_sweep(
                inst, sa_best, fm_best, av_best, t_deadline, log,
                _on_sweep_improve, "final",
                free_size=mh_free, radius=mh_radius, submip_time=mh_submip_time)
            mh_calls += 1
            mh_improves += n_sweep
            Tmax_best = av_best.Tmax
            fp_best = penalized_objective(av_best)
            if n_sweep > 0:
                convergence_log.append({
                    "iteracao": vns_iter + 1,
                    "tempo_s": round(time.time() - t_start, 3),
                    "tmax_best_h": Tmax_best, "fp_best": fp_best,
                    "fp_iter": fp_best, "gap_pct": gap_de(Tmax_best),
                    "k": 0, "melhoria": 1, "rejeitada_infactivel": 0})

    # C-105: forward-backward / COMPACTAÇÃO de makespan PÓS-tudo (Valls et al.
    # 2005; Debels et al. 2006). Tenta esvaziar a última sprint ocupada movendo
    # cada uma de suas tarefas p/ a sprint factível mais cedo (precedência +
    # capacidade + janela, via evaluate). Esvaziar a última sprint derruba o
    # makespan 1 sprint. SEGURO: roda após a varredura, via gate, só aceita
    # melhoria. C3 é chain-bound (tipicamente no-op), mas é GERAL e ajuda
    # instâncias sprint-bound futuras; nunca piora.
    if incumbent_feasible:
        comp_improved = True
        while comp_improved and time.time() - t_start < T_limit:
            comp_improved = False
            s_last = max(sa_best)
            if s_last <= 1:
                break
            sa_t = list(sa_best)
            for j in [t for t in inst["topo"] if sa_best[t] == s_last]:
                s_min = max((sa_t[k] for k in inst["Pred"][j]), default=1)
                for s_new in range(s_min, s_last):
                    cand = list(sa_t)
                    cand[j] = s_new
                    cand = repair_topologico(inst, cand, list(av_best.Dur), fm_best)
                    av_c = evaluate(inst, cand, fm_best)
                    if (av_c.pen_cap < EPS and av_c.pen_window < EPS
                            and av_c.pen_tasks < EPS):
                        sa_t = cand
                        break
            av_t = evaluate(inst, sa_t, fm_best)
            ok, sa_a, fm_a, av_a, _chk = acceptance_gate(
                inst, sa_t, fm_best, av_t, log, vns_iter)
            if ok and incumbent_improves(av_a, av_best):
                sa_best, fm_best, av_best = sa_a, fm_a, av_a
                Tmax_best = av_best.Tmax
                fp_best = penalized_objective(av_best)
                comp_improved = True
                log.milestone(f"      [compactacao] Tmax={Tmax_best:.1f}h  "
                              f"gap={gap_de(Tmax_best):.1f}%")

    runtime = time.time() - t_start
    VNS_Gap = gap_de(Tmax_best)

    # ------------------------------------------------------------------
    # Verificação final integral + relatórios
    # ------------------------------------------------------------------
    log.section(f"RESULTADO — {name}")
    chk_final = vc.verify_all(inst, sa_best, fm_best, av_best)
    log.console(f"   Tmax={Tmax_best:.1f}h | gap={VNS_Gap:.1f}% | "
                f"t={runtime:.1f}s | iteracoes={vns_iter} | "
                f"candidatos rejeitados por infactibilidade={n_rejected}")
    log.console("   Verificacao final (43 restricoes): "
                + ("FACTIVEL" if chk_final["feasible"]
                   else f"INFACTIVEL — {chk_final['summary_str']}"))
    log.block_to_file("\nRELATORIO FINAL DAS 43 CONSTRAINTS:\n"
                      + vc.detailed_report(chk_final))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "constraints_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"Instancia: {name}\nTmax: {Tmax_best:.2f}h  "
                f"Gap: {VNS_Gap:.2f}%\n\n")
        f.write(vc.detailed_report(chk_final))

    # ------------------------------------------------------------------
    # Saída CSV (mesmo formato do Gurobi/allocations_super.csv)
    # ------------------------------------------------------------------
    allocs = []
    for i in range(M):
        for j in range(N):
            frac = fm_best[j][i]
            if frac < EPS:
                continue
            s = sa_best[j]
            allocs.append({
                "Sprint": s,
                "Dev": f"Dev_{i}",
                "Tarefa": f"T{j}",
                "Fracao": frac,
                "T_Din": av_best.T_din[i][j],
                "Inicio": av_best.Inic[j],
                "Fim": av_best.Inic[j] + av_best.Dur[j],
                "Duracao_Tarefa": av_best.Dur[j],
                "T_Nominal_Original": inst["T_nom"][j],
                "T_Base_Calculado": inst["T_sta"][i][j],
                "F_Tech_ET": inst["F_tech"][i][j],
                "F_Beh_EC": inst["F_beh"][i][j],
                "P_Ctx": av_best.P_ctx[i][s],
                "P_Com": av_best.P_com[j],
                "P_Phi_Fadiga": av_best.P_phi[i][s],
                "P_Match": av_best.P_match[j],
                "B_Apr_Acumulado": av_best.B_apr[i][j],
                "F_Total_Multiplicador": av_best.F_total[i][j],
                "Cg_Idx_Sustentabilidade": av_best.Cg[i][s],
                "VNS_Gap": VNS_Gap,
                "Runtime": runtime,
            })
    pd.DataFrame(allocs).to_csv(
        os.path.join(out_dir, "allocations.csv"), index=False)
    pd.DataFrame(convergence_log).to_csv(
        os.path.join(out_dir, "convergence_log.csv"), index=False)
    pd.DataFrame(vnd_log).to_csv(
        os.path.join(out_dir, "vnd_log.csv"), index=False)
    if submip_log:
        pd.DataFrame(submip_log).to_csv(
            os.path.join(out_dir, "submip_log.csv"), index=False)

    log.console(f"   Saida: {out_dir}/allocations.csv, "
                f"convergence_log.csv, vnd_log.csv, submip_log.csv, "
                f"debug_log.txt, constraints_report.txt")
    log.console(f"   Matheuristica: {mh_calls} chamadas de sub-MIP, "
                f"{mh_improves} melhorias aceitas (Gurobi)")
    log.close()

    return {
        "Cenario": name,
        "Status": "Sucesso" if chk_final["feasible"] else "Infactivel",
        "VNS_Gap (%)": f"{VNS_Gap:.2f}%",
        "Tempo (s)": round(runtime, 2),
        "Tmax": Tmax_best,
        "LB": LB,
        "T_limit": T_limit,
        "iteracoes": vns_iter,
        "rejeitadas": n_rejected,
        "mh_calls": mh_calls,
        "mh_improves": mh_improves,
        "convergence_log": convergence_log,
        "vnd_log": vnd_log,
    }


if __name__ == "__main__":
    start_total = time.time()
    resultados = []
    for fp in sorted(glob.glob("data/instances/*.json")):
        resultados.append(solve_vns(fp))
    end_total = time.time() - start_total
    print("\n" + "=" * 70)
    print("RESUMO VNS")
    print("=" * 70)
    for r in resultados:
        print(f"{r['Cenario']:<35} | {r['VNS_Gap (%)']:>10} | "
              f"{r['Tempo (s)']:>8.2f}s | {r['Status']}")
    print(f"Total: {end_total:.1f}s")
    print("=" * 70)
