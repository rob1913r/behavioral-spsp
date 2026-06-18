# -*- coding: utf-8 -*-
"""
============================================================================
 constraints.py — modelo da v5 (vns_v2): 43 restrições (herdadas) +
                  construtor do MILP REDUZIDO para a matheurística
============================================================================

A v5 REUSA, sem nenhuma alteração, a única fonte de verdade do modelo da v1
(`optimizer/vns/constraints.py`): `load_instance`, `evaluate`,
`penalized_objective`, `verify_all`, `detailed_report` e as 43 restrições
`r06…r48`. Isso garante que a verificação integral (o "gate") e a função
objetivo são EXATAMENTE as mesmas da metaheurística validada.

A ESTA base a v5 acrescenta:

  build_reduced_model(...) — monta o MILP GLOBAL (Eq. 6–48, paridade com
      `gurobi/solver.py` e `validation.py`) em que as binárias das tasks
      FIXAS estão travadas no incumbente e as do `free_set` ficam LIVRES
      dentro de um corredor (janela de sprints + devs candidatos). O Gurobi
      reotimiza o pedaço ATÉ A OTIMALIDADE, no contexto global (todas as
      restrições de acoplamento — capacidade, precedência, janela, makespan —
      enxergam também as tasks fixas), de modo que o resultado é factível por
      construção. Modelo enxuto: variáveis só nos (dev,sprint) possíveis de
      cada task; as variáveis de ordem serial `v` (que explodem em O(M·N²·S))
      só são criadas para pares que podem coabitar um mesmo (dev,sprint).

  extract_solution(...) — lê a solução do sub-MIP e devolve (sa, fm) no mesmo
      formato do resto do solver (frações em múltiplos de 1/12).

Convenção de índices: i = dev, j = task, s = sprint (inteiros).
"""

# --- Herda TODA a semântica do modelo da v1 (paridade obrigatória) ---
from optimizer.vns.constraints import *                       # noqa: F401,F403
from optimizer.vns.constraints import (                       # re-exporta p/ o solver
    load_instance, evaluate, penalized_objective, verify_all, detailed_report,
    Evaluation, CONSTRAINTS, topological_sort, EPS, TOL,
)

from . import config as cfg

import gurobipy as gp
from gurobipy import GRB


# ============================================================================
# CONSTRUTOR DO MILP REDUZIDO (fix-and-optimize / corridor)
# ============================================================================

# Constantes de linearização (idênticas ao gurobi/solver.py e ao validation.py)
_M_MAX, _M_MIN = 100.0, -100.0
_BIG_M = 100000
_D_MAX = cfg.MAX_DEVS_PER_TASK            # 4 (Eq. 38 / tamanho de equipe)
_F_FRACS = cfg.F_FRACS                     # [3, 4, 6, 8, 9, 12]

# Parâmetros do modelo (GRUPO A — iguais ao optimizer.py)
_ALPHA_EIXO, _LMBDA_APR = cfg.ALPHA_AXIS, cfg.LAMBDA_LEARN
_ALPHA_DEP,  _LMBDA_DEP = cfg.ALPHA_DEP, cfg.LAMBDA_DEP
_L_FAT, _PHI_FAT = cfg.FATIGUE_THRESHOLD, cfg.FATIGUE_WEIGHT
_SETUP = cfg.SETUP_HOURS
_CAPUTIL = cfg.CAP_UTIL
_H_SPAN = cfg.H_SPAN
_COG_DIV = cfg.COG_LOAD_DIVISOR


def _team(fm, j, M):
    return [i for i in range(M) if fm[j][i] > EPS]


def _candidate_devs(inst, fm, j, n_cand):
    """Devs candidatos de uma task livre: os n_cand de menor T_sta + os
    devs que já estão na task no incumbente (garante warm start factível)."""
    M = inst["M"]
    rank = sorted(range(M), key=lambda i: inst["T_sta"][i][j])
    cands = set(rank[:max(1, n_cand)])
    cands.update(_team(fm, j, M))
    return sorted(cands)


def build_reduced_model(inst, sa, fm, free_set, sprint_radius, dev_candidates,
                        time_limit, mip_gap, threads=1, seed=0):
    """Monta o MILP global com as tasks fixas travadas e o free_set livre.

    Retorna (model, refs) onde refs = dict com as estruturas para extrair a
    solução: {'z': {(j,s):var}, 'x': {(i,j,s):var}, 'free': set, 'poss': ...}.
    """
    M, N, S_max = inst["M"], inst["N"], inst["S_max"]
    P_size = inst["P_size"]
    Pd = inst["Pd"]
    F_beh = inst["F_beh"]
    T_sta = inst["T_sta"]
    Pred = inst["Pred"]
    Gap_prec = inst["Gap_prec"]
    axis = inst["axis"]
    pred_dist = inst["pred_dist"]
    free = set(free_set)

    # ---- Domínios possíveis (corredor) por task ----
    # poss_is[j] = lista de (i, s) onde a task j PODE ter w=1.
    #   fixa  : exatamente o incumbente (um por dev da equipe).
    #   livre : devs candidatos × janela de sprints [s_inc-Δ, s_inc+Δ].
    poss_is = {}
    sprints_of = {}     # j -> sprints possíveis (p/ z, u, P_com, P_match)
    fix_frac = {}       # (j) -> {i: f12} no incumbente (tasks fixas)
    for j in range(N):
        if j in free:
            cand = _candidate_devs(inst, fm, j, dev_candidates)
            lo = max(1, sa[j] - sprint_radius)
            hi = min(S_max, sa[j] + sprint_radius)
            sprs = list(range(lo, hi + 1))
            poss_is[j] = [(i, s) for i in cand for s in sprs]
            sprints_of[j] = sprs
        else:
            s0 = sa[j]
            team = _team(fm, j, M)
            poss_is[j] = [(i, s0) for i in team]
            sprints_of[j] = [s0]
            fix_frac[j] = {i: int(round(fm[j][i] * 12)) for i in team}

    # Conjuntos auxiliares
    IS = set()                       # (i, s) ativos no modelo
    tasks_at = {}                    # (i, s) -> [tasks possíveis ali]
    triples = []                     # (i, j, s) possíveis (p/ w, lambda, ...)
    for j in range(N):
        for (i, s) in poss_is[j]:
            IS.add((i, s))
            tasks_at.setdefault((i, s), []).append(j)
            triples.append((i, j, s))
    IS = sorted(IS)
    # C-103: índice (j,s)->[devs] pré-computado uma vez (evita rescans lineares
    # de `triples` O(N·S·|triples|) nas Eq. de Brooks/janela — acelera o BUILD).
    triples_by_js = {}
    for (i, j, s) in triples:
        triples_by_js.setdefault((j, s), []).append(i)

    # ---- Modelo ----
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("submip_spsp", env=env)
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.Threads = threads
    m.Params.Seed = seed
    m.Params.MIPFocus = getattr(cfg, "MH_GUROBI_MIPFOCUS", 0)

    # Variáveis por triple (i, j, s)
    w = m.addVars(triples, vtype=GRB.BINARY, name="w")
    x = m.addVars(triples, lb=0, ub=1, name="x")
    lam = m.addVars([(i, j, s, f) for (i, j, s) in triples for f in _F_FRACS],
                    vtype=GRB.BINARY, name="lam")
    B_apr = m.addVars(triples, lb=0, name="Bapr")
    F_tot = m.addVars(triples, lb=-10.0, name="Ftot")
    gamma = m.addVars([(i, j, s, f) for (i, j, s) in triples for f in _F_FRACS],
                      lb=-10.0, name="gam")
    T_aux = m.addVars(triples, lb=0, name="Taux")
    T_din = m.addVars(triples, lb=0, name="Tdin")
    y = m.addVars([(i, j, s, d) for (i, j, s) in triples
                   for d in range(1, _D_MAX + 1)], vtype=GRB.BINARY, name="y")

    # Variáveis por (j, s)
    js = [(j, s) for j in range(N) for s in sprints_of[j]]
    z = m.addVars(js, vtype=GRB.BINARY, name="z")
    u = m.addVars([(j, s, d) for (j, s) in js for d in range(1, _D_MAX + 1)],
                  vtype=GRB.BINARY, name="u")
    Devq = m.addVars([(j, s, p) for (j, s) in js for p in range(P_size)],
                     lb=0, name="Dev")
    P_com = m.addVars(js, lb=0, name="Pcom")
    P_match = m.addVars(js, lb=0, name="Pmatch")

    # Variáveis por (i, s)
    P_ctx = m.addVars(IS, lb=0, name="Pctx")
    Cg = m.addVars(IS, lb=0, name="Cg")
    P_phi = m.addVars(IS, lb=0, name="Pphi")

    # Variáveis por task / globais
    Inic = m.addVars(range(N), lb=0, name="Inic")
    Dur = m.addVars(range(N), lb=0, name="Dur")
    Tmax = m.addVar(lb=0, name="Tmax")

    # Variáveis de ordem serial v — SÓ para pares que podem coabitar (i, s)
    v = {}
    for (i, s) in IS:
        ts = tasks_at[(i, s)]
        for a in range(len(ts)):
            for b in range(a + 1, len(ts)):
                j, k = ts[a], ts[b]
                v[(i, j, k, s)] = m.addVar(vtype=GRB.BINARY)
                v[(i, k, j, s)] = m.addVar(vtype=GRB.BINARY)

    # ---- Travar binárias das tasks FIXAS no incumbente ----
    for j in range(N):
        if j in free:
            continue
        s0 = sa[j]
        z[j, s0].lb = z[j, s0].ub = 1
        for i in fix_frac[j]:
            w[i, j, s0].lb = w[i, j, s0].ub = 1
            f12 = fix_frac[j][i]
            for f in _F_FRACS:
                val = 1 if f == f12 else 0
                lam[i, j, s0, f].lb = lam[i, j, s0, f].ub = val

    # ===================== RESTRIÇÕES (Eq. 6–48) =====================
    # Por (i, s): contexto, fadiga, capacidade, ordem serial
    for (i, s) in IS:
        ts = tasks_at[(i, s)]
        # Eq. 16: máx. 5 tasks por dev/sprint
        m.addConstr(gp.quicksum(w[i, j, s] for j in ts) <= cfg.MAX_TASKS_PER_DEV_SPRINT)
        # Eq. 17: P_ctx = 0.20·max(0, nº tasks − 1)
        aux_c = m.addVar(lb=-GRB.INFINITY); tmp_c = m.addVar(lb=0)
        m.addConstr(aux_c == gp.quicksum(w[i, j, s] for j in ts) - 1)
        m.addGenConstrMax(tmp_c, [aux_c], 0.0)
        m.addConstr(P_ctx[i, s] == cfg.CONTEXT_WEIGHT * tmp_c)
        # Eq. 19: Cg = Σ w·F_beh / 30
        m.addConstr(Cg[i, s] == gp.quicksum(w[i, j, s] * F_beh[i][j] for j in ts) / _COG_DIV)
        # Eq. 20: P_phi = 9·max(0, Cg − 0.10)
        aux_p = m.addVar(lb=-GRB.INFINITY); tmp_p = m.addVar(lb=0)
        m.addConstr(aux_p == Cg[i, s] - _L_FAT)
        m.addGenConstrMax(tmp_p, [aux_p], 0.0)
        m.addConstr(P_phi[i, s] == _PHI_FAT * tmp_p)
        # Eq. 47: capacidade útil por dev/sprint
        m.addConstr(gp.quicksum(T_din[i, j, s] for j in ts) <= _CAPUTIL)
        # Eq. 42–45: ordem serial sem sobreposição (apenas pares coabitáveis)
        for a in range(len(ts)):
            for b in range(a + 1, len(ts)):
                j, k = ts[a], ts[b]
                m.addConstr(v[i, j, k, s] + v[i, k, j, s] >= w[i, j, s] + w[i, k, s] - 1)
                m.addConstr(Inic[k] >= Inic[j] + T_din[i, j, s] - _BIG_M * (1 - v[i, j, k, s]))
                m.addConstr(Inic[j] >= Inic[k] + T_din[i, k, s] - _BIG_M * (1 - v[i, k, j, s]))

    # Por (j, s): Brooks, Team Match, tamanho de equipe
    for (j, s) in js:
        devs_js = triples_by_js.get((j, s), [])
        # Eq. 18: P_com = 0.05·d(d−1)/2
        m.addConstr(P_com[j, s] == cfg.COMM_WEIGHT * gp.quicksum(
            (d * (d - 1) / 2) * u[j, s, d] for d in range(1, _D_MAX + 1)))
        # Eq. 6: Σ_d u = z
        m.addConstr(gp.quicksum(u[j, s, d] for d in range(1, _D_MAX + 1)) == z[j, s])
        # Eq. 7: Σ_i w = Σ_d d·u
        m.addConstr(gp.quicksum(w[i, j, s] for i in devs_js) ==
                    gp.quicksum(d * u[j, s, d] for d in range(1, _D_MAX + 1)))
        # Eq. 12–15: Team Match
        for p in range(P_size):
            peq = gp.quicksum((Pd[i][p] / d) * y[i, j, s, d]
                              for i in devs_js for d in range(2, _D_MAX + 1))
            soma_u = gp.quicksum(u[j, s, d] for d in range(2, _D_MAX + 1))
            m.addConstr(Devq[j, s, p] >= peq - cfg.TEAM_MATCH_TARGET * soma_u)
            m.addConstr(Devq[j, s, p] >= cfg.TEAM_MATCH_TARGET * soma_u - peq)
        m.addConstr(P_match[j, s] == cfg.TEAM_MATCH_WEIGHT *
                    gp.quicksum(Devq[j, s, p] for p in range(P_size)) / cfg.TEAM_MATCH_NORM)

    # Por triple (i, j, s): aprendizado, F_total, linearização, T_din, y
    for (i, j, s) in triples:
        # Eq. 21–23: B_apr (b_eixo só sprints anteriores; b_dep todos)
        b_eixo = gp.quicksum(
            w[i, k, p] * (_LMBDA_APR ** (s - p))
            for k in axis[j] for p in range(1, s) if (i, k, p) in w)
        b_dep = gp.quicksum(
            w[i, k, p] * (_LMBDA_DEP ** (dist - 1))
            for k, dist in pred_dist[j].items()
            for p in range(1, S_max + 1) if (i, k, p) in w)
        m.addConstr(B_apr[i, j, s] == _ALPHA_EIXO * b_eixo + _ALPHA_DEP * b_dep)
        # Eq. 8 / 40: w = Σ_f lambda ; x = Σ_f f·lambda / 12
        m.addConstr(w[i, j, s] == gp.quicksum(lam[i, j, s, f] for f in _F_FRACS))
        m.addConstr(x[i, j, s] == gp.quicksum(f * lam[i, j, s, f] for f in _F_FRACS) / 12.0)
        # Eq. 24: F_total
        m.addConstr(F_tot[i, j, s] == 1.0 + P_ctx[i, s] + P_com[j, s]
                    + P_phi[i, s] + P_match[j, s] - B_apr[i, j, s])
        # Eq. 25–28: linearização gamma = F_total·lambda
        for f in _F_FRACS:
            m.addConstr(gamma[i, j, s, f] <= _M_MAX * lam[i, j, s, f])
            m.addConstr(gamma[i, j, s, f] >= _M_MIN * lam[i, j, s, f])
            m.addConstr(gamma[i, j, s, f] <= F_tot[i, j, s] - _M_MIN * (1 - lam[i, j, s, f]))
            m.addConstr(gamma[i, j, s, f] >= F_tot[i, j, s] - _M_MAX * (1 - lam[i, j, s, f]))
        # Eq. 29: T_aux (lb=0 — paridade com baseline) e T_din
        m.addConstr(T_aux[i, j, s] == T_sta[i][j] * (1.0 / 12.0) *
                    gp.quicksum(f * gamma[i, j, s, f] for f in _F_FRACS))
        m.addConstr(T_din[i, j, s] == T_aux[i, j, s] + w[i, j, s] * _SETUP)
        # Eq. 34: Dur cobre T_din
        m.addConstr(Dur[j] >= T_din[i, j, s])
        # Eq. 9–11: y = w·u
        for d in range(1, _D_MAX + 1):
            m.addConstr(y[i, j, s, d] <= w[i, j, s])
            m.addConstr(y[i, j, s, d] <= u[j, s, d])
            m.addConstr(y[i, j, s, d] >= w[i, j, s] + u[j, s, d] - 1)

    # Por task j: backlog, sprint única, janela, precedência, makespan
    for j in range(N):
        sprs = sprints_of[j]
        # Eq. 30: backlog completo (Σ f·lambda = 12)
        m.addConstr(gp.quicksum(f * lam[i, j, s, f]
                                for (i, jj, s) in triples if jj == j
                                for f in _F_FRACS) == 12)
        # Eq. 32: exatamente 1 sprint
        m.addConstr(gp.quicksum(z[j, s] for s in sprs) == 1)
        for s in sprs:
            devs_js = triples_by_js.get((j, s), [])
            # Eq. 31: z <= Σ_i w
            m.addConstr(z[j, s] <= gp.quicksum(w[i, j, s] for i in devs_js))
            # Eq. 33: início após abertura da sprint
            m.addConstr(Inic[j] >= ((s - 1) * _H_SPAN) * z[j, s])
        # Eq. 35: término dentro da janela
        m.addConstr(Inic[j] + Dur[j] <= gp.quicksum(s * _H_SPAN * z[j, s] for s in sprs))
        # Eq. 41: precedência
        for k in Pred[j]:
            m.addConstr(Inic[j] >= Inic[k] + Dur[k] + Gap_prec)
        # Eq. 48: makespan
        m.addConstr(Tmax >= Inic[j] + Dur[j])

    # ---- Warm start (MIPStart = incumbente p/ as tasks livres) ----
    for j in free:
        s0 = sa[j]
        for (jj, s) in [(j, s) for s in sprints_of[j]]:
            z[jj, s].Start = 1 if s == s0 else 0
        team = _team(fm, j, M)
        for (i, jj, s) in triples:
            if jj != j:
                continue
            on = (i in team and s == s0)
            w[i, j, s].Start = 1 if on else 0
            f12 = int(round(fm[j][i] * 12)) if on else -1
            for f in _F_FRACS:
                lam[i, j, s, f].Start = 1 if (on and f == f12) else 0

    m.setObjective(Tmax, GRB.MINIMIZE)

    refs = {"z": z, "x": x, "lam": lam, "free": free, "env": env,
            "sprints_of": sprints_of, "triples": triples, "Tmax": Tmax,
            "n_v": len(v), "n_triples": len(triples)}
    return m, refs


def extract_solution(inst, sa, fm, model, refs):
    """Lê a solução do sub-MIP e devolve (sa2, fm2) — cópias do incumbente
    com APENAS as tasks livres atualizadas; frações em múltiplos de 1/12."""
    M = inst["M"]
    z, x = refs["z"], refs["x"]
    free, sprints_of, triples = refs["free"], refs["sprints_of"], refs["triples"]

    sa2 = list(sa)
    fm2 = [row[:] for row in fm]
    for j in free:
        # sprint escolhida
        s_sel = max(sprints_of[j], key=lambda s: z[j, s].X)
        sa2[j] = s_sel
        # frações no sprint escolhido (arredonda p/ múltiplo exato de 1/12)
        row = [0.0] * M
        for (i, jj, s) in triples:
            if jj == j and s == s_sel:
                f12 = int(round(x[i, j, s].X * 12))
                if f12 > 0:
                    row[i] = f12 / 12.0
        fm2[j] = row
    return sa2, fm2
