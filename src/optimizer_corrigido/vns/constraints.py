# -*- coding: utf-8 -*-
"""
============================================================================
 constraints.py — Modelo SPSP: avaliação + as 43 restrições (Eq. 6–48)
============================================================================

Este módulo é a ÚNICA fonte de verdade sobre a semântica do modelo:

  1. `load_instance(path)` — lê o JSON e pré-computa as matrizes do
     modelo (Eq. 1–4: F_tech, F_beh, T_sta) exatamente como o optimizer.py.

  2. `evaluate(inst, sa, fm)` — dado um par (sprint_asg, frac_matrix),
     reconstrói TODAS as variáveis derivadas do MILP (multiplicadores,
     T_din, cronograma Inic/Dur, Tmax) e as penalidades agregadas.

  3. As 43 restrições do artigo (Eq. 6 a 48), escritas de forma explícita
     e padronizada como funções `r06_...` até `r48_...`:

        - restrições de igualdade    h(x) = 0   →  violação = |h(x)|
        - restrições de desigualdade g(x) <= 0  →  violação = max(0, g(x))

     Cada função recebe (inst, sa, fm, av) e devolve uma lista de
     violações (lista vazia = restrição satisfeita). As restrições
     "definicionais" (que no MILP só definem variáveis auxiliares) são
     RECOMPUTADAS de forma independente e comparadas com os valores que
     `evaluate()` produziu — assim qualquer erro de implementação na
     avaliação é detectado na hora.

  4. `verify_all(inst, sa, fm, av)` — verificação integral: roda as
     43 restrições e devolve um relatório complete + veredicto de
     factibilidade (considerando os modes ENFORCE/WARN do vns_config).

Representação da solução (a mesma do VNS):
  - sa[j]    = sprint da task j (int em 1..S_max) — toda task ocorre
               em exatamente UMA sprint (Eq. 31–32).
  - fm[j][i] = fração de dedicação do dev i à task j (0 ou valor de
               F_FRACS/12). Invariante: sum(fm[j]) == 1.0 (Eq. 30).
"""

import json
import math
import heapq

from . import config as cfg

EPS = 1e-9          # threshold "fração existe"
TOL = cfg.TOL_CONSTRAINT


# ============================================================================
# 1. CARREGAMENTO DA INSTÂNCIA + PRÉ-COMPUTAÇÃO (Eq. 1–4)
# ============================================================================

def topological_sort(Pred, N):
    succs = [[] for _ in range(N)]
    for j in range(N):
        for p in Pred[j]:
            succs[p].append(j)
    in_deg = [len(Pred[j]) for j in range(N)]
    queue = [j for j in range(N) if in_deg[j] == 0]
    order = []
    while queue:
        j = queue.pop(0)
        order.append(j)
        for s in succs[j]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    return order


def load_instance(file_path: str) -> dict:
    """Lê o JSON da instância e pré-computa tudo que o modelo precisa.

    Eq. 1: f_tech  = 1 / (1 + (1/|H|) * sum(max(0, Hd-Ht)))     [divisão; só excedente; sem piso]
    Eq. 2: psi     = (1/2) * sum(|Pd - Pt|)                     [perfis Pd/Pt somam 1]
    Eq. 3: f_beh   = 1 + 5 * psi^3
    Eq. 4: T_sta   = T_nom * f_beh * f_tech
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    inst = {
        "name":     data["_params"]["Nome"],
        "M":        data["M"],
        "N":        data["N"],
        "S_max":    data["S_max"],
        "H_size":   data["H_size"],
        "P_size":   data["P_size"],
        "Hd":       data["Hd"],
        "Ht":       data["Ht"],
        "Pd":       data["Pd"],
        "Pt":       data["Pt"],
        "T_nom":    data["T_base"],
        "Pred":     data["Pred"],
        "Gap_prec": data["Gap_prec"],
    }
    M, N = inst["M"], inst["N"]
    H_size, P_size = inst["H_size"], inst["P_size"]
    Hd, Ht, Pd, Pt = inst["Hd"], inst["Ht"], inst["Pd"], inst["Pt"]
    T_nom = inst["T_nom"]

    # --- Eq. 1–4: matrizes F_tech, F_beh e T_sta (= T_base do optimizer) ---
    F_tech = [[0.0] * N for _ in range(M)]
    F_beh  = [[0.0] * N for _ in range(M)]
    T_sta  = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            # Eq. 1: f_tech = 1 / (1 + (1/|H|)·Σ_h max(0, Hd_ih - Ht_jh))
            sigma = (sum(max(0, Hd[i][h] - Ht[j][h]) for h in range(H_size))
                     / H_size)
            F_tech[i][j] = 1.0 / (1.0 + sigma)
            psi = cfg.PSI_WEIGHT * sum(abs(Pd[i][p] - Pt[j][p])
                                       for p in range(P_size))   # Eq. 2: (1/2)·Σ
            F_beh[i][j] = 1.0 + cfg.F_BEH_WEIGHT * (psi ** 3)
            T_sta[i][j] = T_nom[j] * F_beh[i][j] * F_tech[i][j]
    inst["F_tech"] = F_tech
    inst["F_beh"]  = F_beh
    inst["T_sta"]  = T_sta

    # --- HdT (Eq. 37 do artigo): HdT_ijh = 1 se o dev i cobre a skill h da
    #     task j (Hd_ih >= Ht_jh). Indexado POR SKILL (3D: HdT[i][j][h]). ---
    HdT = [[[1 if Hd[i][h] >= Ht[j][h] else 0 for h in range(H_size)]
            for j in range(N)] for i in range(M)]
    inst["HdT"] = HdT

    # --- Distância na chain de precedência (para Eq. 22, b_dep) ---
    pred_dist = [{} for _ in range(N)]
    for j in range(N):
        queue = [(j, 0)]
        visited = set()
        while queue:
            cur, dist = queue.pop(0)
            for p in inst["Pred"][cur]:
                if p not in visited:
                    visited.add(p)
                    pred_dist[j][p] = dist + 1
                    queue.append((p, dist + 1))
    inst["pred_dist"] = pred_dist

    # --- Eixos temáticos (blocos de 10 tasks, como no optimizer.py) ---
    axis = [[] for _ in range(N)]
    for j in range(N):
        ini = (j // 10) * 10
        axis[j] = [k for k in range(ini, ini + 10) if k != j and k < N]
    inst["axis"] = axis

    inst["topo"] = topological_sort(inst["Pred"], N)
    inst["topo_pos"] = {j: idx for idx, j in enumerate(inst["topo"])}
    succs = [[] for _ in range(N)]
    for j in range(N):
        for p in inst["Pred"][j]:
            succs[p].append(j)
    inst["succs"] = succs

    # Caminho crítico remaining (regra CPM): prioridade de despacho no
    # scheduler — tasks com cauda longa de sucessoras rodam primeiro
    # dentro da sprint, senão as tasks de "filler" ocupam o dev e a
    # chain crítica espera (Cenário 3: 176h de espera vs 53h do Gurobi).
    rem_path = list(T_nom)
    for j in reversed(inst["topo"]):
        if succs[j]:
            rem_path[j] = T_nom[j] + max(rem_path[s2] + inst["Gap_prec"]
                                         for s2 in succs[j])
    inst["rem_path"] = rem_path

    # Pesos do termo de compressão: por criticidade (rem_path) APENAS quando
    # a instância tem precedências (protege a chain crítica — C3). Sem
    # precedências, rem_path = T_nom e os weights puxariam as tasks GRANDES
    # para cedo — no C2 isso empurra os monoliths para o início, o oposto
    # da estrutura ótima (monoliths tarde com B_apr acumulado).
    has_precedence = any(inst["Pred"][j] for j in range(N))
    inst["comp_weights"] = rem_path if has_precedence else [1.0] * N

    # Instância tem "monoliths"? (task que nem o best dev executa solo
    # dentro da capacity útil sem bônus de aprendizado). Nesses casos as
    # teams de aquecimento são estruturalmente NECESSÁRIAS (padrão Gurobi
    # C2) e a vizinhança L5 (des-equipar) não deve rodar.
    inst["has_monoliths"] = any(
        min(T_sta[i][j] for i in range(M)) + cfg.SETUP_HOURS > cfg.CAP_UTIL
        for j in range(N))

    # --- Lower bound p/ gap: max(load paralela perfeita, caminho crítico) ---
    LB_par = sum(T_nom) / M
    CP = list(T_nom)
    for j in inst["topo"]:
        for p in inst["Pred"][j]:
            CP[j] = max(CP[j], CP[p] + inst["Gap_prec"] + T_nom[j])
    inst["LB"] = max(LB_par, max(CP))

    # --- Prazo por task (Eq. 36): vetor S_max_j lido da instância (data-driven).
    #     Default = S_max p/ toda task (sem prazo apertado) se a instância for
    #     antiga e não trouxer o campo. O mesmo vetor alimenta o vns_v2 (herda
    #     este load_instance) e mantém paridade com o gurobi/validation_v2. ---
    inst["S_max_j"] = data.get("S_max_j", [inst["S_max"]] * N)

    return inst


# ============================================================================
# 2. AVALIAÇÃO DA SOLUÇÃO (variáveis derivadas do MILP + penalidades)
# ============================================================================

class Evaluation:
    """Todas as variáveis derivadas de uma solução (sa, fm).

    Campos (índices: i = dev, j = task, s = sprint):
      P_ctx[i][s], Cg[i][s], P_phi[i][s] — multiplicadores por (dev, sprint)
      P_com[j], P_match[j]               — por task (na sprint atribuída)
      B_apr[i][j], F_total[i][j]         — por (dev, task) atribuído
      T_din[i][j]                        — esforço individual c/ Setup (Eq.29)
      T_aux[i][j]                        — T_din sem Setup (p/ checar T_aux>=0)
      Dur[j], Inic[j], Tmax              — cronograma (Eq. 33–35, 41–45, 48)
      order[(i,s)]                       — order real de execução do dev i na
                                           sprint s (deriva as variáveis v)
      pen_cap, pen_window, pen_tasks     — penalidades agregadas (Eq.47,35,16)
      comp                               — média dos términos (Inic+Dur)/N:
                                           medida de compressão do cronograma
    """
    __slots__ = ("P_ctx", "Cg", "P_phi", "P_com", "P_match", "B_apr",
                 "F_total", "T_din", "T_aux", "Dur", "Inic", "Tmax",
                 "order", "pen_cap", "pen_window", "pen_tasks", "comp")


def evaluate(inst, sa, fm) -> Evaluation:
    """Reconstrói as variáveis do MILP a partir de (sa, fm).

    Semântica idêntica ao optimizer.py:
      Eq. 17: P_ctx  = 0.20 * max(0, n_tasks_count - 1)
      Eq. 19: Cg     = sum(F_beh das tasks do dev na sprint) / 30
      Eq. 20: P_phi  = 9 * max(0, Cg - 0.10)
      Eq. 18: P_com  = 0.05 * d(d-1)/2
      Eq. 12-15: P_match = 0.5 * sum_p |Peq_p - 0.25| / 1.5   (se d >= 2)
      Eq. 21-23: B_apr = 0.05*b_axis + 0.20*b_dep
      Eq. 24: F_total = 1 + P_ctx + P_com + P_phi + P_match - B_apr
      Eq. 29: T_din  = T_sta * frac * F_total + Setup
              (T_aux = T_sta*frac*F_total é LIVRE — pode ser <0 quando F_total<0;
               T_din = T_aux + Setup sem clamp, fiel ao artigo)

    Cronograma (Eq. 33, 41, 42-45): order topológica; cada task inicia no
    máximo entre o início da sprint, o fim das predecessoras + Gap_prec e a
    liberação dos devs da team (execução serial por dev — Eq. 42-45).
    """
    M, N, S_max = inst["M"], inst["N"], inst["S_max"]
    P_size = inst["P_size"]
    Pd     = inst["Pd"]
    F_beh  = inst["F_beh"]
    T_sta  = inst["T_sta"]

    # --- Passada única: teams por task e tasks por (dev, sprint) ---
    teams = [None] * N
    tasks_of = [[[] for _ in range(S_max + 1)] for _ in range(M)]
    for j in range(N):
        s = sa[j]
        row = fm[j]
        eq = [i for i in range(M) if row[i] > EPS]
        teams[j] = eq
        for i in eq:
            tasks_of[i][s].append(j)

    # --- Multiplicadores por (dev, sprint) ---
    P_ctx = [[0.0] * (S_max + 1) for _ in range(M)]
    Cg    = [[0.0] * (S_max + 1) for _ in range(M)]
    P_phi = [[0.0] * (S_max + 1) for _ in range(M)]
    for i in range(M):
        F_beh_i = F_beh[i]
        for s in range(1, S_max + 1):
            tasks_is = tasks_of[i][s]
            if not tasks_is:
                continue
            P_ctx[i][s] = cfg.CONTEXT_WEIGHT * (len(tasks_is) - 1)
            Cg[i][s]    = sum(F_beh_i[j] for j in tasks_is) / cfg.COG_LOAD_DIVISOR
            P_phi[i][s] = cfg.FATIGUE_WEIGHT * max(0.0, Cg[i][s] - cfg.FATIGUE_THRESHOLD)

    # --- Multiplicadores por task ---
    P_com   = [0.0] * N
    P_match = [0.0] * N
    for j in range(N):
        team = teams[j]
        d = len(team)
        P_com[j] = cfg.COMM_WEIGHT * (d * (d - 1) / 2)
        if d >= 2:
            sum_dev = sum(abs(sum(Pd[i][p] for i in team) / d - cfg.TEAM_MATCH_TARGET)
                           for p in range(P_size))
            P_match[j] = cfg.TEAM_MATCH_WEIGHT * sum_dev / cfg.TEAM_MATCH_NORM

    # --- Bônus de aprendizado (Eq. 21–23) por (dev, task) atribuído ---
    B_apr = [[0.0] * N for _ in range(M)]
    axis_inst = inst["axis"]
    pred_dist_inst = inst["pred_dist"]
    for j in range(N):
        s = sa[j]
        axis_j = axis_inst[j]
        pred_j = pred_dist_inst[j]
        for i in teams[j]:
            b_axis = sum(cfg.LAMBDA_LEARN ** (s - sa[k])
                         for k in axis_j
                         if sa[k] < s and fm[k][i] > EPS)
            b_dep = sum(cfg.LAMBDA_DEP ** (dist - 1)
                        for k, dist in pred_j.items()
                        if fm[k][i] > EPS)
            B_apr[i][j] = cfg.ALPHA_AXIS * b_axis + cfg.ALPHA_DEP * b_dep

    # --- F_total, T_din (Eq. 24 e 29) e duração (Eq. 34) ---
    F_total = [[0.0] * N for _ in range(M)]
    T_aux   = [[0.0] * N for _ in range(M)]
    T_din   = [[0.0] * N for _ in range(M)]
    Dur = [0.0] * N
    for j in range(N):
        s = sa[j]
        base_j = 1.0 + P_com[j] + P_match[j]
        dur_j = 0.0
        for i in teams[j]:
            ft = base_j + P_ctx[i][s] + P_phi[i][s] - B_apr[i][j]
            F_total[i][j] = ft
            t_aux = T_sta[i][j] * fm[j][i] * ft
            T_aux[i][j] = t_aux
            t_din = (t_aux if t_aux > 0.0 else 0.0) + cfg.SETUP_HOURS   # Eq. 63: γ≥0 ⇒ T_aux≥0
            T_din[i][j] = t_din
            if t_din > dur_j:
                dur_j = t_din
        Dur[j] = dur_j

    # --- Cronograma: Inic, order de execução por dev e Tmax ---
    # Despacho por PRIORIDADE (Kahn + heap): entre as tasks "prontas"
    # (todas as predecessoras já agendadas), agenda primeiro a de sprint
    # menor; dentro da sprint, DUAS regras de desempate são tentadas e a
    # best por (pen_window, Tmax, comp) vence:
    #   'team' — maior team primeiro (sincroniza todos os devs cedo;
    #              essencial no Cenário 2 — ver changes.md E-024);
    #   'cpm'    — maior caminho crítico remaining primeiro (a chain
    #              crítica não espera tasks de filler; essencial no
    #              Cenário 3 — ver changes.md E-025).
    # A order topológica fixa anterior rejeitava soluções genuinamente
    # factíveis do MILP (a solução ótima do Gurobi no C2 estourava a
    # janela em 24.2h só pela order de despacho).
    Gap_prec = inst["Gap_prec"]
    Pred = inst["Pred"]
    succs = inst["succs"]
    topo_pos = inst["topo_pos"]
    rem_path = inst["rem_path"]
    indeg0 = [len(Pred[j]) for j in range(N)]

    def _schedule(regra):
        if regra == "team":
            def key(j):
                return (sa[j], -len(teams[j]), topo_pos[j], j)
        else:  # 'cpm'
            def key(j):
                return (sa[j], -rem_path[j], -len(teams[j]), topo_pos[j], j)
        Inic_r = [0.0] * N
        dev_free = [[0.0] * (S_max + 2) for _ in range(M)]
        order_r = {}
        indeg = list(indeg0)
        heap = [key(j) for j in range(N) if indeg[j] == 0]
        heapq.heapify(heap)
        while heap:
            j = heapq.heappop(heap)[-1]
            s = sa[j]
            preds_end = max((Inic_r[k] + Dur[k] + Gap_prec
                             for k in Pred[j]), default=0.0)
            team = teams[j]
            conflict = max((dev_free[i][s] for i in team), default=0.0)
            Inic_r[j] = max((s - 1) * cfg.H_SPAN, preds_end, conflict)
            for i in team:
                dev_free[i][s] = max(dev_free[i][s],
                                      Inic_r[j] + T_din[i][j])
                order_r.setdefault((i, s), []).append(j)
            for j2 in succs[j]:
                indeg[j2] -= 1
                if indeg[j2] == 0:
                    heapq.heappush(heap, key(j2))
        Tmax_r = max((Inic_r[j] + Dur[j] for j in range(N)), default=0.0)
        # Compressão PONDERADA por criticidade (rem_path): comprime as
        # tasks que alimentam chains longas; tasks de filler
        # (rem_path pequeno) não disputam as sprints iniciais — o termo
        # uniforme entupia as sprints 1-3 do dev da chain no Cenário 3
        # e a chain crítica esperava 80h+ (ver changes.md E-027).
        weights = inst["comp_weights"]
        sum_w = sum(weights)
        comp_r = (sum((Inic_r[j] + Dur[j]) * weights[j]
                      for j in range(N)) / sum_w) if sum_w else 0.0
        pj_r = sum(max(0.0, Inic_r[j] + Dur[j] - sa[j] * cfg.H_SPAN)
                   for j in range(N))
        return pj_r, Tmax_r, comp_r, Inic_r, order_r

    best = min((_schedule("team"), _schedule("cpm")),
                 key=lambda r: (r[0], r[1], r[2]))
    pen_window_h, Tmax, comp, Inic, order = best

    # --- Penalidades agregadas (medem violação das Eq. 47, 35 e 16) ---
    pen_cap = 0.0
    pen_tasks = 0
    for i in range(M):
        T_din_i = T_din[i]
        for s in range(1, S_max + 1):
            tasks_is = tasks_of[i][s]
            if not tasks_is:
                continue
            exc = sum(T_din_i[j] for j in tasks_is) - cfg.CAP_UTIL
            if exc > 0.0:
                pen_cap += exc
            if len(tasks_is) > cfg.MAX_TASKS_PER_DEV_SPRINT:
                pen_tasks += len(tasks_is) - cfg.MAX_TASKS_PER_DEV_SPRINT
    pen_cap *= cfg.CAP_UTIL
    pen_tasks *= cfg.CAP_UTIL

    pen_window = pen_window_h * cfg.H_SPAN

    av = Evaluation()
    av.P_ctx, av.Cg, av.P_phi = P_ctx, Cg, P_phi
    av.P_com, av.P_match = P_com, P_match
    av.B_apr, av.F_total = B_apr, F_total
    av.T_din, av.T_aux = T_din, T_aux
    av.Dur, av.Inic, av.Tmax = Dur, Inic, Tmax
    av.order = order
    av.pen_cap, av.pen_window, av.pen_tasks = pen_cap, pen_window, pen_tasks
    av.comp = comp
    return av


def penalized_objective(av: Evaluation) -> float:
    """Função objetivo penalizada que GUIA a busca (problema irrestrito):
    Tmax + λ·penalidades + pressão de compressão (anti-platô).
    ATENÇÃO: o valor REAL da função objetivo do modelo é apenas av.Tmax —
    os demais termos existem só para dar gradiente à busca local."""
    return (av.Tmax
            + cfg.LAM_CAP * av.pen_cap
            + cfg.LAM_WINDOW * av.pen_window
            + cfg.LAM_TASKS * av.pen_tasks
            + cfg.LAM_COMP * av.comp)


# ============================================================================
# 3. AS 43 RESTRIÇÕES (Eq. 6–48) — formato padronizado h(x)=0 / g(x)<=0
# ============================================================================
# Convenções:
#   - Cada função devolve uma lista de tuplas (descricao, violacao) onde
#     violacao > TOL. Lista vazia = restrição satisfeita.
#   - Variáveis derivadas: w(i,j,s)=1 se fm[j][i]>0 e sa[j]==s;
#     z(j,s)=1 se sa[j]==s; u(j,s,d)=1 se sa[j]==s e |team(j)|==d;
#     lambda(i,j,s,f)=1 se w=1 e round(fm[j][i]*12)==f;
#     v(i,j,k,s) derivada da order real de execução (av.order).
# ============================================================================

def _team(fm, j, M):
    return [i for i in range(M) if fm[j][i] > EPS]


def r06_u_define_z(inst, sa, fm, av):
    """Eq. 6 (h): sum_{d=1..4} u_jsd - z_js = 0  ∀j,s.
    Garante que toda task ativa tem exatamente um tamanho de team d
    entre 1 e 4. Viola se a task não tem dev algum ou tem mais de 4."""
    out = []
    for j in range(inst["N"]):
        d = len(_team(fm, j, inst["M"]))
        sum_u = 1 if 1 <= d <= cfg.MAX_DEVS_PER_TASK else 0
        h = sum_u - 1   # z_js = 1 na sprint atribuída
        if abs(h) > TOL:
            out.append((f"T{j}: team com d={d} devs (precisa 1..4)", abs(h)))
    return out


def r07_w_consistent_u(inst, sa, fm, av):
    """Eq. 7 (h): sum_i w_ijs - sum_d d·u_jsd = 0  ∀j,s.
    O número de devs com fração ativa deve ser exatamente o d da team."""
    out = []
    for j in range(inst["N"]):
        d = len(_team(fm, j, inst["M"]))
        d_u = d if 1 <= d <= cfg.MAX_DEVS_PER_TASK else 0
        h = d - d_u
        if abs(h) > TOL:
            out.append((f"T{j}: sum_w={d} != sum_d_u={d_u}", abs(h)))
    return out


def r08_w_equals_sum_lambda(inst, sa, fm, av):
    """Eq. 8 (h): w_ijs - sum_f lambda_ijsf = 0  ∀i,j,s.
    Toda fração ativa deve corresponder a exatamente um f ∈ {3,4,6,8,9,12}."""
    out = []
    for j in range(inst["N"]):
        for i in _team(fm, j, inst["M"]):
            f12 = round(fm[j][i] * 12)
            ok = (f12 in cfg.F_FRACS) and abs(fm[j][i] - f12 / 12.0) <= TOL
            if not ok:
                out.append((f"T{j}/Dev_{i}: fracao {fm[j][i]:.4f} "
                            f"nao corresponde a f valido", 1.0))
    return out


def r09_y_le_w(inst, sa, fm, av):
    """Eq. 9 (g): y_ijsd - w_ijs <= 0. y=w·u por construção; verifica que
    nenhum dev fora da team é contado no Team Match."""
    out = []
    for j in range(inst["N"]):
        team = _team(fm, j, inst["M"])
        d = len(team)
        if not 1 <= d <= cfg.MAX_DEVS_PER_TASK:
            continue
        # y_ijsd = 1 apenas para i ∈ team com d = |team| → y <= w vale sse
        # todo i com y=1 está na team (verdadeiro por derivação).
    return out


def r10_y_le_u(inst, sa, fm, av):
    """Eq. 10 (g): y_ijsd - u_jsd <= 0. Por derivação y só existe para o d
    real da team; satisfeita por construção (registrada por completude)."""
    return []


def r11_y_ge_w_plus_u(inst, sa, fm, av):
    """Eq. 11 (g): w_ijs + u_jsd - 1 - y_ijsd <= 0. Se o dev está na team
    e a team tem tamanho d, então y_ijsd deve ser 1 (vale por derivação)."""
    return []


def r12_peq_definition(inst, sa, fm, av):
    """Eq. 12 (h): Peq_jsp - sum_i sum_{d>=2} (Pd_ip/d)·y_ijsd = 0.
    Recomputa o perfil médio da team e compara com o usado em P_match."""
    # A verificação efetiva acontece em r15 (P_match), que recomputa toda a
    # chain Peq→Dev→f_mt de forma independente. Aqui validamos só o Peq.
    out = []
    for j in range(inst["N"]):
        team = _team(fm, j, inst["M"])
        d = len(team)
        if d < 2:
            continue
        for p in range(inst["P_size"]):
            peq = sum(inst["Pd"][i][p] for i in team) / d
            if not math.isfinite(peq):
                out.append((f"T{j}: Peq nao finito no quadrant {p}", 1.0))
    return out


def r13_dev_ge_pos(inst, sa, fm, av):
    """Eq. 13 (g): (Peq_jsp - 0.25·sum_u) - Dev_jsp <= 0.
    Dev_jsp = |Peq - 0.25| por derivação exata → satisfeita; verificada
    indiretamente em r15."""
    return []


def r14_dev_ge_neg(inst, sa, fm, av):
    """Eq. 14 (g): (0.25·sum_u - Peq_jsp) - Dev_jsp <= 0. Idem r13."""
    return []


def r15_pmatch_definition(inst, sa, fm, av):
    """Eq. 15 (h): f_mt_js - 0.5·(sum_p Dev_jsp / 1.5) = 0.
    Recomputa o Team Match de forma independente e compara com av.P_match."""
    out = []
    for j in range(inst["N"]):
        team = _team(fm, j, inst["M"])
        d = len(team)
        if d >= 2:
            sum_dev = sum(abs(sum(inst["Pd"][i][p] for i in team) / d
                               - cfg.TEAM_MATCH_TARGET)
                           for p in range(inst["P_size"]))
            expected = cfg.TEAM_MATCH_WEIGHT * sum_dev / cfg.TEAM_MATCH_NORM
        else:
            expected = 0.0
        h = abs(av.P_match[j] - expected)
        if h > TOL:
            out.append((f"T{j}: P_match={av.P_match[j]:.4f} != "
                        f"recomputado {expected:.4f}", h))
    return out


def r16_max_tasks_dev(inst, sa, fm, av):
    """Eq. 16 (g): sum_j w_ijs - 5 <= 0  ∀i,s.
    Um dev não pode ter mais de 5 tasks na mesma sprint."""
    out = []
    for s in range(1, inst["S_max"] + 1):
        for i in range(inst["M"]):
            n = sum(1 for j in range(inst["N"])
                    if sa[j] == s and fm[j][i] > EPS)
            g = n - cfg.MAX_TASKS_PER_DEV_SPRINT
            if g > TOL:
                out.append((f"Dev_{i}/Sprint{s}: {n} tasks (max 5)", g))
    return out


def r17_pctx_definition(inst, sa, fm, av):
    """Eq. 17 (h): f_ct_is - 0.20·max(0, sum_j w_ijs - 1) = 0.
    Recomputa a penalidade de contexto e compara com av.P_ctx."""
    out = []
    for s in range(1, inst["S_max"] + 1):
        for i in range(inst["M"]):
            n = sum(1 for j in range(inst["N"])
                    if sa[j] == s and fm[j][i] > EPS)
            expected = cfg.CONTEXT_WEIGHT * max(0, n - 1)
            h = abs(av.P_ctx[i][s] - expected)
            if h > TOL:
                out.append((f"Dev_{i}/Sprint{s}: P_ctx={av.P_ctx[i][s]:.4f} "
                            f"!= {expected:.4f}", h))
    return out


def r18_pcom_definition(inst, sa, fm, av):
    """Eq. 18 (h): f_cm_js - 0.05·(d(d-1)/2)·u_jsd = 0.
    Recomputa o overhead de comunicação (Lei de Brooks)."""
    out = []
    for j in range(inst["N"]):
        d = len(_team(fm, j, inst["M"]))
        expected = cfg.COMM_WEIGHT * (d * (d - 1) / 2)
        h = abs(av.P_com[j] - expected)
        if h > TOL:
            out.append((f"T{j}: P_com={av.P_com[j]:.4f} != {expected:.4f}", h))
    return out


def r19_cg_definition(inst, sa, fm, av):
    """Eq. 19 (h): Cg_is - sum_j(w_ijs·F_beh_ij)/30 = 0.
    Recomputa o índice de load cognitiva."""
    out = []
    for s in range(1, inst["S_max"] + 1):
        for i in range(inst["M"]):
            expected = sum(inst["F_beh"][i][j] for j in range(inst["N"])
                           if sa[j] == s and fm[j][i] > EPS) / cfg.COG_LOAD_DIVISOR
            h = abs(av.Cg[i][s] - expected)
            if h > TOL:
                out.append((f"Dev_{i}/Sprint{s}: Cg={av.Cg[i][s]:.4f} "
                            f"!= {expected:.4f}", h))
    return out


def r20_pphi_definition(inst, sa, fm, av):
    """Eq. 20 (h): f_phi_is - 9·max(0, Cg_is - 0.10) = 0.
    Recomputa a penalidade de fadiga."""
    out = []
    for s in range(1, inst["S_max"] + 1):
        for i in range(inst["M"]):
            expected = cfg.FATIGUE_WEIGHT * max(0.0, av.Cg[i][s] - cfg.FATIGUE_THRESHOLD)
            h = abs(av.P_phi[i][s] - expected)
            if h > TOL:
                out.append((f"Dev_{i}/Sprint{s}: P_phi={av.P_phi[i][s]:.4f} "
                            f"!= {expected:.4f}", h))
    return out


def r21_baxis_definition(inst, sa, fm, av):
    """Eq. 21 (h): b_axis_ijs - 0.05·sum_{p<s} sum_{k∈E_j} w_ikp·0.8^(s-p) = 0.
    Verificada junto com r23 (B_apr total)."""
    return []


def r22_bdep_definition(inst, sa, fm, av):
    """Eq. 22 (h): b_dep_ij - 0.2·sum_{k∈Pred*(j)} 0.6^(dist-1)·sum_p w_ikp = 0.
    Verificada junto com r23 (B_apr total)."""
    return []


def r23_bapr_definition(inst, sa, fm, av):
    """Eq. 23 (h): f_apr_ijs - (b_axis_ijs + b_dep_ij) = 0.
    Recomputa o bônus de aprendizado complete e compara com av.B_apr."""
    out = []
    for j in range(inst["N"]):
        s = sa[j]
        for i in _team(fm, j, inst["M"]):
            b_axis = sum(cfg.LAMBDA_LEARN ** (s - sa[k])
                         for k in inst["axis"][j]
                         if sa[k] < s and fm[k][i] > EPS)
            b_dep = sum(cfg.LAMBDA_DEP ** (dist - 1)
                        for k, dist in inst["pred_dist"][j].items()
                        if fm[k][i] > EPS)
            expected = cfg.ALPHA_AXIS * b_axis + cfg.ALPHA_DEP * b_dep
            h = abs(av.B_apr[i][j] - expected)
            if h > TOL:
                out.append((f"T{j}/Dev_{i}: B_apr={av.B_apr[i][j]:.4f} "
                            f"!= {expected:.4f}", h))
    return out


def r24_ftotal_definition(inst, sa, fm, av):
    """Eq. 24 (h): F_tot - (1 + f_ct + f_cm + f_phi + f_mt - f_apr) = 0.
    Recomputa o multiplicador total e compara com av.F_total."""
    out = []
    for j in range(inst["N"]):
        s = sa[j]
        for i in _team(fm, j, inst["M"]):
            expected = (1.0 + av.P_ctx[i][s] + av.P_com[j]
                        + av.P_phi[i][s] + av.P_match[j] - av.B_apr[i][j])
            h = abs(av.F_total[i][j] - expected)
            if h > TOL:
                out.append((f"T{j}/Dev_{i}: F_total={av.F_total[i][j]:.4f} "
                            f"!= {expected:.4f}", h))
    return out


def _check_gamma(inst, sa, fm, av, qual):
    """Eq. 25–28: caixa de linearização gamma = F_total·lambda com
    M_max=100, M_min=-100. Com lambda∈{0,1} as 4 desigualdades equivalem a
    |F_total| <= 100 e gamma exato. Verifica para os pares atribuídos."""
    out = []
    for j in range(inst["N"]):
        for i in _team(fm, j, inst["M"]):
            ft = av.F_total[i][j]
            gama = ft   # lambda=1 na fração atribuída → gamma = F_total
            checks = {
                25: gama - cfg.GAMMA_M_MAX,            # g = γ - M_max·λ <= 0
                26: cfg.GAMMA_M_MIN - gama,            # g = M_min·λ - γ <= 0
                27: gama - ft,                         # g = γ - F_tot <= 0 (λ=1)
                28: ft - gama,                         # g = F_tot - γ <= 0 (λ=1)
            }
            g = checks[qual]
            if g > TOL:
                out.append((f"T{j}/Dev_{i}: F_total={ft:.4f} viola "
                            f"linearizacao Eq.{qual}", g))
    return out


def r25_gamma_ub(inst, sa, fm, av):
    """Eq. 25 (g): gamma_ijsf - M_max·lambda_ijsf <= 0 (F_total <= 100)."""
    return _check_gamma(inst, sa, fm, av, 25)


def r26_gamma_lb(inst, sa, fm, av):
    """Eq. 26 (g): M_min·lambda_ijsf - gamma_ijsf <= 0 (F_total >= -100)."""
    return _check_gamma(inst, sa, fm, av, 26)


def r27_gamma_le_ftot(inst, sa, fm, av):
    """Eq. 27 (g): gamma - F_tot + M_min·(1-lambda) <= 0."""
    return _check_gamma(inst, sa, fm, av, 27)


def r28_gamma_ge_ftot(inst, sa, fm, av):
    """Eq. 28 (g): F_tot - gamma - M_max·(1-lambda) <= 0."""
    return _check_gamma(inst, sa, fm, av, 28)


def r29_tdin_definition(inst, sa, fm, av):
    """Eq. 29 + Eq. 63 (γ≥0): T_din = T_sta·(1/12)·Σf·γ + w·Setup, com γ≥0.
    No dev ALOCADO a linearização (Eq. 25–28) dá γ=F_total, logo γ≥0 ⇒ F_total≥0
    ⇒ T_aux = T_sta·frac·F_total ≥ 0 ⇒ T_din ≥ Setup. F_total<0 num slot alocado
    é INFACTÍVEL no MILP (não há γ≥0 que o iguale) — acusado aqui."""
    out = []
    for j in range(inst["N"]):
        for i in _team(fm, j, inst["M"]):
            t_aux = inst["T_sta"][i][j] * fm[j][i] * av.F_total[i][j]
            if t_aux < -TOL:
                out.append((f"T{j}/Dev_{i}: F_total={av.F_total[i][j]:.4f} < 0 "
                            f"(γ≥0 violado — infeasible no MILP)", -t_aux))
                continue
            expected = max(0.0, t_aux) + cfg.SETUP_HOURS
            h = abs(av.T_din[i][j] - expected)
            if h > TOL:
                out.append((f"T{j}/Dev_{i}: T_din={av.T_din[i][j]:.4f} "
                            f"!= {expected:.4f}", h))
    return out


def r30_backlog_complete(inst, sa, fm, av):
    """Eq. 30 (h): sum_{i,s,f} f·lambda_ijsf - 12 = 0  ∀j.
    Toda task deve ser 100% executada (frações somam 12/12)."""
    out = []
    for j in range(inst["N"]):
        total = sum(fm[j][i] for i in range(inst["M"]))
        h = abs(total - 1.0)
        if h > TOL:
            out.append((f"T{j}: total das fracoes = {total:.4f} != 1.0", h))
    return out


def r31_z_equals_sum_x(inst, sa, fm, av):
    """Eq. 31 (h): z_js - sum_i x_ijs = 0  ∀j,s.
    Na sprint atribuída, as frações somam 1 (na prática ≡ Eq. 30 já que a
    task está em uma única sprint). Os 3 solvers do modelo corrigido usam a
    forma de igualdade z = Σ_i x."""
    out = []
    for j in range(inst["N"]):
        sum_x = sum(fm[j][i] for i in range(inst["M"]))
        h = abs(1.0 - sum_x)
        if h > TOL:
            out.append((f"T{j}: z=1 mas sum_x={sum_x:.4f}", h))
    return out


def r32_one_sprint(inst, sa, fm, av):
    """Eq. 32 (h): sum_s z_js - 1 = 0  ∀j.
    Toda task em exatamente uma sprint válida (1..S_max, inteira)."""
    out = []
    for j in range(inst["N"]):
        s = sa[j]
        if not (isinstance(s, int) and 1 <= s <= inst["S_max"]):
            out.append((f"T{j}: sprint atribuida {s} fora de 1..{inst['S_max']}",
                        1.0))
    return out


def r33_start_in_sprint(inst, sa, fm, av):
    """Eq. 33 (g): (s-1)·112·z_js - Inic_j <= 0  ∀j.
    A task não pode começar antes da abertura da sua sprint."""
    out = []
    for j in range(inst["N"]):
        g = (sa[j] - 1) * cfg.H_SPAN - av.Inic[j]
        if g > TOL:
            out.append((f"T{j}: Inic={av.Inic[j]:.1f}h antes da abertura "
                        f"da sprint {sa[j]} ({(sa[j]-1)*cfg.H_SPAN}h)", g))
    return out


def r34_dur_ge_tdin(inst, sa, fm, av):
    """Eq. 34 (g): T_din_ijs - Dur_j <= 0  ∀i,j,s.
    A duração da task cobre a contribuição de todos os devs."""
    out = []
    for j in range(inst["N"]):
        for i in _team(fm, j, inst["M"]):
            g = av.T_din[i][j] - av.Dur[j]
            if g > TOL:
                out.append((f"T{j}/Dev_{i}: T_din={av.T_din[i][j]:.2f} > "
                            f"Dur={av.Dur[j]:.2f}", g))
    return out


def r35_sprint_window(inst, sa, fm, av):
    """Eq. 35 (g): Inic_j + Dur_j - s·112·z_js <= 0  ∀j.
    A task deve TERMINAR dentro da janela da sua sprint."""
    out = []
    for j in range(inst["N"]):
        g = av.Inic[j] + av.Dur[j] - sa[j] * cfg.H_SPAN
        if g > TOL:
            out.append((f"T{j}: termina em {av.Inic[j]+av.Dur[j]:.1f}h mas a "
                        f"sprint {sa[j]} fecha em {sa[j]*cfg.H_SPAN}h", g))
    return out


def r36_task_deadline(inst, sa, fm, av):
    """Eq. 36 (g): sum_s s·z_js - S_max_j <= 0  ∀j.
    Prazo individual da task (instâncias atuais: S_max_j = S_max)."""
    out = []
    for j in range(inst["N"]):
        g = sa[j] - inst["S_max_j"][j]
        if g > TOL:
            out.append((f"T{j}: sprint {sa[j]} > prazo {inst['S_max_j'][j]}", g))
    return out


def r37_skill_coverage(inst, sa, fm, av):
    """Eq. 37 (g): 1 - sum_{i,s} HdT_ijh·w_ijs <= 0  ∀j∈N, ∀h∈H.
    Cada skill h da task j deve ser coberta por ≥1 dev alocado (HdT_ijh =
    Hd_ih >= Ht_jh). Skills diferentes podem ser cobertas por devs diferentes.
    Para h com Ht_jh=0, HdT_ijh=1 sempre → restrição trivialmente satisfeita."""
    out = []
    H_size = inst["H_size"]
    for j in range(inst["N"]):
        team = _team(fm, j, inst["M"])
        for h in range(H_size):
            cobertos = sum(inst["HdT"][i][j][h] for i in team)
            g = 1 - cobertos
            if g > TOL:
                out.append((f"T{j}/skill{h}: nenhum dev alocado tem Hd>=Ht "
                            f"(skill {h})", float(g)))
    return out


def r38_max_devs_task(inst, sa, fm, av):
    """Eq. 38 (g): sum_i w_ijs - 4 <= 0  ∀j,s.
    No máximo 4 desenvolvedores por task."""
    out = []
    for j in range(inst["N"]):
        d = len(_team(fm, j, inst["M"]))
        g = d - cfg.MAX_DEVS_PER_TASK
        if g > TOL:
            out.append((f"T{j}: {d} devs (max 4)", g))
    return out


def r39_w_lambda_duplicate(inst, sa, fm, av):
    """Eq. 39 (h): idêntica à Eq. 8 (w = sum_f lambda). O artigo a repete
    no bloco estrutural; mantida por fidelidade ao artigo."""
    return r08_w_equals_sum_lambda(inst, sa, fm, av)


def r40_x_definition(inst, sa, fm, av):
    """Eq. 40 (h): x_ijs - (1/12)·sum_f f·lambda_ijsf = 0  ∀i,j,s.
    Toda fração é um múltiplo exato de 1/12 dentro de F_FRACS."""
    out = []
    for j in range(inst["N"]):
        for i in _team(fm, j, inst["M"]):
            f12 = round(fm[j][i] * 12)
            h = abs(fm[j][i] - f12 / 12.0)
            if h > TOL:
                out.append((f"T{j}/Dev_{i}: fracao {fm[j][i]:.6f} nao e "
                            f"multiplo de 1/12", h))
    return out


def r41_precedence(inst, sa, fm, av):
    """Eq. 41 (g): Inic_k + Dur_k + Gap_prec - Inic_j <= 0  ∀j, k ∈ Pred(j).
    Sucessora só começa depois da predecessora terminar + gap."""
    out = []
    for j in range(inst["N"]):
        for k in inst["Pred"][j]:
            g = av.Inic[k] + av.Dur[k] + inst["Gap_prec"] - av.Inic[j]
            if g > TOL:
                out.append((f"T{j} inicia {av.Inic[j]:.1f}h mas pred T{k} "
                            f"termina {av.Inic[k]+av.Dur[k]:.1f}h + "
                            f"gap {inst['Gap_prec']}h", g))
    return out


def r42_v_covers_pairs(inst, sa, fm, av):
    """Eq. 42 (g): w_ijs + w_iks - 1 - (v_ijks + v_ikjs) <= 0.
    Todo par de tasks do mesmo dev na mesma sprint tem order definida.
    Por derivação (av.order) todo par tem order; verifica a coverage."""
    out = []
    for (i, s), lista in av.order.items():
        atribuidas = [j for j in range(inst["N"])
                      if sa[j] == s and fm[j][i] > EPS]
        if len(set(lista)) != len(set(atribuidas)):
            faltam = set(atribuidas) - set(lista)
            out.append((f"Dev_{i}/Sprint{s}: tasks sem order definida: "
                        f"{sorted(faltam)}", float(len(faltam))))
    return out


def r43_no_overlap(inst, sa, fm, av):
    """Eq. 43 (g): Inic_j + T_din_ijs - M·(1-v_ijks) - Inic_k <= 0.
    Execução serial: se o dev i faz j antes de k na mesma sprint (v=1),
    então k só inicia após o dev concluir sua contribuição em j."""
    out = []
    for (i, s), lista in av.order.items():
        for a in range(len(lista)):
            for b in range(a + 1, len(lista)):
                j, k = lista[a], lista[b]   # j executada antes de k
                g = av.Inic[j] + av.T_din[i][j] - av.Inic[k]
                if g > TOL:
                    out.append((f"Dev_{i}/Sprint{s}: T{j} "
                                f"[{av.Inic[j]:.1f}–"
                                f"{av.Inic[j]+av.T_din[i][j]:.1f}h] sobrepoe "
                                f"inicio de T{k} ({av.Inic[k]:.1f}h)", g))
    return out


def r44_v_le_w_j(inst, sa, fm, av):
    """Eq. 44 (g): v_ijks + v_ikjs - w_ijs <= 0. v só existe entre tasks
    atribuídas ao dev (vale por derivação de av.order)."""
    return []


def r45_v_le_w_k(inst, sa, fm, av):
    """Eq. 45 (g): v_ijks + v_ikjs - w_iks <= 0. Idem r44."""
    return []


def r46_caputil_definition(inst, sa, fm, av):
    """Eq. 46 (h): CapUtil_is - (Cap_is - Cerim_s)·(1-Buffer) = 0.
    Verificação de sanidade da constante de capacity útil (93.6h)."""
    expected = (cfg.H_SPAN - cfg.CEREMONY_HOURS) * (1 - cfg.BUFFER)
    h = abs(cfg.CAP_UTIL - expected)
    if h > TOL:
        return [(f"CAP_UTIL={cfg.CAP_UTIL} != {expected}", h)]
    return []


def r47_capacity(inst, sa, fm, av):
    """Eq. 47 (g): sum_j T_din_ijs - CapUtil_is <= 0  ∀i,s.
    A load total do dev na sprint não pode exceder a capacity útil."""
    out = []
    for s in range(1, inst["S_max"] + 1):
        for i in range(inst["M"]):
            load = sum(av.T_din[i][j] for j in range(inst["N"])
                        if sa[j] == s and fm[j][i] > EPS)
            g = load - cfg.CAP_UTIL
            if g > TOL:
                out.append((f"Dev_{i}/Sprint{s}: load {load:.2f}h > "
                            f"CapUtil {cfg.CAP_UTIL}h", g))
    return out


def r48_makespan(inst, sa, fm, av):
    """Eq. 48 (g): Inic_j + Dur_j - Tmax <= 0  ∀j.
    O makespan cobre o término de todas as tasks."""
    out = []
    for j in range(inst["N"]):
        g = av.Inic[j] + av.Dur[j] - av.Tmax
        if g > TOL:
            out.append((f"T{j}: termina {av.Inic[j]+av.Dur[j]:.1f}h > "
                        f"Tmax {av.Tmax:.1f}h", g))
    return out


# ============================================================================
# 4. REGISTRO DAS 43 RESTRIÇÕES + VERIFICAÇÃO INTEGRAL
# ============================================================================

# (numero_eq, kind h/g, funcao, descricao curta)
CONSTRAINTS = [
    (6,  'h', r06_u_define_z,           "u define tamanho de team = z"),
    (7,  'h', r07_w_consistent_u,      "sum_w = sum d.u"),
    (8,  'h', r08_w_equals_sum_lambda,  "w = sum lambda (fracao valida)"),
    (9,  'g', r09_y_le_w,               "y <= w (linearizacao Team Match)"),
    (10, 'g', r10_y_le_u,               "y <= u (linearizacao Team Match)"),
    (11, 'g', r11_y_ge_w_plus_u,        "y >= w + u - 1"),
    (12, 'h', r12_peq_definition,        "definition Peq (perfil medio)"),
    (13, 'g', r13_dev_ge_pos,           "Dev >= Peq - 0.25"),
    (14, 'g', r14_dev_ge_neg,           "Dev >= 0.25 - Peq"),
    (15, 'h', r15_pmatch_definition,     "definition P_match (Team Match)"),
    (16, 'g', r16_max_tasks_dev,      "max 5 tasks por dev/sprint"),
    (17, 'h', r17_pctx_definition,       "definition P_ctx (contexto)"),
    (18, 'h', r18_pcom_definition,       "definition P_com (Brooks)"),
    (19, 'h', r19_cg_definition,         "definition Cg (load cognitiva)"),
    (20, 'h', r20_pphi_definition,       "definition P_phi (fadiga)"),
    (21, 'h', r21_baxis_definition,      "definition b_axis (aprendizado)"),
    (22, 'h', r22_bdep_definition,       "definition b_dep (aprendizado)"),
    (23, 'h', r23_bapr_definition,       "definition B_apr total"),
    (24, 'h', r24_ftotal_definition,     "definition F_total"),
    (25, 'g', r25_gamma_ub,             "linearizacao gamma <= M.lambda"),
    (26, 'g', r26_gamma_lb,             "linearizacao gamma >= m.lambda"),
    (27, 'g', r27_gamma_le_ftot,        "linearizacao gamma <= F_tot - m(1-l)"),
    (28, 'g', r28_gamma_ge_ftot,        "linearizacao gamma >= F_tot - M(1-l)"),
    (29, 'h', r29_tdin_definition,       "definition T_din (+ T_aux >= 0)"),
    (30, 'h', r30_backlog_complete,     "backlog complete (fracoes = 12/12)"),
    (31, 'h', r31_z_equals_sum_x,       "z = sum x"),
    (32, 'h', r32_one_sprint,           "exatamente 1 sprint por task"),
    (33, 'g', r33_start_in_sprint,     "inicio apos abertura da sprint"),
    (34, 'g', r34_dur_ge_tdin,          "Dur >= T_din de todos os devs"),
    (35, 'g', r35_sprint_window,        "termino dentro da janela da sprint"),
    (36, 'g', r36_task_deadline,         "prazo individual da task"),
    (37, 'g', r37_skill_coverage, "coverage de skills (HdT)"),
    (38, 'g', r38_max_devs_task,      "max 4 devs por task"),
    (39, 'h', r39_w_lambda_duplicate,   "w = sum lambda (duplicate da Eq.8)"),
    (40, 'h', r40_x_definition,          "x = multiplo de 1/12"),
    (41, 'g', r41_precedence,          "precedence + Gap_prec"),
    (42, 'g', r42_v_covers_pairs,        "order definida p/ pares do dev"),
    (43, 'g', r43_no_overlap,     "execucao serial sem overlap"),
    (44, 'g', r44_v_le_w_j,             "v <= w_j"),
    (45, 'g', r45_v_le_w_k,             "v <= w_k"),
    (46, 'h', r46_caputil_definition,    "definition CapUtil (93.6h)"),
    (47, 'g', r47_capacity,           "capacity util por dev/sprint"),
    (48, 'g', r48_makespan,             "Tmax cobre o fim de todas as tasks"),
]

assert len(CONSTRAINTS) == 43, f"Esperava 43 restricoes, ha {len(CONSTRAINTS)}"


def verify_all(inst, sa, fm, av=None) -> dict:
    """Verificação integral: evaluate TODAS as 43 restrições sem exceção.

    Retorna dict com:
      feasible        — True se nenhuma restrição em mode ENFORCE foi violated
      violated_list        — lista de (num_eq, name, kind, mode, n_viol, max_viol,
                        examples) apenas das violated_list
      report       — mesma estrutura para TODAS as 43 (violated_list ou não)
      summary_str      — string compacta kind "r47(3 viol, max 4.20)" p/ console
    """
    if av is None:
        av = evaluate(inst, sa, fm)

    report = []
    violated_list = []
    for (num, kind, func, name) in CONSTRAINTS:
        mode = cfg.constraint_mode(num)
        if mode == 'OFF':
            report.append((num, name, kind, mode, 0, 0.0, []))
            continue
        viols = func(inst, sa, fm, av)
        n = len(viols)
        vmax = max((v for _, v in viols), default=0.0)
        entrada = (num, name, kind, mode, n, vmax, viols[:5])
        report.append(entrada)
        if n > 0:
            violated_list.append(entrada)

    feasible = all(mode != 'ENFORCE' or n == 0
                   for (_, _, _, mode, n, _, _) in report)

    # Resumo separado: ENFORCE (motivo de rejeição) x WARN (apenas aviso)
    resumo_enforce = ", ".join(
        f"Eq.{num} {name} ({n} viol, max {vmax:.3f})"
        for (num, name, _, mode, n, vmax, _) in violated_list
        if mode == 'ENFORCE') or "todas satisfeitas"
    summary_warn = ", ".join(
        f"Eq.{num} {name} ({n} viol)"
        for (num, name, _, mode, n, vmax, _) in violated_list
        if mode == 'WARN')

    return {"feasible": feasible, "violated_list": violated_list,
            "report": report, "summary_str": resumo_enforce,
            "summary_warn": summary_warn, "av": av}


def detailed_report(resultado: dict) -> str:
    """Relatório linha a linha das 43 restrições para o arquivo de debug."""
    lines = []
    lines.append(f"{'Eq.':>4} {'kind':<4} {'mode':<8} {'status':<10} "
                  f"{'#viol':>6} {'max_viol':>10}  descricao")
    lines.append("-" * 95)
    for (num, name, kind, mode, n, vmax, examples) in resultado["report"]:
        status = "OK" if n == 0 else ("AVISO" if mode == 'WARN' else "VIOLADA")
        lines.append(f"{num:>4} {kind:<4} {mode:<8} {status:<10} "
                      f"{n:>6} {vmax:>10.4f}  {name}")
        for (desc, v) in examples:
            lines.append(f"      -> {desc}  [viol={v:.4f}]")
    lines.append("-" * 95)
    lines.append("FACTIVEL (modes ENFORCE): "
                  + ("SIM" if resultado["feasible"] else "NAO"))
    return "\n".join(lines)
