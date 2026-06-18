"""
validation_v2.py — valida a factibilidade da solução de uma metaheurística do
modelo CORRIGIDO (optimizer_corrigido) contra o MILP do artigo (Eq. 1–67).

É a versão "nova" do validation.py: implementa EXATAMENTE o mesmo modelo dos
solvers corrigidos — f_tech por DIVISÃO (Eq. 1); ψ = ½·Σ (Eq. 2); F^tot ∈ ℝ,
γ ≥ 0, T_din ≥ 0 (Eq. 62–64); z = Σx (Eq. 31); janela com δ·j_lab (Eq. 33/35);
prazo (Eq. 36); cobertura por skill HdT_ijh (Eq. 37); Σw ≤ 4 (Eq. 38); v ≤ w
(Eq. 43/44). Lê δ/Setup/Cerim/Buffer/S_j/j_lab da instância. Valida os
resultados em data/results_v2/. O `validation.py` (modelo ANTIGO) segue intocado
para o fluxo da versão antiga do pipeline.

Uso:
    python src/validation_v2.py data/instances/<inst>.json \\
        data/results_v2/vns/<inst>/allocations.csv
"""
import os
import sys
import json
import glob
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def reconstruct_solution(csv_path: str):
    """Lê allocations.csv e reconstrói sprint_asg e frac_matrix."""
    df = pd.read_csv(csv_path)
    tarefas = sorted(df["Tarefa"].unique(), key=lambda t: int(t.replace("T", "")))
    devs    = sorted(df["Dev"].unique(),    key=lambda d: int(d.replace("Dev_", "")))
    N = len(tarefas); M = len(devs)

    sprint_asg  = {t: 0 for t in tarefas}
    frac_matrix = {t: {d: 0.0 for d in devs} for t in tarefas}

    for _, row in df.iterrows():
        sprint_asg[row["Tarefa"]] = int(row["Sprint"])
        frac_matrix[row["Tarefa"]][row["Dev"]] = float(row["Fracao"])

    return sprint_asg, frac_matrix, tarefas, devs


def check_timing_direct(instance_path: str, csv_path: str) -> dict:
    """Verificação direta (sem Gurobi) das restrições de timing da solução."""
    with open(instance_path) as f:
        data = json.load(f)

    nome = data["_params"]["Nome"]
    Pred = data["Pred"]
    Gap_prec = data["Gap_prec"]
    H_span = data.get("delta", 14) * data.get("j_lab", 8)   # δ·j_lab (horas por sprint)

    df = pd.read_csv(csv_path)
    tarefas = sorted(df["Tarefa"].unique(), key=lambda t: int(t.replace("T", "")))

    inic = {}; fim = {}; sprint_de = {}
    for t in tarefas:
        row = df[df["Tarefa"] == t].iloc[0]
        inic[t]     = float(row["Inicio"])
        fim[t]      = float(row["Fim"])
        sprint_de[t] = int(row["Sprint"])

    violations = []

    # --- 1. Janela de sprint ---
    for t in tarefas:
        limite = sprint_de[t] * H_span
        if fim[t] > limite + 1e-6:
            violations.append(
                f"  JANELA: {t} termina em {fim[t]:.1f}h mas sprint {sprint_de[t]} "
                f"vai até {limite:.0f}h (overflow {fim[t]-limite:.1f}h)"
            )

    # --- 2. Precedência ---
    for j_idx, t in enumerate(tarefas):
        for k_idx in Pred[j_idx]:
            k = tarefas[k_idx]
            if inic[t] < fim[k] + Gap_prec - 1e-6:
                violations.append(
                    f"  PREC: {t} inicia em {inic[t]:.1f}h mas predecessor {k} "
                    f"termina em {fim[k]:.1f}h + gap={Gap_prec}h "
                    f"(faltam {fim[k]+Gap_prec-inic[t]:.1f}h)"
                )

    # --- 3. Sobreposição de devs ---
    for (dev, sprint), grp in df.groupby(["Dev", "Sprint"]):
        tasks_ds = list(grp["Tarefa"])
        for i1 in range(len(tasks_ds)):
            for i2 in range(i1 + 1, len(tasks_ds)):
                t1, t2 = tasks_ds[i1], tasks_ds[i2]
                tdin1 = float(df[(df["Tarefa"] == t1) & (df["Dev"] == dev)]["T_Din"].iloc[0])
                tdin2 = float(df[(df["Tarefa"] == t2) & (df["Dev"] == dev)]["T_Din"].iloc[0])
                fim1_dev = inic[t1] + tdin1
                fim2_dev = inic[t2] + tdin2
                overlap = (inic[t1] < fim2_dev - 1e-6) and (inic[t2] < fim1_dev - 1e-6)
                if overlap:
                    violations.append(
                        f"  OVERLAP: {dev} sprint {sprint}: {t1}[{inic[t1]:.1f}-{fim1_dev:.1f}] "
                        f"e {t2}[{inic[t2]:.1f}-{fim2_dev:.1f}] se sobrepõem"
                    )

    ok = len(violations) == 0
    tag = "OK — timing consistente" if ok else f"VIOLAÇÕES ({len(violations)} encontradas)"
    print(f"   [CHECK DIRETO] {nome}: {tag}")
    if violations:
        for v in violations[:15]:
            print(v)
        if len(violations) > 15:
            print(f"   ... e mais {len(violations)-15} violações.")

    return {"cenario": nome, "timing_ok": ok, "n_violacoes": len(violations)}


def validate(instance_path: str, csv_path: str) -> dict:
    """Constrói o MILP do modelo CORRIGIDO (Eq. 1–67), fixa as binárias na
    solução da metaheurística e resolve o LP para confirmar factibilidade."""
    with open(instance_path, "r") as f:
        data = json.load(f)

    nome_cenario = data["_params"]["Nome"]
    sprint_asg, frac_matrix, tarefas_str, devs_str = reconstruct_solution(csv_path)

    alpha_eixo, lmbda_apr = 0.05, 0.80
    alpha_dep,  lmbda_dep = 0.20, 0.60
    l_fat, phi_fat        = 0.10, 9.0
    Cerim_s   = data.get("Cerim_s", 8)
    Buffer    = data.get("Buffer", 0.1)
    Setup_dev = data.get("Setup", 2.0)
    j_lab     = data.get("j_lab", 8)
    delta     = data.get("delta", 14)
    Cap       = data.get("Cap", delta * j_lab)   # Eq. 45: capacidade nominal Cap_is (= δ·j_lab)
    BIG_M = 100000

    M_list   = [f"Dev_{i}" for i in range(data["M"])]
    N_list   = [f"T{j}"    for j in range(data["N"])]
    S_list   = list(range(1, data["S_max"] + 1))
    H        = list(range(data["H_size"]))
    P_quads  = list(range(data["P_size"]))
    F_fracs  = [3, 4, 6, 8, 9, 12]

    Hd, Ht, Pd, Pt = data["Hd"], data["Ht"], data["Pd"], data["Pt"]
    T_nom = data["T_base"]
    pred_indices = data["Pred"]
    Gap_prec = data["Gap_prec"]

    S_j_val = data.get("S_j", 4)
    S_j = {j: S_j_val for j in N_list}                   # Eq. 6–15/38: tamanho de equipe (=4)
    # Eq. 36: prazo (sprint-limite) POR TAREFA, lido da instância (vetor S_max_j).
    S_max_j_list = data.get("S_max_j", [data["S_max"]] * data["N"])
    S_max_deadline = {N_list[j]: S_max_j_list[j] for j in range(data["N"])}  # Eq. 36

    # Pré-computação (idêntica ao optimizer_corrigido)
    f_beh_dict, T_base_dict, F_tech_dict = {}, {}, {}
    pred_dist = {j_idx: {} for j_idx in range(len(N_list))}
    for j_idx in range(len(N_list)):
        queue = [(j_idx, 0)]; visited = set()
        while queue:
            current, dist = queue.pop(0)
            for p_idx in pred_indices[current]:
                if p_idx not in visited:
                    visited.add(p_idx)
                    pred_dist[j_idx][p_idx] = dist + 1
                    queue.append((p_idx, dist + 1))

    for i_idx, i in enumerate(M_list):
        for j_idx, j in enumerate(N_list):
            # Eq. 1: f_tech = 1 / (1 + (1/|H|)·Σ_h max(0, Hd-Ht))  (divisão; só excedente)
            sigma = sum(max(0, Hd[i_idx][h] - Ht[j_idx][h]) for h in H) / len(H)
            F_tech_dict[(i, j)] = 1.0 / (1.0 + sigma)
            # Eq. 2: psi = (1/2)·Σ_p |Pd-Pt|
            psi = 0.5 * sum(abs(Pd[i_idx][p] - Pt[j_idx][p]) for p in P_quads)
            f_beh_dict[(i, j)] = 1.0 + 5.0 * (psi ** 3)
            T_base_dict[(i, j)] = T_nom[j_idx] * f_beh_dict[(i, j)] * F_tech_dict[(i, j)]

    # HdT_ijh (Eq. 37, POR SKILL): 1 se Hd_ih >= Ht_jh
    def hdt(i_idx, j_idx, h):
        return 1 if Hd[i_idx][h] >= Ht[j_idx][h] else 0

    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    model = gp.Model("validate_vns_v2", env=env)
    model.Params.TimeLimit    = 120
    model.Params.SolutionLimit = 1

    lmbda = model.addVars(M_list, N_list, S_list, F_fracs, vtype=GRB.BINARY)
    x     = model.addVars(M_list, N_list, S_list, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    z     = model.addVars(N_list, S_list, vtype=GRB.BINARY)
    w     = model.addVars(M_list, N_list, S_list, vtype=GRB.BINARY)
    v     = model.addVars(M_list, N_list, N_list, S_list, vtype=GRB.BINARY)
    Inic  = model.addVars(N_list, vtype=GRB.CONTINUOUS, lb=0)
    Dur   = model.addVars(N_list, vtype=GRB.CONTINUOUS, lb=0)
    Tmax  = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
    u     = model.addVars(N_list, S_list, range(1, 5), vtype=GRB.BINARY)
    y     = model.addVars(M_list, N_list, S_list, range(1, 5), vtype=GRB.BINARY)
    P_ctx   = model.addVars(M_list, S_list, vtype=GRB.CONTINUOUS, lb=0)
    P_com   = model.addVars(N_list, S_list, vtype=GRB.CONTINUOUS, lb=0)
    Cg_idx  = model.addVars(M_list, S_list, vtype=GRB.CONTINUOUS, lb=0)
    P_phi   = model.addVars(M_list, S_list, vtype=GRB.CONTINUOUS, lb=0)
    Dev_var = model.addVars(N_list, S_list, P_quads, vtype=GRB.CONTINUOUS, lb=0)
    P_match = model.addVars(N_list, S_list, vtype=GRB.CONTINUOUS, lb=0)
    B_apr   = model.addVars(M_list, N_list, S_list, vtype=GRB.CONTINUOUS, lb=0)
    # Eq. 62–64: F^tot ∈ ℝ; γ ≥ 0; T_aux/T_din ≥ 0
    F_total = model.addVars(M_list, N_list, S_list, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
    gamma_lin = model.addVars(M_list, N_list, S_list, F_fracs, vtype=GRB.CONTINUOUS, lb=0.0)
    T_aux   = model.addVars(M_list, N_list, S_list, vtype=GRB.CONTINUOUS, lb=0)
    T_din   = model.addVars(M_list, N_list, S_list, vtype=GRB.CONTINUOUS, lb=0)

    # --- Fixar variáveis binárias na solução da metaheurística ---
    for i in M_list:
        for j in N_list:
            for s in S_list:
                frac_val = frac_matrix.get(j, {}).get(i, 0.0)
                sp_val   = sprint_asg.get(j, 1)
                w_val    = 1 if (frac_val > 1e-9 and sp_val == s) else 0
                w[i, j, s].lb = w[i, j, s].ub = w_val
                frac_12 = round(frac_val * 12) if w_val else 0
                for f in F_fracs:
                    lv = 1 if (w_val and f == frac_12) else 0
                    lmbda[i, j, s, f].lb = lmbda[i, j, s, f].ub = lv

    for j in N_list:
        sp_val = sprint_asg.get(j, 1)
        for s in S_list:
            z[j, s].lb = z[j, s].ub = (1 if s == sp_val else 0)

    # --- Restrições (Eq. 1–67, idênticas ao optimizer_corrigido) ---
    for i in M_list:
        for s in S_list:
            model.addConstr(gp.quicksum(w[i, j, s] for j in N_list) <= 5)   # Eq. 16
            aux_ctx  = model.addVar(lb=-GRB.INFINITY)
            temp_ctx = model.addVar(lb=0)
            model.addConstr(aux_ctx == gp.quicksum(w[i, j, s] for j in N_list) - 1)
            model.addGenConstrMax(temp_ctx, [aux_ctx], 0.0)
            model.addConstr(P_ctx[i, s] == 0.20 * temp_ctx)   # Eq. 17
            model.addConstr(Cg_idx[i, s] == gp.quicksum(
                w[i, j, s] * f_beh_dict[(i, j)] for j in N_list) / 30.0)   # Eq. 19
            aux_phi  = model.addVar(lb=-GRB.INFINITY)
            temp_phi = model.addVar(lb=0)
            model.addConstr(aux_phi == Cg_idx[i, s] - l_fat)
            model.addGenConstrMax(temp_phi, [aux_phi], 0.0)
            model.addConstr(P_phi[i, s] == phi_fat * temp_phi)   # Eq. 20

    for j in N_list:
        j_idx = int(j.replace("T", ""))
        for s in S_list:
            model.addConstr(P_com[j, s] == 0.05 * gp.quicksum(
                (d * (d - 1) / 2) * u[j, s, d] for d in range(1, S_j[j] + 1)))   # Eq. 18
            model.addConstr(gp.quicksum(u[j, s, d] for d in range(1, S_j[j] + 1)) == z[j, s])  # Eq. 6
            model.addConstr(gp.quicksum(w[i, j, s] for i in M_list) ==
                            gp.quicksum(d * u[j, s, d] for d in range(1, S_j[j] + 1)))  # Eq. 7
            for p in P_quads:
                peq = gp.quicksum(
                    gp.quicksum((Pd[i_i][p] / d) * y[i, j, s, d]
                                for d in range(2, S_j[j] + 1))
                    for i_i, i in enumerate(M_list))
                soma_u = gp.quicksum(u[j, s, d] for d in range(2, S_j[j] + 1))
                model.addConstr(Dev_var[j, s, p] >= peq - 0.25 * soma_u)   # Eq. 13
                model.addConstr(Dev_var[j, s, p] >= 0.25 * soma_u - peq)   # Eq. 14
            model.addConstr(P_match[j, s] == 0.5 * gp.quicksum(
                Dev_var[j, s, p] for p in P_quads) / 1.5)   # Eq. 15

            for i in M_list:
                grupo_inicio = (j_idx // 10) * 10
                eixo_ids = [k for k in range(grupo_inicio, grupo_inicio + 10)
                            if k != j_idx and k < len(N_list)]
                b_eixo = gp.quicksum(
                    w[i, f"T{k}", sp] * (lmbda_apr ** (s - sp))
                    for k in eixo_ids for sp in range(1, s))   # Eq. 21
                b_dep = gp.quicksum(
                    w[i, f"T{k}", sp] * (lmbda_dep ** (pred_dist[j_idx][k] - 1))
                    for k in pred_dist[j_idx] for sp in S_list)   # Eq. 22
                model.addConstr(B_apr[i, j, s] == alpha_eixo * b_eixo + alpha_dep * b_dep)  # Eq. 23
                model.addConstr(w[i, j, s] == gp.quicksum(lmbda[i, j, s, f] for f in F_fracs))  # Eq. 8
                model.addConstr(x[i, j, s] == gp.quicksum(
                    f * lmbda[i, j, s, f] for f in F_fracs) / 12.0)   # Eq. 39
                model.addConstr(F_total[i, j, s] == (1.0 + P_ctx[i, s] + P_com[j, s]
                                                      + P_phi[i, s] + P_match[j, s]
                                                      - B_apr[i, j, s]))   # Eq. 24
                M_max, M_min = 100.0, -100.0
                for f in F_fracs:
                    model.addConstr(gamma_lin[i, j, s, f] <= M_max * lmbda[i, j, s, f])   # Eq. 25
                    model.addConstr(gamma_lin[i, j, s, f] >= M_min * lmbda[i, j, s, f])   # Eq. 26
                    model.addConstr(gamma_lin[i, j, s, f] <= F_total[i, j, s] - M_min * (1 - lmbda[i, j, s, f]))  # Eq. 27
                    model.addConstr(gamma_lin[i, j, s, f] >= F_total[i, j, s] - M_max * (1 - lmbda[i, j, s, f]))  # Eq. 28
                model.addConstr(T_aux[i, j, s] == T_base_dict[(i, j)] * (1.0 / 12.0) *
                                gp.quicksum(f * gamma_lin[i, j, s, f] for f in F_fracs))
                model.addConstr(T_din[i, j, s] == T_aux[i, j, s] + w[i, j, s] * Setup_dev)   # Eq. 29
                model.addConstr(Dur[j] >= T_din[i, j, s])   # Eq. 34
                for d in range(1, S_j[j] + 1):
                    model.addConstr(y[i, j, s, d] <= w[i, j, s])   # Eq. 9
                    model.addConstr(y[i, j, s, d] <= u[j, s, d])   # Eq. 10
                    model.addConstr(y[i, j, s, d] >= w[i, j, s] + u[j, s, d] - 1)   # Eq. 11

    for j in N_list:
        jt = int(j.replace("T", ""))
        model.addConstr(gp.quicksum(
            f * lmbda[i, j, s, f] for i in M_list for s in S_list for f in F_fracs) == 12)  # Eq. 30
        model.addConstr(gp.quicksum(z[j, s] for s in S_list) == 1)   # Eq. 32
        for s in S_list:
            model.addConstr(z[j, s] == gp.quicksum(x[i, j, s] for i in M_list))   # Eq. 31: z = Σx
            model.addConstr(gp.quicksum(w[i, j, s] for i in M_list) <= 4)   # Eq. 38
            model.addConstr(Inic[j] >= ((s - 1) * delta * j_lab) * z[j, s])   # Eq. 33
        model.addConstr(Inic[j] + Dur[j] <= gp.quicksum(
            (s * delta * j_lab) * z[j, s] for s in S_list))   # Eq. 35
        model.addConstr(gp.quicksum(s * z[j, s] for s in S_list) <= S_max_deadline[j])   # Eq. 36
        for h in H:   # Eq. 37: cobertura por skill (HdT_ijh)
            model.addConstr(gp.quicksum(
                hdt(i_idx, jt, h) * w[i, j, s]
                for i_idx, i in enumerate(M_list) for s in S_list) >= 1)
        for k in pred_indices[jt]:
            model.addConstr(Inic[j] >= Inic[f"T{k}"] + Dur[f"T{k}"] + Gap_prec)   # Eq. 40

    for i in M_list:
        for s in S_list:
            model.addConstr(gp.quicksum(T_din[i, j, s] for j in N_list) <=
                            (Cap - Cerim_s) * (1 - Buffer))   # Eq. 45/46: CapUtil=(Cap-Cerim)(1-Buffer)
            for j in N_list:
                for k in N_list:
                    if j != k:
                        model.addConstr(v[i, j, k, s] + v[i, k, j, s] >= w[i, j, s] + w[i, k, s] - 1)  # Eq. 41
                        model.addConstr(Inic[k] >= Inic[j] + T_din[i, j, s]
                                        - BIG_M * (1 - v[i, j, k, s]))   # Eq. 42
                        model.addConstr(v[i, j, k, s] + v[i, k, j, s] <= w[i, j, s])   # Eq. 43
                        model.addConstr(v[i, j, k, s] + v[i, k, j, s] <= w[i, k, s])   # Eq. 44

    for j in N_list:
        model.addConstr(Tmax >= Inic[j] + Dur[j])   # Eq. 47

    model.setObjective(Tmax, GRB.MINIMIZE)
    model.optimize()

    status = model.Status
    factivel = model.SolCount > 0
    tmax_gurobi = model.ObjVal if factivel else None

    resultado = {
        "cenario":      nome_cenario,
        "factivel":     factivel,
        "status_gurobi": status,
        "tmax_gurobi":  tmax_gurobi,
    }

    tag = "FACTIVEL" if factivel else f"INFACTIVEL (status={status})"
    tmax_str = f"  Tmax_Gurobi={tmax_gurobi:.1f}h" if factivel else ""
    print(f"   [VALIDACAO] {nome_cenario}: {tag}{tmax_str}")

    if not factivel and status == GRB.INFEASIBLE:
        print("   [VALIDACAO] Computando IIS para identificar restricoes violadas...")
        model.computeIIS()
        iis_constrs = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
        print(f"   [VALIDACAO] IIS: {iis_constrs[:10]} ...")

    return resultado


def validate_all(instances_dir: str, vns_results_dir: str):
    resultados = []
    for inst_path in sorted(glob.glob(os.path.join(instances_dir, "*.json"))):
        with open(inst_path) as f:
            nome = json.load(f)["_params"]["Nome"]
        csv_path = os.path.join(vns_results_dir, nome, "allocations.csv")
        if not os.path.exists(csv_path):
            print(f"   [VALIDACAO] CSV nao encontrado: {csv_path}")
            continue
        check = check_timing_direct(inst_path, csv_path)
        res = validate(inst_path, csv_path)
        res["timing_ok"]    = check["timing_ok"]
        res["n_violacoes"]  = check["n_violacoes"]
        resultados.append(res)
    return resultados


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    instances_dir   = os.path.join(base, "data", "instances")
    vns_results_dir = os.path.join(base, "data", "results_v2", "vns")
    if len(sys.argv) == 3:
        validate(sys.argv[1], sys.argv[2])
    else:
        validate_all(instances_dir, vns_results_dir)
