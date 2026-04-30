import os
import json
import glob
import time
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_managerial_poc(file_path):
    with open(file_path, 'r') as f: data = json.load(f)
    nome_cenario = data["_params"]["Nome"]
    
    # Parâmetros Calibrados
    alpha_eixo, lmbda_apr = 0.05, 0.80
    alpha_dep, lmbda_dep = 0.20, 0.60
    l_fat, phi_fat = 0.10, 9.0

    M = [f"Dev_{i}" for i in range(data["M"])]
    N = [f"T{j}" for j in range(data["N"])]
    S = list(range(1, data["S_max"] + 1))
    H, P_quads = list(range(data["H_size"])), list(range(data["P_size"]))
    F_fracs = [3, 4, 6, 8, 9, 12] 
    
    Hd, Ht, Pd, Pt, T_nom = data["Hd"], data["Ht"], data["Pd"], data["Pt"], data["T_base"]
    Cerim_s, Buffer, Setup_dev = 8, 0.1, 2.0
    S_max_j = {j: 4 for j in N} 
    BIG_M = 100000 

    pred_indices = data["Pred"]
    pred_dist = {j_idx: {} for j_idx in range(len(N))}
    for j_idx in range(len(N)):
        queue = [(j_idx, 0)]
        visited = set()
        while queue:
            current, dist = queue.pop(0)
            for p_idx in pred_indices[current]:
                if p_idx not in visited:
                    visited.add(p_idx)
                    pred_dist[j_idx][p_idx] = dist + 1
                    queue.append((p_idx, dist + 1))

    # Matrizes Técnicas e Comportamentais Pre-calculadas
    f_beh_dict, T_base_dict, F_tech_dict = {}, {}, {}
    for i_idx, i in enumerate(M):
        for j_idx, j in enumerate(N):
            # Fator Técnico (Excedente)
            et_val = 0.5 * sum(max(0, Hd[i_idx][h] - Ht[j_idx][h])**2 for h in H) / (len(H) * 25.0)
            f_tech = max(0.1, 1.0 - et_val)
            F_tech_dict[(i,j)] = f_tech
            
            # Fator Comportamental (CORREÇÃO DE NORMALIZAÇÃO AQUI)
            # A diferença máxima possível em 4 quadrantes (1 a 5) é 4*4 = 16.
            psi = sum(abs(Pd[i_idx][p] - Pt[j_idx][p]) for p in P_quads) / (len(P_quads) * 4.0)
            f_beh_val = 1.0 + (5.0 * (psi**3))
            f_beh_dict[(i,j)] = f_beh_val
            
            # Tempo Base Calculado
            T_base_dict[(i,j)] = T_nom[j_idx] * f_beh_val * f_tech

    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    
    model = gp.Model("SPSP_MILP", env=env)
    
    model.Params.TimeLimit = 36000 # 10 de tempo limite
    model.Params.MIPGap = 0.05 # Permitir até 5% de gap para encontrar soluções mais rapidamente, dada a complexidade do modelo.
    
    lmbda = model.addVars(M, N, S, F_fracs, vtype=GRB.BINARY)
    x = model.addVars(M, N, S, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    z = model.addVars(N, S, vtype=GRB.BINARY)
    w = model.addVars(M, N, S, vtype=GRB.BINARY)
    v = model.addVars(M, N, N, S, vtype=GRB.BINARY) 
    Inic = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0)
    Dur = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0)
    Tmax = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
    u = model.addVars(N, S, range(1, 5), vtype=GRB.BINARY) 
    y = model.addVars(M, N, S, range(1, 5), vtype=GRB.BINARY)
    
    P_ctx = model.addVars(M, S, vtype=GRB.CONTINUOUS, lb=0)
    P_com = model.addVars(N, S, vtype=GRB.CONTINUOUS, lb=0)
    Cg_idx = model.addVars(M, S, vtype=GRB.CONTINUOUS, lb=0)
    P_phi = model.addVars(M, S, vtype=GRB.CONTINUOUS, lb=0)
    Dev = model.addVars(N, S, P_quads, vtype=GRB.CONTINUOUS, lb=0)
    P_match = model.addVars(N, S, vtype=GRB.CONTINUOUS, lb=0)
    B_apr = model.addVars(M, N, S, vtype=GRB.CONTINUOUS, lb=0)
    F_total = model.addVars(M, N, S, vtype=GRB.CONTINUOUS, lb=-10.0)
    gamma_lin = model.addVars(M, N, S, F_fracs, vtype=GRB.CONTINUOUS, lb=-10.0)
    T_aux = model.addVars(M, N, S, vtype=GRB.CONTINUOUS, lb=0)
    T_din = model.addVars(M, N, S, vtype=GRB.CONTINUOUS, lb=0)

    for i in M:
        for s in S:
            model.addConstr(gp.quicksum(w[i,j,s] for j in N) <= 5)
            
            aux_ctx = model.addVar(lb=-GRB.INFINITY)
            temp_ctx = model.addVar(lb=0)
            model.addConstr(aux_ctx == gp.quicksum(w[i,j,s] for j in N) - 1)
            model.addGenConstrMax(temp_ctx, [aux_ctx], 0.0)
            model.addConstr(P_ctx[i,s] == 0.20 * temp_ctx)
            
            model.addConstr(Cg_idx[i,s] == gp.quicksum(w[i,j,s] * f_beh_dict[(i,j)] for j in N) / 30.0)
            aux_phi = model.addVar(lb=-GRB.INFINITY)
            temp_phi = model.addVar(lb=0)
            model.addConstr(aux_phi == Cg_idx[i,s] - l_fat)
            model.addGenConstrMax(temp_phi, [aux_phi], 0.0)
            model.addConstr(P_phi[i,s] == phi_fat * temp_phi)

    for j in N:
        j_idx = int(j.replace("T",""))
        for s in S:
            model.addConstr(P_com[j,s] == 0.05 * gp.quicksum((d*(d-1)/2) * u[j,s,d] for d in range(1, S_max_j[j]+1)))
            model.addConstr(gp.quicksum(u[j,s,d] for d in range(1, S_max_j[j]+1)) == z[j,s])
            model.addConstr(gp.quicksum(w[i,j,s] for i in M) == gp.quicksum(d * u[j,s,d] for d in range(1, S_max_j[j]+1)))
            for p in P_quads:
                peq = gp.quicksum(gp.quicksum((Pd[i_idx][p]/d) * y[i,j,s,d] for d in range(2, S_max_j[j]+1)) for i_idx, i in enumerate(M))
                soma_u = gp.quicksum(u[j,s,d] for d in range(2, S_max_j[j]+1))
                model.addConstr(Dev[j,s,p] >= peq - 0.25 * soma_u)
                model.addConstr(Dev[j,s,p] >= 0.25 * soma_u - peq)
            model.addConstr(P_match[j,s] == 0.5 * (gp.quicksum(Dev[j,s,p] for p in P_quads) / 1.5))
            
            for i in M:
                grupo_inicio = (j_idx // 10) * 10
                eixo_indices = [k_idx for k_idx in range(grupo_inicio, grupo_inicio + 10) if k_idx != j_idx and k_idx < len(N)]
                
                b_eixo_expr = gp.quicksum(w[i, f"T{k_idx}", p] * (lmbda_apr**(s-p)) for k_idx in eixo_indices for p in range(1, s))
                b_dep_expr = gp.quicksum(w[i, f"T{k_idx}", p] * (lmbda_dep**(pred_dist[j_idx][k_idx] - 1)) for k_idx in pred_dist[j_idx] for p in S)
                model.addConstr(B_apr[i,j,s] == (alpha_eixo * b_eixo_expr) + (alpha_dep * b_dep_expr))
                model.addConstr(w[i,j,s] == gp.quicksum(lmbda[i,j,s,f] for f in F_fracs))
                model.addConstr(x[i,j,s] == gp.quicksum(f * lmbda[i,j,s,f] for f in F_fracs) / 12.0)
                model.addConstr(F_total[i,j,s] == 1.0 + P_ctx[i,s] + P_com[j,s] + P_phi[i,s] + P_match[j,s] - B_apr[i,j,s])
                M_max, M_min = 100.0, -100.0
                for f in F_fracs:
                    model.addConstr(gamma_lin[i,j,s,f] <= M_max * lmbda[i,j,s,f])
                    model.addConstr(gamma_lin[i,j,s,f] >= M_min * lmbda[i,j,s,f])
                    model.addConstr(gamma_lin[i,j,s,f] <= F_total[i,j,s] - M_min * (1 - lmbda[i,j,s,f]))
                    model.addConstr(gamma_lin[i,j,s,f] >= F_total[i,j,s] - M_max * (1 - lmbda[i,j,s,f]))
                model.addConstr(T_aux[i,j,s] == T_base_dict[(i,j)] * (1.0/12.0) * gp.quicksum(f * gamma_lin[i,j,s,f] for f in F_fracs))
                model.addConstr(T_din[i,j,s] == T_aux[i,j,s] + (w[i,j,s] * Setup_dev))
                model.addConstr(Dur[j] >= T_din[i,j,s])
                for d in range(1, S_max_j[j]+1):
                    model.addConstr(y[i,j,s,d] <= w[i,j,s])
                    model.addConstr(y[i,j,s,d] <= u[j,s,d])
                    model.addConstr(y[i,j,s,d] >= w[i,j,s] + u[j,s,d] - 1)

    # Backlog e Capacidade
    for j in N:
        model.addConstr(gp.quicksum(f * lmbda[i,j,s,f] for i in M for s in S for f in F_fracs) == 12)
        model.addConstr(gp.quicksum(z[j,s] for s in S) == 1)
        for s in S:
            model.addConstr(z[j,s] <= gp.quicksum(w[i,j,s] for i in M))
            model.addConstr(Inic[j] >= ((s-1)*14*8) * z[j,s])
        model.addConstr(Inic[j] + Dur[j] <= gp.quicksum((s*14*8) * z[j,s] for s in S))
        for k in data["Pred"][int(j.replace("T",""))]:
            model.addConstr(Inic[j] >= Inic[f"T{k}"] + Dur[f"T{k}"] + data["Gap_prec"])

    for i in M:
        for s in S:
            model.addConstr(gp.quicksum(T_din[i,j,s] for j in N) <= (14*8 - Cerim_s) * (1 - Buffer))
            for j in N:
                for k in N:
                    if j != k:
                        model.addConstr(v[i,j,k,s] + v[i,k,j,s] >= w[i,j,s] + w[i,k,s] - 1)
                        model.addConstr(Inic[k] >= Inic[j] + T_din[i,j,s] - BIG_M * (1 - v[i,j,k,s]))

    for j in N: model.addConstr(Tmax >= Inic[j] + Dur[j])

    print(f"⌛ Otimizando '{nome_cenario}' (Aguarde, calculando MILP puro)...")
    start_time = time.time()
    model.setObjective(Tmax, GRB.MINIMIZE)
    model.optimize()
    runtime = time.time() - start_time
    
    resultado = {
        "Cenario": nome_cenario,
        "Status": "Falha/Infactível",
        "MIPGap (%)": "N/A",
        "Tempo (s)": round(runtime, 2)
    }
    
    if model.SolCount > 0:
        gap_pct = model.MIPGap * 100.0
        resultado["Status"] = "Sucesso"
        resultado["MIPGap (%)"] = f"{gap_pct:.2f}%"
        
        os.makedirs(f"data/results/{nome_cenario}", exist_ok=True)
        allocs = []
        for i in M:
            for j in N:
                for s in S:
                    if x[i,j,s].X > 0.001: 
                        allocs.append({
                            "Sprint": s, 
                            "Dev": i, 
                            "Tarefa": j, 
                            "Fracao": x[i,j,s].X, 
                            "T_Din": T_din[i,j,s].X, 
                            "Inicio": Inic[j].X,
                            "Fim": Inic[j].X + Dur[j].X,
                            "Duracao_Tarefa": Dur[j].X,
                            "T_Nominal_Original": T_nom[int(j.replace("T",""))],
                            "T_Base_Calculado": T_base_dict[(i,j)],
                            "F_Tech_ET": F_tech_dict[(i,j)],
                            "F_Beh_EC": f_beh_dict[(i,j)],
                            "P_Ctx": P_ctx[i,s].X,
                            "P_Com": P_com[j,s].X,
                            "P_Phi_Fadiga": P_phi[i,s].X,
                            "P_Match": P_match[j,s].X,
                            "B_Apr_Acumulado": B_apr[i,j,s].X,
                            "F_Total_Multiplicador": F_total[i,j,s].X,
                            "Cg_Idx_Sustentabilidade": Cg_idx[i,s].X,
                            "MIPGap": gap_pct,
                            "Runtime": runtime
                        })
        pd.DataFrame(allocs).to_csv(f"data/results/{nome_cenario}/allocations_super.csv", index=False)
    
    return resultado

if __name__ == "__main__":
    start_total = time.time()
    
    resultados_finais = []
    
    arquivos = sorted(glob.glob("data/instances/*.json"))
    for f in arquivos: 
        res = solve_managerial_poc(f)
        resultados_finais.append(res)
        
    end_total = time.time() - start_total
    horas = int(end_total // 3600)
    minutos = int((end_total % 3600) // 60)
    segundos = end_total % 60
    
    print("\n" + "="*70)
    print("📊 RESUMO DE DESEMPENHO COMPUTACIONAL (SBPO)")
    print("="*70)
    print(f"{'Cenário':<35} | {'Status':<10} | {'MIPGap':<10} | {'Tempo (s)':<10}")
    print("-" * 70)
    for r in resultados_finais:
        print(f"{r['Cenario']:<35} | {r['Status']:<10} | {r['MIPGap (%)']:>10} | {r['Tempo (s)']:>8.2f}s")
    print("="*70)
    print(f"🚀 Tempo Total da Bateria: {horas}h {minutos}m {segundos:.2f}s")
    print("="*70 + "\n")
