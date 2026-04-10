# -*- coding: utf-8 -*-
"""
Módulo de Otimização para o Problema de Escalonamento de Projetos de Software (SPSP).

Este script processa as instâncias JSON geradas, formula o modelo matemático
em Programação Linear Inteira Mista (MILP) e o submete ao solver Gurobi.
Os resultados são exportados para um arquivo CSV unificado contendo as métricas
de execução e o makespan final de cada instância.
"""

import os
import glob
import json
import time
import datetime
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Caminhos padronizados para o repositório
INPUT_DIR = os.path.join("data", "instances")
RESULTS_FILE = os.path.join("data", "results", "metrics.csv")

def run_solver(filepath):
    """
    Constrói e resolve o modelo MILP para uma instância específica.
    Retorna um dicionário com as métricas de convergência e dados do projeto.
    """
    with open(filepath, "r", encoding="utf-8") as f: 
        d = json.load(f)
        
    M, N, S_max = d["M"], d["N"], d["S_max"]
    H_size, P_size = d["H_size"], d["P_size"]
    group = d.get("_group", "G_Default")
    params = d.get("_params", {})

    T_ij = np.zeros((M, N))
    EC = np.zeros((M, N))
    ET = np.zeros((M, N))
    k_val = params.get("k", 5.0)

    # Pré-computação das métricas de tempo, excedente técnico e erro comportamental
    for i in range(M):
        for j in range(N):
            ET[i, j] = sum((max(0, d["Hd"][i][h] - d["Ht"][j][h])) ** 2 for h in range(H_size)) / H_size
            
            # Anula o erro comportamental apenas no cenário de Baseline estrito
            if group == "exp2_behavioral" and params.get("Cenario", "") == "0_Baseline":
                EC[i, j] = 0.0
            else:
                x_mismatch = sum(abs(d["Pd"][i][p] - d["Pt"][j][p]) for p in range(P_size)) / 2.0
                EC[i, j] = k_val * (x_mismatch ** 3)
            
            T_ij[i, j] = d["T_base"][j] * (1.0 + EC[i, j] - 0.5 * ET[i, j])

    # Inicialização do modelo Gurobi
    model = gp.Model("TaskAllocation")
    model.setParam("OutputFlag", 0)   # Execução silenciosa
    model.setParam("TimeLimit", 3600) # Limite de 1 hora por instância
    model.setParam("MIPGap", 0.1)     # MIPGap de 0,1%

    # Variáveis de Decisão
    x = model.addVars(M, N, S_max, vtype=GRB.BINARY, name="x")
    y = model.addVars(S_max, vtype=GRB.BINARY, name="y")
    TF = model.addVars(S_max, vtype=GRB.CONTINUOUS, lb=0, name="TF")
    Tmax = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Tmax")
    
    model.setObjective(Tmax, GRB.MINIMIZE)

    # Restrições
    # 1. Atribuição única
    for j in range(N): 
        model.addConstr(gp.quicksum(x[i, j, s] for i in range(M) for s in range(S_max)) == 1)
        
    # 2. Qualificação técnica mínima
    for j in range(N):
        for h in range(H_size):
            model.addConstr(gp.quicksum(d["Hd"][i][h] * x[i, j, s] for i in range(M) for s in range(S_max)) >= d["Ht"][j][h])
            
    # 3. Capacidade da sprint
    for i in range(M):
        for s in range(S_max): 
            model.addConstr(gp.quicksum(T_ij[i, j] * x[i, j, s] for j in range(N)) <= d["Cap"][i][s])
            
    # 4. Precedência estrutural
    for j in range(N):
        for pred_k in d["Pred"][j]:
            model.addConstr(gp.quicksum((s+1)*x[i,j,s] for i in range(M) for s in range(S_max)) >= 
                            gp.quicksum((s+1)*x[i,pred_k,s] for i in range(M) for s in range(S_max)) + 1)
                            
    # 5. Lógicas de ativação de sprint e sequenciamento
    for s in range(S_max):
        model.addConstr(gp.quicksum(x[i, j, s] for i in range(M) for j in range(N)) >= y[s])
        if s > 0: 
            model.addConstr(y[s - 1] >= y[s])
        for i in range(M):
            for j in range(N): 
                model.addConstr(x[i, j, s] <= y[s])

    # 6. Conversão de esforço em tempo calendário
    for i in range(M):
        model.addConstr(gp.quicksum(x[i, j, 0] * T_ij[i, j] for j in range(N)) / d["j_lab"] <= TF[0])
        for s in range(1, S_max):
            model.addConstr(d["delta_s"] * s * y[s] + gp.quicksum(x[i, j, s] * T_ij[i, j] for j in range(N)) / d["j_lab"] <= TF[s])
            
    # 7. Função objetivo engloba o tempo final
    for s in range(S_max): 
        model.addConstr(TF[s] <= Tmax)

    model.update() 
    num_vars = model.NumVars
    
    # Processamento e extração de resultados
    t0 = time.time()
    model.optimize()
    runtime = time.time() - t0

    feasible = model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0
    status_map = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIMEOUT", GRB.INFEASIBLE: "INFEASIBLE"}

    row = {
        "file": os.path.basename(filepath),
        "group": group, "M": M, "N": N, "S_max": S_max,
        "NumVars": num_vars,
        "Density": params.get("Density", 0.0),
        "Cenario": params.get("Cenario", ""),
        "TechLvl": params.get("TechLvl", 0.0),
        "formula_type": "CUBIC",
        "k": k_val,
        "status": status_map.get(model.status, "OTHER"),
        "feasible": int(feasible),
        "runtime": runtime,
        "mksp": Tmax.X if feasible else np.nan,
        "avg_EC": float(EC.mean())
    }
    return row

if __name__ == "__main__":
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    # Busca recursiva para ler as instâncias nas subpastas recém-criadas
    search_pattern = os.path.join(INPUT_DIR, "**", "*.json")
    ficheiros = sorted(glob.glob(search_pattern, recursive=True))
    
    if not ficheiros:
        print(f"[ERRO] Nenhuma instancia encontrada em '{INPUT_DIR}'. Execute 'instance_gen.py' primeiro.")
        exit(1)
        
    print(f"\nIniciando resolucao de {len(ficheiros)} instancias com Gurobi Optimizer...\n")
    
    tempo_inicio_global = time.time()
    
    resultados = []
    for idx, f in enumerate(ficheiros, 1):
        row = run_solver(f)
        resultados.append(row)
        print(f"[{idx:04d}/{len(ficheiros)}] {row['group']} | {row['status']:<10} | Tempo: {row['runtime']:.1f}s")
        
        # Salvamento incremental
        if idx % 20 == 0: 
            pd.DataFrame(resultados).to_csv(RESULTS_FILE, index=False)
    
    # Salvamento final
    pd.DataFrame(resultados).to_csv(RESULTS_FILE, index=False)
    
    tempo_fim_global = time.time()
    duracao_total_segundos = tempo_fim_global - tempo_inicio_global
    duracao_formatada = str(datetime.timedelta(seconds=int(duracao_total_segundos)))
    
    print(f"\nResultados exportados para: '{RESULTS_FILE}'")
    print(f"Tempo total computacional: {duracao_formatada}\n")
