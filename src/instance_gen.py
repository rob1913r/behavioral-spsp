# -*- coding: utf-8 -*-
"""
Módulo de Geração de Instâncias para o Problema de Escalonamento de Projetos de Software (SPSP).

Este script gera os conjuntos de dados sintéticos em formato JSON utilizados pelos experimentos
computacionais. A reprodutibilidade matemática é garantida através de uma semente global (SEED),
assegurando que distribuições estatísticas de habilidade e comportamento sejam sempre idênticas.
"""

import os
import json
import random
import numpy as np

# Configurações globais de reprodutibilidade
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)

# Diretório base de saída
BASE_OUTPUT_DIR = os.path.join("data", "instances")

def random_dirichlet(size=4, concentration=1.0):
    """
    Gera uma distribuição de probabilidade estocástica para os perfis comportamentais.
    A soma vetorial dos valores gerados é rigidamente 1.0.
    """
    return np.random.dirichlet([concentration] * size).tolist()

def blend_profile(target, noise, size=4):
    """
    Mescla um perfil alvo com um nível de ruído aleatório, simulando desvios de personalidade.
    Mantém a consistência da soma vetorial igual a 1.0.
    """
    noise_vec = np.random.dirichlet([1.0] * size)
    blended = (1.0 - noise) * np.array(target) + noise * noise_vec
    blended = np.clip(blended, 1e-9, None)
    return (blended / blended.sum()).tolist()

def generate_instances():
    """
    Orquestra a geração das instâncias paramétricas divididas nos 
    cinco grupos de validação propostos na modelagem, salvando-as em subpastas específicas.
    """
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Parâmetros estruturais fixos do modelo
    H_size = 4  # Quantidade de habilidades técnicas
    P_size = 4  # Quantidade de quadrantes comportamentais (Ex: Metodologia Solides)
    
    FORMULA_FINAL = "CUBIC" 
    K_FINAL = 5.0
    DAYS_PER_SPRINT = 14
    HOURS_PER_DAY = 8

    print(f"\nIniciando a geração de instâncias computacionais no diretório '{BASE_OUTPUT_DIR}'...\n")

    # =====================================================================
    # GRUPO 1: Escalabilidade do MILP
    # =====================================================================
    exp1_dir = os.path.join(BASE_OUTPUT_DIR, "exp1_scalability")
    os.makedirs(exp1_dir, exist_ok=True)
    print(f"-> Gerando Grupo 1: Escalabilidade (Salvando em: {exp1_dir})...")
    
    for n in [10, 20, 30, 40, 50, 60, 70, 80]:
        for r in range(30):
            m, s_max = 5, n
            
            Ht = [[random.uniform(0.05, 0.25) for _ in range(H_size)] for _ in range(n)]
            Pt = [random_dirichlet(P_size) for _ in range(n)]
            Hd = [[random.uniform(0.50, 1.0) for _ in range(H_size)] for _ in range(m)]
            Pd = [random_dirichlet(P_size) for _ in range(m)]
            T_base = [random.uniform(20.0, 60.0) for _ in range(n)] 
            
            Pred = [[] for _ in range(n)]
            for j in range(1, n):
                if random.random() < 0.05: 
                    Pred[j].append(random.randint(max(0, j-3), j-1))
            
            inst = {
                "M": m, "N": n, "S_max": s_max, "H_size": H_size, "P_size": P_size, 
                "delta_s": DAYS_PER_SPRINT, "j_lab": HOURS_PER_DAY,
                "Hd": Hd, "Ht": Ht, "Pd": Pd, "Pt": Pt, "T_base": T_base, 
                "Cap": [[160.0] * s_max] * m, "Pred": Pred,
                "_group": "Exp1_Scalability", 
                "_params": {"rep": r, "formula_type": FORMULA_FINAL, "k": K_FINAL}
            }
            with open(os.path.join(exp1_dir, f"Exp1_N{n:03d}_r{r:02d}.json"), "w") as f: 
                json.dump(inst, f)

    # =====================================================================
    # GRUPO 2: Ruído Comportamental (Cenários Controlados)
    # =====================================================================
    exp2_dir = os.path.join(BASE_OUTPUT_DIR, "exp2_behavioral")
    os.makedirs(exp2_dir, exist_ok=True)
    print(f"-> Gerando Grupo 2: Ruído Comportamental (Salvando em: {exp2_dir})...")
    
    cenarios = [
        ("0_Baseline", 0.0), 
        ("1_Natural", 0.2),  
        ("2_Critico", 0.6),  
        ("3_Extremo", 1.0)   
    ]
    
    for r in range(30):
        n, m, s_max = 30, 5, 30
        
        # Geração Isolada: Cria as tarefas e a senioridade técnica uma única vez por repetição.
        # Assim, os 4 cenários enfrentam o mesmo projeto.
        Ht_base = [[random.uniform(0.05, 0.25) for _ in range(H_size)] for _ in range(n)]
        T_base_val = [random.uniform(30.0, 80.0) for _ in range(n)] 
        Hd_base = [[random.uniform(0.50, 1.0) for _ in range(H_size)] for _ in range(m)]
        
        # Perfis naturais padrão para usar no Baseline e no Natural
        Pt_natural = [random_dirichlet(P_size) for _ in range(n)]
        Pd_natural = [random_dirichlet(P_size) for _ in range(m)]

        for nome_cenario, nivel in cenarios:
            if nome_cenario == "3_Extremo":
                Pt = [[0.01, 0.01, 0.01, 0.97] for _ in range(n)]
                Pd = [[0.97, 0.01, 0.01, 0.01] for _ in range(m)]
            elif nome_cenario == "2_Critico":
                Pt = [[0.1, 0.7, 0.1, 0.1] for _ in range(n)] 
                Pd = [[0.6, 0.1, 0.2, 0.1] for _ in range(m)] 
            else:
                Pt = Pt_natural
                Pd = Pd_natural

            inst = {
                "M": m, "N": n, "S_max": s_max, "H_size": H_size, "P_size": P_size, 
                "delta_s": DAYS_PER_SPRINT, "j_lab": HOURS_PER_DAY,
                "Hd": Hd_base, "Ht": Ht_base, "Pd": Pd, "Pt": Pt, "T_base": T_base_val, 
                "Cap": [[9999.0] * s_max] * m, "Pred": [[] for _ in range(n)],
                "_group": "Exp2_Behavioral", 
                "_params": {"Cenario": nome_cenario, "rep": r, "formula_type": FORMULA_FINAL, "k": K_FINAL}
            }
            with open(os.path.join(exp2_dir, f"Exp2_{nome_cenario}_r{r:02d}.json"), "w") as f: 
                json.dump(inst, f)

    # =====================================================================
    # GRUPO 3: Planejamento de Capacidade Tática e Lei de Brooks
    # =====================================================================
    exp3_dir = os.path.join(BASE_OUTPUT_DIR, "exp3_heatmap")
    os.makedirs(exp3_dir, exist_ok=True)
    print(f"-> Gerando Grupo 3: Lei de Brooks (Salvando em: {exp3_dir})...")
    
    N_vals = [25, 30, 35, 40, 45, 50, 55, 60]
    M_vals = [3, 4, 5, 6, 7, 8]
    
    for r in range(15):
        # Gera um pool comum para manter a comparabilidade isolada ao redimensionar a equipe
        Ht_pool = [[random.uniform(0.05, 0.25) for _ in range(H_size)] for _ in range(max(N_vals))]
        Pt_pool = [random_dirichlet(P_size) for _ in range(max(N_vals))]
        T_base_pool = [random.uniform(30.0, 80.0) for _ in range(max(N_vals))] 
        Hd_pool = [[random.uniform(0.50, 1.0) for _ in range(H_size)] for _ in range(max(M_vals))]
        Pd_pool = [random_dirichlet(P_size) for _ in range(max(M_vals))]
        
        for n in N_vals:
            for m in M_vals:
                inst = {
                    "M": m, "N": n, "S_max": n, "H_size": H_size, "P_size": P_size, 
                    "delta_s": DAYS_PER_SPRINT, "j_lab": HOURS_PER_DAY,
                    "Hd": Hd_pool[:m], "Ht": Ht_pool[:n], "Pd": Pd_pool[:m], "Pt": Pt_pool[:n], 
                    "T_base": T_base_pool[:n], "Cap": [[160.0] * n] * m, "Pred": [[] for _ in range(n)],
                    "_group": "Exp3_Heatmap", 
                    "_params": {"rep": r, "formula_type": FORMULA_FINAL, "k": K_FINAL}
                }
                with open(os.path.join(exp3_dir, f"Exp3_M{m:02d}_N{n:02d}_r{r:02d}.json"), "w") as f: 
                    json.dump(inst, f)

    # =====================================================================
    # GRUPO 4: Densidade de Precedência
    # =====================================================================
    exp4_dir = os.path.join(BASE_OUTPUT_DIR, "exp4_precedence")
    os.makedirs(exp4_dir, exist_ok=True)
    print(f"-> Gerando Grupo 4: Densidade de Precedência (Salvando em: {exp4_dir})...")
    
    for d in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for r in range(30):
            n, m, s_max = 25, 5, 8 
            
            Ht = [[random.uniform(0.05, 0.25) for _ in range(H_size)] for _ in range(n)]
            Pt = [random_dirichlet(P_size) for _ in range(n)]
            Hd = [[random.uniform(0.50, 1.0) for _ in range(H_size)] for _ in range(m)]
            Pd = [random_dirichlet(P_size) for _ in range(m)]
            T_base = [random.uniform(20.0, 60.0) for _ in range(n)]
            
            Pred = [[] for _ in range(n)]
            for j in range(1, n):
                for k in range(max(0, j - 4), j):
                    if random.random() < d: 
                        Pred[j].append(k)
            
            inst = {
                "M": m, "N": n, "S_max": s_max, "H_size": H_size, "P_size": P_size, 
                "delta_s": DAYS_PER_SPRINT, "j_lab": HOURS_PER_DAY,
                "Hd": Hd, "Ht": Ht, "Pd": Pd, "Pt": Pt, "T_base": T_base, 
                "Cap": [[160.0] * s_max] * m, "Pred": Pred,
                "_group": "Exp4_Precedence", 
                "_params": {"Density": d, "rep": r, "formula_type": FORMULA_FINAL, "k": K_FINAL}
            }
            with open(os.path.join(exp4_dir, f"Exp4_Dens{int(d*100):03d}_r{r:02d}.json"), "w") as f: 
                json.dump(inst, f)

    # =====================================================================
    # GRUPO 5: Compensação Técnica vs Ruído Comportamental
    # =====================================================================
    exp5_dir = os.path.join(BASE_OUTPUT_DIR, "exp5_surplus")
    os.makedirs(exp5_dir, exist_ok=True)
    print(f"-> Gerando Grupo 5: Compensação Técnica vs Comportamento (Salvando em: {exp5_dir})...\n")
    
    for tech_lvl in [0.3, 0.5, 0.7, 0.9, 1.0]: 
        for r in range(30):
            n, m, s_max = 30, 5, 30
            
            Ht = [[random.uniform(0.15, 0.25) for _ in range(H_size)] for _ in range(n)]
            Pt = [random_dirichlet(P_size) for _ in range(n)]
            Hd = [[random.uniform(tech_lvl - 0.1, tech_lvl) for _ in range(H_size)] for _ in range(m)]
            
            # Força o alinhamento comportamental para isolar a variável de qualificação técnica
            Pd = Pt[:m] + [Pt[0]] * (m - len(Pt)) 
            T_base = [random.uniform(30.0, 80.0) for _ in range(n)]
            
            inst = {
                "M": m, "N": n, "S_max": s_max, "H_size": H_size, "P_size": P_size, 
                "delta_s": DAYS_PER_SPRINT, "j_lab": HOURS_PER_DAY,
                "Hd": Hd, "Ht": Ht, "Pd": Pd, "Pt": Pt, "T_base": T_base, 
                "Cap": [[160.0] * s_max] * m, "Pred": [[] for _ in range(n)],
                "_group": "Exp5_Surplus", 
                "_params": {"TechLvl": tech_lvl, "rep": r, "formula_type": FORMULA_FINAL, "k": K_FINAL}
            }
            with open(os.path.join(exp5_dir, f"Exp5_Tech{int(tech_lvl*100):03d}_r{r:02d}.json"), "w") as f: 
                json.dump(inst, f)

if __name__ == "__main__":
    generate_instances()
