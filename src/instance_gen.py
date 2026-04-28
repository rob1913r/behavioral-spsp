import os
import json

os.makedirs("data/instances", exist_ok=True)

# Parâmetros Expandidos para o SBPO
H_size, P_size = 4, 4
M = 5         # 5 Desenvolvedores
S_max = 15    # Até 15 Sprints (Proj. Longo)

# Habilidades (Hd) e Perfis (Pd) dos 5 Devs (Especialistas + 1 Coringa)
Hd_base = [[5,1,1,1], [1,5,1,1], [1,1,5,1], [1,1,1,5], [3,3,3,3]]
Pd_base = [[5,1,1,1], [1,5,1,1], [1,1,5,1], [1,1,1,5], [3,3,3,3]]

def criar_json(nome, n_tarefas, hd, pd, ht, pt, t_base, pred):
    dados = {
        "_params": {"Nome": nome},
        "M": M, "N": n_tarefas, "H_size": H_size, "P_size": P_size, "S_max": S_max,
        "Hd": hd, "Pd": pd, "Ht": ht, "Pt": pt, "T_base": t_base,
        "Pred": pred, "Gap_prec": 2
    }
    with open(f"data/instances/{nome}.json", "w") as f:
        json.dump(dados, f, indent=4)

def gerar_instancias_sbpo():
    # ---------------------------------------------------------------------
    # CENÁRIO 1: A Maratona Concorrente (Forçar 4+ Tarefas Simultâneas)
    # ---------------------------------------------------------------------
    N_1 = 75
    Ht_1, Pt_1, T_base_1, Pred_1 = [], [], [], [[] for _ in range(N_1)]
    for j in range(N_1):
        # Distribui as tarefas rotativamente entre os perfis ideais
        idx = j % 5
        Ht_1.append(Hd_base[idx])
        Pt_1.append(Pd_base[idx])
        T_base_1.append(12) # Tarefas muito curtas (12h). Cabem umas 6 na Sprint!
        
    criar_json("Cenario_1_Maratona_Multitarefa", N_1, Hd_base, Pd_base, Ht_1, Pt_1, T_base_1, Pred_1)

    # ---------------------------------------------------------------------
    # CENÁRIO 2: Os Monólitos (Forçar Trabalho em Equipe / Frações)
    # ---------------------------------------------------------------------
    N_2 = 40
    Ht_2, Pt_2, T_base_2, Pred_2 = [], [], [], [[] for _ in range(N_2)]
    for j in range(N_2):
        if j % 4 == 0:
            # A cada 4 tarefas, um MONÓLITO ABSURDO que exige perfis mistos
            Ht_2.append([4, 4, 1, 1])
            Pt_2.append([4, 4, 1, 1])
            T_base_2.append(110) # 110h base! Com atrito, passa de 130h. Um dev SOZINHO não consegue fazer na sprint.
        else:
            # Tarefas normais para preencher espaço
            Ht_2.append(Hd_base[j % 5])
            Pt_2.append(Pd_base[j % 5])
            T_base_2.append(25)
            
    criar_json("Cenario_2_Monolitos_Equipe", N_2, Hd_base, Pd_base, Ht_2, Pt_2, T_base_2, Pred_2)

    # ---------------------------------------------------------------------
    # CENÁRIO 3: Gargalo do Especialista (Teste de Precedência Longa)
    # ---------------------------------------------------------------------
    N_3 = 50
    Ht_3, Pt_3, T_base_3, Pred_3 = [], [], [], [[] for _ in range(N_3)]
    for j in range(N_3):
        if j < 25:
            # Cadeia Crítica: Apenas o Dev 0 tem o perfil técnico perfeito (5,1,1,1)
            Ht_3.append([5, 1, 1, 1])
            Pt_3.append([5, 1, 1, 1])
            T_base_3.append(20)
            if j > 0:
                Pred_3[j].append(j-1) # Precedência Estrita
        else:
            # Tarefas soltas para os outros 4 Devs
            Ht_3.append(Hd_base[(j % 4) + 1])
            Pt_3.append(Pd_base[(j % 4) + 1])
            T_base_3.append(30)
            
    criar_json("Cenario_3_Cadeia_Gargalo", N_3, Hd_base, Pd_base, Ht_3, Pt_3, T_base_3, Pred_3)
    
    print("✅ 3 Instâncias Avançadas (SBPO) criadas em data/instances/")

if __name__ == "__main__":
    gerar_instancias_sbpo()
