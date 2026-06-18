# -*- coding: utf-8 -*-
"""
instance_gen.py — gerador determinístico das instâncias SPSP.

Cria os 3 cenários sintéticos em data/instances/*.json (Maratona
Multitarefa, Monolitos em Equipe, Cadeia/Gargalo). É determinístico:
reexecutar produz arquivos idênticos.

ESCALA DOS DADOS (modelo corrigido — ver planning/changes.md):
  - Hd/Ht: habilidades em [0,1], 2 casas (via (x-1)/4); 0=nenhum, 1=expertise máx.
  - Pd/Pt: perfis comportamentais que SOMAM 1, 2 casas (maior-resto/Hamilton).
  - j_lab = 8: jornada diária (horas/dia) usada nas Eq. 33/35.
Regenerar SOBRESCREVE data/instances/*.json (os baselines antigos em
data/results/ ficam obsoletos — rode os 3 solvers corrigidos p/ refazê-los).
"""
import os
import json

os.makedirs("data/instances", exist_ok=True)


def _skill(x):
    """Habilidade 1–5 → [0,1] (Hd/Ht): (x-1)/4, 2 casas. 0=nenhum, 1=expertise máx."""
    return round((x - 1) / 4.0, 2)


def _profile(vec):
    """Perfil (Pd/Pt) → distribuição que SOMA 1, 2 casas (maior-resto/Hamilton)."""
    total = sum(vec)
    scaled = [v / total * 100.0 for v in vec]
    floors = [int(s) for s in scaled]
    resto = 100 - sum(floors)
    ordem = sorted(range(len(vec)), key=lambda k: scaled[k] - floors[k], reverse=True)
    for k in range(resto):
        floors[ordem[k]] += 1
    return [round(f / 100.0, 2) for f in floors]


def _scale_skills(mat):
    return [[_skill(v) for v in row] for row in mat]


def _norm_profiles(mat):
    return [_profile(row) for row in mat]


# Parâmetros Expandidos para o SBPO
H_size, P_size = 4, 4
M = 5         # 5 Desenvolvedores
S_max = 15    # Até 15 Sprints (Proj. Longo)

# Habilidades (Hd) e Perfis (Pd) dos 5 Devs (Especialistas + 1 Coringa)
Hd_base = [[5,1,1,1], [1,5,1,1], [1,1,5,1], [1,1,1,5], [3,3,3,3]]
Pd_base = [[5,1,1,1], [1,5,1,1], [1,1,5,1], [1,1,1,5], [3,3,3,3]]

def write_instance(nome, n_tarefas, hd, pd, ht, pt, t_base, pred, m=None, s_max=None,
                   j_lab=8, delta=14, setup=2.0, cerim=8, buffer=0.1, s_j=4, gap_prec=2,
                   cap=None, s_max_j=None):
    s_max_val = s_max if s_max is not None else S_max
    # Eq. 45: capacidade nominal do recurso por (dev,sprint). Padrão = δ·j_lab
    # (horas úteis brutas da sprint, antes de descontar cerimônias e buffer).
    cap_val = cap if cap is not None else delta * j_lab
    # Eq. 36: prazo (sprint-limite) por tarefa. Padrão = S_max para TODAS as tarefas
    # (sem prazo apertado), mas agora gravado EXPLÍCITO e POR TAREFA no JSON, de
    # modo que a Eq. 36 passa a ser data-driven (cada solver lê o vetor da instância).
    # Apertar o prazo de tarefas específicas aqui é o botão para exercitar a Eq. 36.
    s_max_j_val = s_max_j if s_max_j is not None else [s_max_val] * n_tarefas
    dados = {
        "_params": {"Nome": nome},
        "M": m if m is not None else M,
        "N": n_tarefas,
        "H_size": H_size, "P_size": P_size,
        "S_max": s_max_val,
        # --- Parâmetros de calendário/estrutura do modelo ---
        "j_lab": j_lab,        # jornada diária em horas (Eq. 33/35): converte dias→horas
        "delta": delta,        # duração da sprint em DIAS (Eq. 33/35)
        "Setup": setup,        # horas de setup por (dev,tarefa) (Eq. 29) — Setup>0 ⇒ T_din>0
        "Cerim_s": cerim,      # horas de cerimônias por sprint (Eq. 45)
        "Buffer": buffer,      # margem de segurança da capacidade (Eq. 45)
        "Cap": cap_val,        # capacidade nominal Cap_is do recurso (Eq. 45) = δ·j_lab
        "S_j": s_j,            # máx. devs simultâneos por tarefa (=4, frações 12-avos) (Eq. 6–15/38)
        "S_max_j": s_max_j_val,  # prazo (sprint-limite) por tarefa (Eq. 36)
        # --- Dados da instância (domínios Eq. 65–67: Hd/Ht∈[0,1]; Pd/Pt∈[0,1] somando 1) ---
        "Hd": _scale_skills(hd), "Pd": _norm_profiles(pd),
        "Ht": _scale_skills(ht), "Pt": _norm_profiles(pt),
        "T_base": t_base,
        "Pred": pred,
        "Gap_prec": gap_prec,  # intervalo de segurança entre precedências (Eq. 40)
    }
    with open(f"data/instances/{nome}.json", "w") as f:
        json.dump(dados, f, indent=4)

def generate_instances():
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
        
    write_instance("Cenario_1_Maratona_Multitarefa", N_1, Hd_base, Pd_base, Ht_1, Pt_1, T_base_1, Pred_1)

    # ---------------------------------------------------------------------
    # CENÁRIO 2: Os Monólitos (SIMPLIFICADO PARA RODAR RÁPIDO)
    # ---------------------------------------------------------------------
    N_2, M_2, S_max_2 = 12, 4, 5  # <-- Redução drástica para resolver a complexidade
    
    # Cortamos o Dev 4 (Coringa), ficam só os 4 Especialistas
    Hd_2 = Hd_base[:4]
    Pd_2 = Pd_base[:4]
    
    Ht_2, Pt_2, T_base_2, Pred_2 = [], [], [], [[] for _ in range(N_2)]
    
    for j in range(N_2):
        if j % 4 == 0:
            # Tarefas 0, 4 e 8 são os MONÓLITOS
            # Exigem habilidade do Dev 0 e do Dev 1 simultaneamente
            Ht_2.append([4, 4, 1, 1])
            Pt_2.append([4, 4, 1, 1])
            T_base_2.append(100) # 100h é maior que as ~93h úteis da Sprint, forçando o Trabalho em Equipe!
        else:
            # Tarefas normais para girar o backlog
            idx = j % 4
            Ht_2.append(Hd_2[idx])
            Pt_2.append(Pd_2[idx])
            T_base_2.append(25)
            
    write_instance("Cenario_2_Monolitos_Equipe", N_2, Hd_2, Pd_2, Ht_2, Pt_2, T_base_2, Pred_2, m=M_2, s_max=S_max_2)
    
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
            
    write_instance("Cenario_3_Cadeia_Gargalo", N_3, Hd_base, Pd_base, Ht_3, Pt_3, T_base_3, Pred_3)

if __name__ == "__main__":
    generate_instances()
