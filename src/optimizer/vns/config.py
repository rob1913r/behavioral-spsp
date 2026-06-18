# -*- coding: utf-8 -*-
"""
============================================================================
 config.py — TODOS os parâmetros do VNS em um único lugar
============================================================================

Este módulo concentra TODAS as constantes e "botões de ajuste" do algoritmo.
Está dividido em 3 grupos:

  GRUPO A — Parâmetros do MODELO matemático (espelham optimizer.py).
            NÃO ALTERAR: precisam ser idênticos ao Gurobi para que a
            comparação VNS vs Gurobi seja válida.

  GRUPO B — Parâmetros de BUSCA do VNS (balanço convergência x diversidade).
            Estes são os botões que calibram o comportamento da
            metaheurística. Podem ser ajustados livremente.

  GRUPO C — Parâmetros de DEBUG e VERIFICAÇÃO de restrições.
            Controlam o nível de log e o mode de cada restrição.
"""

# ============================================================================
# GRUPO A — PARÂMETROS DO MODELO (idênticos ao optimizer.py — NÃO ALTERAR)
# ============================================================================

# --- Curva de aprendizado (Eq. 21–23 do artigo) ---
ALPHA_AXIS = 0.05   # weight do bônus por task do mesmo axis temático
LAMBDA_LEARN  = 0.80   # decaimento temporal do bônus de axis (curva de esquecimento)
ALPHA_DEP  = 0.20   # weight do bônus por predecessora executada pelo mesmo dev
LAMBDA_DEP  = 0.60   # decaimento por distância na chain de precedência

# --- Fadiga por load cognitiva (Eq. 19–20) ---
FATIGUE_THRESHOLD   = 0.10      # threshold de load cognitiva (Cg) a partir do qual há fadiga
FATIGUE_WEIGHT = 9.0       # multiplicador da penalidade de fadiga

# --- Penalidade de contexto (Eq. 17) ---
CONTEXT_WEIGHT = 0.20     # 0.20 * max(0, n_tasks_count - 1)

# --- Overhead de comunicação / Lei de Brooks (Eq. 18) ---
COMM_WEIGHT = 0.05     # 0.05 * d(d-1)/2

# --- Team Match (Eq. 12–15) ---
TEAM_MATCH_TARGET  = 0.25   # target do perfil médio da team (Eq. 13–14)
TEAM_MATCH_WEIGHT  = 0.5    # factor externo da Eq. 15
TEAM_MATCH_NORM  = 1.5    # normalização da Eq. 15
COG_LOAD_DIVISOR    = 30.0   # divisor do índice de load cognitiva (Eq. 19)

# --- Capacidade e calendário (Eq. 46) ---
CEREMONY_HOURS   = 8          # horas de cerimônias ágeis por sprint
BUFFER    = 0.1        # margem de segurança (10%)
SETUP_HOURS = 2.0        # horas de setup por (dev, task)
H_SPAN    = 14 * 8     # duração da sprint em horas (112h)
CAP_UTIL  = (H_SPAN - CEREMONY_HOURS) * (1 - BUFFER)   # 93.6h por (dev, sprint)

# --- Limites estruturais ---
MAX_TASKS_PER_DEV_SPRINT = 5   # Eq. 16: máx. tasks simultâneas por dev/sprint
MAX_DEVS_PER_TASK        = 4   # Eq. 38: máx. devs por task

# --- Frações discretas de dedicação (em doze avos) ---
F_FRACS = [3, 4, 6, 8, 9, 12]

# Tabela das 7 combinações válidas de frações (somam 12 doze avos).
# Invariante obrigatória: sum(frac_matrix[j]) == 1.0 para toda task j.
VALID_COMBOS = [
    (12,),          # 1 dev:  1.000
    (3, 9),         # 2 devs: 0.250 + 0.750
    (4, 8),         # 2 devs: 0.333 + 0.667
    (6, 6),         # 2 devs: 0.500 + 0.500
    (3, 3, 6),      # 3 devs: 0.250 + 0.250 + 0.500
    (4, 4, 4),      # 3 devs: 0.333 + 0.333 + 0.333
    (3, 3, 3, 3),   # 4 devs: 0.250 + 0.250 + 0.250 + 0.250
]
VALID_COMBOS_SET = {tuple(sorted(c)) for c in VALID_COMBOS}

# --- Constantes de linearização do MILP (Eq. 25–28 e 43) ---
GAMMA_M_MAX = 100.0
GAMMA_M_MIN = -100.0
BIG_M       = 100000

# --- Fator técnico e comportamental (Eq. 1–3, semântica do optimizer.py) ---
F_TECH_WEIGHT  = 0.5     # weight do excedente técnico na Eq. 1
F_TECH_NORM  = 25.0    # normalização (escala de skill 0-5 ao quadrado)
F_TECH_MIN   = 0.1     # piso do factor técnico (max(0.1, ...) no optimizer.py)
F_BEH_WEIGHT   = 5.0     # weight do atrito comportamental na Eq. 3
PSI_NORM_PER_QUAD = 4.0  # ATENÇÃO: optimizer.py usa psi = total/(P_size*4.0).
                         # O artigo (Eq. 2) diz psi = total/2 — divergência
                         # documentada em planning/proximos_passos.md.

# ============================================================================
# GRUPO B — PARÂMETROS DE BUSCA DO VNS (botões de calibração)
# ============================================================================

# --- Orçamento de tempo ---
T_MULT       = 1.0     # multiplicador global do tempo limite de cada instância
T_BASE_REF   = 420.0   # segundos para a instância de referência (N*M*S = 5625)
NVAR_REF     = 5625    # tamanho de referência (Cenário 1: 75*5*15)
T_CLAMP_MIN  = 180.0   # tempo mínimo por instância (antes de T_MULT) —
                       # instâncias pequenas (C2: 87s pela fórmula) têm
                       # alta variância de trajetória; o piso maior dá
                       # mais reinícios/bilhetes por execução
T_CLAMP_MAX  = 900.0   # tempo máximo por instância (antes de T_MULT)

# --- Estrutura da busca ---
K_MAX             = 10     # número de vizinhanças de shaking (N1–N10)
ILS_RESTART_AFTER = 32     # iterações sem melhora até perturbação grande (ILS)
ILS_PERTURB_FRAC  = 1/3    # fração de tasks perturbadas no restart ILS
VND_TIME_FRAC     = 0.40   # fração do tempo restante que cada VND pode usar
VND_MIN_TIME      = 5.0    # tempo mínimo (s) garantido para cada chamada de VND
VND_MAX_TIME      = 8.0    # teto ABSOLUTO (s) por chamada de VND. Curto de
                           # propósito: o VND roda em DOIS ESTÁGIOS — se após
                           # o 1º estágio o candidato está a menos de
                           # VND_EXT_FACTOR do incumbent, ganha mais um
                           # estágio; senão é descartado barato.
VND_EXT_FACTOR     = 1.25   # candidato ganha extensão de VND se
                           # fp < fp_best * este factor
VND_EXT_PROGRESS = 0.5    # ... OU se o último estágio reduziu o fp para
                           # menos de 50% (reconstrução de wreck em pleno
                           # andamento — ex.: C2 sai de 14.5k p/ 2k em 8s)
VND_EXT_MAX_STAGES = 3   # nº máximo de extensões consecutivas (8s cada):
                           # reconstruções longas (C2: ~25s) cabem em
                           # 8+3x8=32s; wrecks sem progresso param cedo
REPAIR_PRE_VND_THRESHOLD = 50.0  # wreck LEVE (fp < este factor x fp_best) vai
                              # CRU para o VND — o gradiente de penalidade é
                              # que guia a reconstrução criativa (foi assim
                              # que o C2 achou o ótimo); o repair iterativo
                              # pré-VND só roda em wrecks médios, pois ele
                              # conserta "do jeito chato" (espalhando) e
                              # apaga a estrutura do shaking
SHAKE_DISCARD_FACTOR = 100.0  # candidato pós-repair com fp acima de
                              # fp_best*este factor é DESCARTADO sem VND.
                              # Calibração (ciclo 4): estados intermediários
                              # RECUPERÁVEIS do C2 (monolith estourando) têm
                              # fp ~26x o incumbent — 20x os descartava e o
                              # C2 perdeu o ótimo; os wrecks irreparáveis do
                              # C3 têm fp >=150x. 100x separa os dois casos.
K_RANDOM_ORDER = True   # True: as vizinhanças k=1..K_MAX são percorridas
                           # em order ALEATÓRIA (re-embaralhada a cada rodada)
                           # — os operadores-template N9/N10 são sorteados
                           # cedo em vez de esperar K_MAX-1 falhas; False:
                           # progressão clássica sequencial do GVNS

# --- Pesos da função penalizada f_pen = Tmax + λ_cap·pen_cap + ... ---
# As penalidades já carregam escala própria (pen_cap é multiplicada por
# CAP_UTIL, pen_window por H_SPAN, pen_tasks por CAP_UTIL), então λ = 1.0
# significa "1 hora de violação custa ~93-112x mais que 1 hora de makespan".
LAM_CAP    = 1.0
LAM_WINDOW = 1.0
LAM_TASKS  = 1.0

# --- Pressão de compressão (anti-platô) ---
# Tmax só cai quando a ÚLTIMA sprint esvazia por inteiro, então mover uma
# task para mais cedo não muda o objetivo — a busca fica sem gradiente.
# Este termo adiciona LAM_COMP × média(Inic_j + Dur_j) ao f_pen: cronogramas
# mais compactos ficam levemente best_ones mesmo com o mesmo Tmax.
# A ACEITAÇÃO do incumbent é lexicográfica (Tmax primeiro, compressão como
# desempate), então o Tmax do best NUNCA piora por causa deste termo.
LAM_COMP = 0.1

# --- Heurística construtiva gulosa ---
F_TOTAL_PESS = 2.0   # multiplicador pessimistic usado na estimativa de load
                     # da gulosa. 2.0 empacota ~3 tasks/dev/sprint, perto
                     # da densidade do Gurobi (F_total real ~1.2-1.7); o
                     # repair + gate seguram excessos. NUNCA reverter a
                     # gulosa para natural-match sem B_apr (changes.md v2.2).

# --- Reparos ---
F_PESS_REPAIR     = 2.0   # multiplicador pessimistic do repair_feasibility_fast
REPAIR_FAST_PASSES = 3    # nº de passadas do repair rápido pós-shaking
REPAIR_CAP_LIMIT_H = 40.0  # só roda o repair exato (caro) de capacity no
                            # gate quando o excesso total <= este valor em
                            # horas; acima disso o candidato está longe demais
                            # de factível e é rejeitado sem gastar orçamento

# --- Operadores de vizinhança ---
LNS_P_K6        = 0.25   # fração de tasks destruídas no shaking k=6 (LNS)
LNS_P_K8        = 0.40   # fração de tasks destruídas no shaking k=8
L3_MIN_SAMPLE  = 15     # mínimo de tasks sampled no VND L3 (frac-shift)
L3_SAMPLE_FRAC = 1/3    # fração de tasks sampled no VND L3
L4_ATTEMPTS   = 12     # attempts de reshuffle por sprint no VND L4
L1_WINDOW       = 3      # VND L1 só tenta sprints a ±L1_WINDOW da atual
                         # (+ 1 sprint aleatória distante) — corta o cost da
                         # prova de ótimo local sem perder os movimentos úteis

# --- N9: aquecimento de axis + monolith tardio (template do Gurobi C2) ---
# Padrão da solução ótima do Gurobi para monoliths: o dev participa (fração
# 3/12) de várias tasks do MESMO axis em sprints anteriores e executa o
# monolith SOLO depois, com B_apr acumulado alto (F_total ~0.8).
N9_TOP_FRAC     = 0.2    # sorteia o "monolith" entre os top 20% maiores T_nom
N9_WARMUP_N     = 4      # nº máximo de tasks axis usadas no aquecimento
N9_WARMUP_FRAC  = 3      # fração (em doze avos) do dev nas tasks de warmup

# --- Reprodutibilidade ---
RANDOM_SEED = None   # None = aleatório a cada execução; inteiro = reprodutível
                   # (manter fixo durante a fase de debug para comparar ciclos)

# --- Critério de qualidade (apenas informativo nos gráficos) ---
VNS_GAP_TARGET = 0.20   # gap target de 20% sobre o lower bound

# ============================================================================
# GRUPO C — DEBUG E VERIFICAÇÃO DE RESTRIÇÕES
# ============================================================================

# Nível de detalhe do log:
#   0 = mínimo (só resumo por instância)
#   1 = normal  (eventos importantes no console + arquivo debug_log.txt)
#   2 = verbose (cada melhoria interna do VND também vai para o arquivo)
DEBUG_LEVEL = 0

# Tolerância numérica para considerar uma restrição violated
TOL_CONSTRAINT = 1e-6

# Modo de cada restrição na verificação integral (por número da equação):
#   'ENFORCE' — violação REJEITA a solução candidata (incumbent nunca viola)
#   'WARN'    — violação é registrada no log mas NÃO rejeita
#   'OFF'     — restrição não é avaliada
# Default: ENFORCE para todas. Exceções listadas abaixo.
CONSTRAINT_MODE_DEFAULT = 'ENFORCE'
CONSTRAINT_MODE = {
    # Eq. 37 (coverage de skills HdT) NÃO existe no optimizer.py.
    # O baseline Gurobi foi gerado sem ela, então o VNS não deve rejeitar
    # soluções por causa dela — apenas avisar. Ver proximos_passos.md.
    37: 'WARN',
}

def constraint_mode(num_eq: int) -> str:
    """Modo de verificação da restrição da equação `num_eq` do artigo."""
    return CONSTRAINT_MODE.get(num_eq, CONSTRAINT_MODE_DEFAULT)

# --- N10: specialist de chain (template do Gurobi C3) ---
# Padrão da solução Gurobi para chains de precedência: o MESMO dev executa
# tasks sucessivas da chain (b_dep=0.2 por predecessora direta) e fica
# specialist (F_total ~0.7-0.9), com ~3 tasks de chain por sprint.
N10_PATH_MAX  = 8     # comprimento máximo do caminho consolidado
N10_PER_SPRINT   = 3     # tasks da chain por sprint na reconstrução

# --- Multi-start da solução inicial ---
# Em instâncias COM precedências, além da gulosa B_apr-aware roda também a
# "gulosa de chains" (cada chain inteira vai para um dev specialist,
# N10_PER_SPRINT tasks por sprint — template do Gurobi C3); cada start
# recebe um VND curto e o best vira o incumbent inicial.
MULTISTART_VND_S = 20.0  # segundos de VND para cada start do multi-start
                         # (o template de monoliths precisa de ~20s para
                         # assentar; com 8s ele competia ainda em obra)
CHAIN_GREEDY_F_PESS = 1.3  # estimativa de F_total do dev specialist na
                            # gulosa de chains (com b_dep, F_total real ~0.9-1.1)
