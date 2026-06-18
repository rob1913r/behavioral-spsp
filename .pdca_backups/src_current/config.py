# -*- coding: utf-8 -*-
"""
============================================================================
 config.py — parâmetros da matheurística híbrida VNS + Gurobi (v5 / vns_v2)
============================================================================

A v5 HERDA integralmente os parâmetros da v1 (`optimizer/vns/config.py`):

  GRUPO A — modelo matemático (idênticos ao optimizer.py — paridade obrigatória)
  GRUPO B — busca do GVNS (convergência × diversidade)
  GRUPO C — debug e modos das 43 restrições

e ACRESCENTA:

  GRUPO D — MATHEURÍSTICA: como a metaheurística recorta o problema em
            subproblemas pequenos e o Gurobi os reotimiza até a otimalidade.

A herança via `import *` GARANTE que os GRUPOS A/B/C ficam byte-a-byte iguais
aos da v1 validada — qualquer ajuste de busca da v5 é feito SOBRESCREVENDO o
parâmetro aqui embaixo, sem tocar na v1.
"""

# --- Herda TODOS os parâmetros da v1 (GRUPOS A, B, C) ---
from optimizer.vns.config import *          # noqa: F401,F403
from optimizer.vns.config import (           # re-exporta explicitamente p/ o solver
    VALID_COMBOS, VALID_COMBOS_SET, F_FRACS, CAP_UTIL, H_SPAN, SETUP_HOURS,
    MAX_TASKS_PER_DEV_SPRINT, MAX_DEVS_PER_TASK, constraint_mode,
)

# ============================================================================
# OVERRIDES DE BUSCA DA v5 (só o que difere da v1)
# ============================================================================

# Orçamento de tempo: 0.5 acelera os ciclos de calibração (decisão do usuário).
# Se a matheurística for mais eficiente que a v1, metade do tempo basta para
# atingir o mesmo nível de solução.
T_MULT = 0.5

# Piso de tempo por instância (antes de T_MULT). A v1 (C-063) usou 180 porque
# instâncias PEQUENAS têm alta variância de trajetória e precisam de mais
# reinícios. Com T_MULT=0.5 o piso efetivo caía para 90s — METADE do orçamento
# que a v1 deu ao C2 (que precisa de ~180s para achar 425.2h via ILS). Subimos
# o piso para 360 → piso efetivo 180s nas instâncias pequenas (C2/C3); o C1
# (210s pela fórmula) não é afetado. Mantém a comparação justa com a v1.
T_CLAMP_MIN = 360.0

# Seed fixa para os ciclos de calibração (a v1 usou 42). Torna os resultados
# reprodutíveis e comparáveis ciclo a ciclo (ver E-035). O RNG da matheurística
# é ISOLADO num stream próprio (ver solver), então o fluxo de sorteios do GVNS
# fica idêntico ao da v1 com a mesma seed.
RANDOM_SEED = None

# ============================================================================
# GRUPO D — MATHEURÍSTICA (Gurobi reotimiza subproblemas guiados pelo VNS)
# ============================================================================

# Liga/desliga toda a camada matheurística. Com MH_ENABLED=False, o vns_v2 se
# comporta como o GVNS puro da v1 (útil para comparação A/B).
MH_ENABLED = True

# --- Tamanho do subproblema (alavanca principal de tempo) ---
# Recorte PRECEDENCE-AWARE: instâncias com precedências (C3) recebem recortes
# MAIORES/mais LARGOS/mais LONGOS — o gargalo do C3 é compactar a cadeia (8→7
# sprints), o que exige restruturar um pedaço maior de uma vez (E-037). Sem
# precedências o recorte menor basta (e o C2 nem usa sub-MIP).
MH_FREE_SET_SIZE = 10    # nº de tasks "livres" por sub-MIP (sem precedências)
MH_FREE_SET_SIZE_PRED = 12   # idem, COM precedências (C3) — recortes grandes
                             # demais (16) couberam em menos sub-MIPs (E-037)
MH_FREE_SET_MIN  = 4     # limite inferior da adaptação de |F|
MH_FREE_SET_MAX  = 16    # limite superior da adaptação de |F|

# --- Corredor (Corridor Method): restringe o domínio das tasks livres ---
MH_SPRINT_RADIUS  = 3    # cada task livre só pode ir a s in [s_inc-Δ, s_inc+Δ]
MH_SPRINT_RADIUS_PRED = 4    # corredor um pouco mais largo p/ C3
MH_DEV_CANDIDATES = 4    # nº de devs candidatos por task livre (+ os incumbentes)

# --- Orçamento do solver exato em cada subproblema ---
MH_SUBMIP_TIME = 12.0    # tempo limite (s) por chamada do Gurobi no sub-MIP
MH_SUBMIP_TIME_PRED = 14.0   # tempo por sub-MIP no C3
MH_SUBMIP_GAP  = 0.0     # MIPGap alvo no sub-MIP (0 = otimalidade no pedaço)
MH_SUBMIP_FRAC = 0.5     # fração do tempo GLOBAL restante que um sub-MIP pode usar

# --- Orçamento de tempo PRECEDENCE-AWARE (E-038/E-039) ---
# O boost de tempo (testado 1.8× no ciclo 8) NÃO ajudou o C3 (seed 42 cai no modo
# RUIM independentemente do tempo). O tempo extra é melhor gasto num POOL DE
# ELITE (várias seeds, fica a melhor) — ver MH_POOL_SEEDS_PRED. Boost desligado.
MH_PRED_TIME_BOOST = 1.0

# --- Pool de elite p/ instâncias com precedências (E-039) ---
# O C3 é bimodal-por-seed (E-030/E-039): cada seed cai num modo (bom ~765 ou ruim
# ~840). Rodamos algumas seeds e ficamos com a MELHOR solução factível (best-of-N
# — prática padrão p/ metaheurísticas estocásticas; "pool de elite" sugerido pela
# própria v1). Cada seed roda o orçamento normal da instância (180s no C3).
MH_BASE_SEED = 42                  # seed das instâncias SEM precedências (C1/C2)
MH_POOL_SEEDS_PRED = [99, 13, 7, 42, 1]   # Ciclo 8 (C-097): pool maior (best-of-N)
                                   # p/ o C3 bimodal+ruidoso — mais sorteios,
                                   # monotônico (nunca pior), captura bacia melhor

# --- Back-off adaptativo: para de injetar sub-MIP no loop quando ele falha
#     repetidamente (devolve o orçamento ao GVNS — ver E-032). A varredura
#     final (Modo B) continua rodando uma vez no fim. ---
MH_MAX_CONSEC_FAILS = 2

# --- Determinismo do Gurobi nos subproblemas (reprodutibilidade) ---
MH_GUROBI_THREADS = 1
MH_GUROBI_SEED    = 0
# MIPFocus=1 nos sub-MIPs: prioriza achar/melhorar boas incumbentes rápido
# dentro do limite de tempo do pedaço. Verificado (E-040): no C3, MIPFocus=1 dá
# ~50h MELHOR que o balanceado (0) — os recortes não são triviais e provar bound
# é caro (LB frouxo). 1=feasibility-focus.
MH_GUROBI_MIPFOCUS = 1

# --- Estratégias de decomposição habilitadas (Seção 5.2 do plano v5) ---
#   'makespan_tail'  — tasks que terminam mais tarde (definem o Tmax) + doadoras
#   'bottleneck_dev' — tasks do dev mais carregado (decomposição por recurso)
#   'critical_chain' — segmento da cadeia crítica de precedência (alvo do C3)
#   'monolith_axis'  — monolito + tasks do mesmo eixo (alvo do C2)
#   'window'         — todas as tasks de uma janela de sprints (decomp. por tempo)
#   'related'        — semente + tasks mais relacionadas (POPMUSIC)
# Ordem importa na varredura (Modo B, executada em sequência): as estratégias
# de CADEIA (critical_chain, bottleneck_dev) são as que mais rendem no C3, então
# vêm cedo; makespan_tail abre (bom salto inicial em qualquer cenário).
MH_STRATEGIES = ["makespan_tail", "critical_chain", "bottleneck_dev",
                 "monolith_axis", "window", "related", "random_scatter",
                 "precedence_chain_repair", "dev_rolling",
                 "rolling_horizon"]   # C-107/C-108/C-109: joelhos da cadeia +
                 # 2º dev mais carregado + janela de sprints deslizante
                 # (Palpant 2004) — aditivas, por último.
                 # Ciclo 12 (C-102): random_scatter = destroy LNS aleatório
                 # disperso (Shaw 1998) — diversificador; entra por último p/ não
                 # deslocar os heavy-lifters (makespan_tail/critical_chain).
MH_STRATEGY_ADAPTIVE = True   # sorteio adaptativo (ALNS) por desempenho (loop)
# Ciclo 20 (C-110): ALNS de verdade — fator de reação (decay) + reward
# diferenciado na atualização dos pesos do loop (Ropke & Pisinger 2006).
MH_ALNS_DECAY = 0.85          # lambda: peso_novo = lambda*peso + (1-lambda)*reward
MH_ALNS_REWARD_BEST = 8.0     # reward de um sub-MIP de loop que MELHORA
MH_ALNS_REWARD_FAIL = 1.0     # reward de um sub-MIP de loop que FALHA

# --- Cadência no loop GVNS (Modo A: sub-MIP como vizinhança) ---
MH_ON_STAGNATION = True   # injeta um sub-MIP quando o GVNS estagna
MH_STAGNATION_AFTER = 10  # nº de iterações sem melhora p/ disparar um sub-MIP
MH_VND_AFTER_MIP = False  # rodar VND de polimento após o sub-MIP (custo extra)
# Injeção de sub-MIP no loop SÓ em instâncias COM precedências (C3): nessas o
# sub-MIP é o caminho. Em instâncias SEM precedências (C1/C2) o GVNS é o caminho
# (o C2 chega a 425.2h só pela trajetória, ~123s) e qualquer sub-MIP de loop só
# rouba tempo do GVNS — ver E-036. A varredura final (Modo B) ainda roda nos dois.
MH_LOOP_ONLY_PRECEDENCE = True

# --- Varredura fix-and-optimize / POPMUSIC (Modo B: intensificação) ---
MH_SWEEP_ON_ILS    = True   # varredura no gatilho de ILS
MH_FINAL_SWEEP     = True   # varredura final no melhor incumbente antes de gravar
# Reserva PRECEDENCE-AWARE (E-036): instâncias com precedências (C3) reservam
# bastante p/ a varredura; sem precedências (C1/C2) reservam o mínimo (o GVNS
# precisa do orçamento — o C2 só crava 425.2h em ~123s de GVNS).
MH_FINAL_RESERVE        = 0.35   # com precedências (C3): equilíbrio GVNS×varredura
MH_FINAL_RESERVE_NOPRED = 0.10   # sem precedências (C1, C2)
MH_SWEEP_MAX_PASSES = 5     # nº máx. de passadas da varredura (POPMUSIC para se 1 passada não melhora)

# --- Saída ---
RESULTS_SUBDIR = "vns_v2"   # data/results/vns_v2/  (NÃO toca gurobi/ nem vns/)
