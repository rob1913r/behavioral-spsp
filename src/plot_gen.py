# -*- coding: utf-8 -*-
"""
Módulo de Geração de Gráficos e Visualizações Analíticas.

Este script é responsável por renderizar os gráficos utilizados no artigo (G01 a G15).
Ele se divide em duas etapas:
1. Renderização de modelos matemáticos teóricos (funções polinomiais).
2. Renderização de dados empíricos consumidos diretamente do arquivo CSV resultante da otimização.
As imagens são exportadas em alta resolução (300 DPI) para o diretório de documentação (LaTeX).
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

# Suprime avisos do Pandas/Matplotlib para uma execução limpa no console
warnings.filterwarnings('ignore')

# Caminhos padronizados do repositório
INPUT_CSV = os.path.join("data", "results", "metrics.csv")
OUTPUT_DIR = os.path.join("docs", "paper", "figures")

# Garante a existência do diretório de imagens do artigo
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuração global de estética dos gráficos com font_scale ampliado (1.5) para leitura perfeita em duas colunas
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.5)

print(f"\nIniciando pipeline de geração de gráficos (Destino: '{OUTPUT_DIR}')...")

# ==============================================================================
# PARTE 1: GRÁFICOS TEÓRICOS (Modelagem Matemática Estática)
# ==============================================================================
print("-> Gerando visualizações teóricas (G01, G02, G15)...")

x = np.linspace(0, 1, 500)
k_fixed = 5.0

# --- G01: Comparação das Ordens Polinomiais ---
plt.figure(figsize=(10, 6), dpi=300)
plt.axvspan(0.0, 0.3, color='limegreen', alpha=0.15, label='Zona de Aceitação')
plt.axvspan(0.3, 0.7, color='orange', alpha=0.15, label='Zona Crítica')
plt.axvspan(0.7, 1.0, color='crimson', alpha=0.15, label='Zona Extrema / Irreal')
plt.plot(x, 1.0 + (k_fixed * x), label='Erro Linear (Ordem 1)', color='gray', linestyle=':', linewidth=2)
plt.plot(x, 1.0 + (k_fixed * (x**2)), label='Erro Quadrático (Ordem 2)', color='steelblue', linestyle='--', linewidth=2)
plt.plot(x, 1.0 + (k_fixed * (x**3)), label='Erro Cúbico Proposto (Ordem 3)', color='indigo', linewidth=3.5)
plt.plot(x, 1.0 + (k_fixed * (x**4)), label='Erro Quártico (Ordem 4)', color='teal', linestyle='-.', linewidth=2.5)
plt.axhline(1.0, color='black', linewidth=1, linestyle='dashed')
plt.title("Modelagem do Impacto Comportamental por Zonas de Risco", fontweight='bold', fontsize=20)
plt.xlabel("Desalinhamento Comportamental (0.0 a 1.0)", fontsize=18)
plt.ylabel("Multiplicador do Tempo de Execução", fontsize=18)
plt.xlim(0, 1)
plt.ylim(0.8, 6.2)
plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "G01_Justificativa_Matematica.png"))

# --- G02: Calibração detalhada do fator k ---
plt.figure(figsize=(10, 6), dpi=300)
plt.axvspan(0.0, 0.3, color='limegreen', alpha=0.15, label='Zona de Aceitação')
plt.axvspan(0.3, 0.7, color='orange', alpha=0.15, label='Zona Crítica')
plt.axvspan(0.7, 1.0, color='crimson', alpha=0.15, label='Zona Extrema / Irreal')

plt.plot(x, 1.0 + (2.0 * (x**3)), label='k=2 (Subestimado)', color='lightgray', linestyle='-.', linewidth=2)
plt.plot(x, 1.0 + (4.0 * (x**3)), label='k=4', color='darkgray', linestyle='--', linewidth=2)
plt.plot(x, 1.0 + (5.0 * (x**3)), label='k=5 (Proposto)', color='indigo', linewidth=3.5)
plt.plot(x, 1.0 + (6.0 * (x**3)), label='k=6', color='indianred', linestyle='--', linewidth=2)
plt.plot(x, 1.0 + (10.0 * (x**3)), label='k=10 (Superestimado)', color='darkred', linestyle=':', linewidth=2.5)

plt.axhline(1.0, color='black', linewidth=1, linestyle='dashed')
plt.axhline(1.5, color='darkgreen', linewidth=1, linestyle='dashed')
plt.text(0.02, 1.6, "Limite Aceitável (1.5x)", color='darkgreen', fontsize=14, fontweight='bold')
plt.axhline(3.0, color='darkorange', linewidth=1, linestyle='dashed')
plt.text(0.02, 3.1, "Limite Crítico (3x)", color='darkorange', fontsize=14, fontweight='bold')
plt.axhline(6.0, color='darkred', linewidth=1, linestyle='dashed')
plt.text(0.02, 6.1, "Limite Inviável (6x)", color='darkred', fontsize=14, fontweight='bold')

plt.title("Calibração Empírica do Fator de Penalidade (k)", fontweight='bold', fontsize=20)
plt.xlabel("Desalinhamento Comportamental (0.0 a 1.0)", fontsize=18)
plt.ylabel("Multiplicador do Tempo de Execução", fontsize=18)
plt.xlim(0, 1)
plt.ylim(0.8, 11.5)
plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "G02_Calibracao_K.png"))

# --- G15: Comparação do Ganho Técnico contra a Perda Comportamental ---
plt.figure(figsize=(10, 6), dpi=300)
plt.axvspan(0.0, 0.3, color='limegreen', alpha=0.15)
plt.axvspan(0.3, 0.7, color='orange', alpha=0.15)
plt.axvspan(0.7, 1.0, color='crimson', alpha=0.15)

y_junior = 1.0 + (5.0 * (x**3)) - (0.5 * 0.0) 
y_pleno = 1.0 + (5.0 * (x**3)) - (0.5 * 0.5)  
y_senior = 1.0 + (5.0 * (x**3)) - (0.5 * 1.0) 

plt.plot(x, y_junior, label='Júnior (ET = 0.0)', color='darkorange', linestyle='--', linewidth=2.5)
plt.plot(x, y_pleno, label='Pleno (ET = 0.5)', color='steelblue', linestyle='-.', linewidth=2.5)
plt.plot(x, y_senior, label='Sênior (ET = 1.0)', color='indigo', linewidth=3.5)

plt.axhline(1.0, color='black', linewidth=1, linestyle='dashed')
plt.text(0.02, 1.05, "Tempo Nominal (100%)", color='black', fontsize=16, fontweight='bold')

cross_idx = np.where(y_senior >= 1.0)[0][0]
cross_x = x[cross_idx]
plt.plot(cross_x, 1.0, marker='o', markersize=8, color="red")
plt.text(cross_x + 0.02, 0.82, f"Vantagem Técnica\nAnulada (x={cross_x:.2f})", color='red', fontweight='bold', fontsize=15)

plt.title("O Paradoxo do Especialista", fontweight='bold', fontsize=20)
plt.xlabel("Desalinhamento Comportamental (0.0 a 1.0)", fontsize=18)
plt.ylabel("Multiplicador do Tempo Final", fontsize=18)
plt.xlim(0, 1)
plt.ylim(0.4, 6.2)
plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "G15_Paradoxo_Especialista.png"))

# ==============================================================================
# PARTE 2: GRÁFICOS EMPÍRICOS (Consumo do Otimizador)
# ==============================================================================
if not os.path.exists(INPUT_CSV):
    print(f"\n[AVISO] Arquivo '{INPUT_CSV}' não encontrado.")
    print("Os gráficos empíricos (G03 a G14) foram ignorados. Rode 'optimizer.py' primeiro.")
else:
    print("-> Lendo CSV e gerando visualizações empíricas (G03 a G14)...\n")
    df = pd.read_csv(INPUT_CSV)
    
    # --- EXP 1: SCALABILITY ---
    df_exp1 = df[(df['group'] == 'Exp1_Scalability') & (df['feasible'] == 1)]
    if not df_exp1.empty:
        plt.figure(figsize=(8, 5), dpi=300)
        sns.boxplot(data=df_exp1, x="N", y="runtime", palette="Blues_d")
        plt.title("Esforço Computacional por Número de Tarefas", fontweight="bold")
        plt.xlabel("Número de Tarefas (N)")
        plt.ylabel("Tempo de Otimização (s)")
        plt.yscale("log") 
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "G03_Scalability_Boxplot_Tasks.png"))

        plt.figure(figsize=(8, 5), dpi=300)
        sns.lineplot(data=df_exp1, x="N", y="runtime", marker="o", color="navy", linewidth=2)
        plt.title("Crescimento do Tempo vs. Tarefas (N)", fontweight="bold")
        plt.xlabel("Número de Tarefas (N)")
        plt.ylabel("Tempo Médio (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "G04_Scalability_Line_Tasks.png"))

        df_exp1['Vars_Bin'] = pd.qcut(df_exp1['NumVars'], q=5, precision=0, duplicates='drop')
        plt.figure(figsize=(8, 5), dpi=300)
        sns.boxplot(data=df_exp1, x="Vars_Bin", y="runtime", palette="Purples_d")
        plt.title("Distribuição do Tempo por Quantidade de Variáveis", fontweight="bold")
        plt.xlabel("Faixas de Variáveis de Decisão")
        plt.ylabel("Tempo de Otimização (s)")
        plt.xticks(rotation=15)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "G05_Scalability_Boxplot_Vars.png"))

        plt.figure(figsize=(8, 5), dpi=300)
        sns.lineplot(data=df_exp1, x="NumVars", y="runtime", marker="o", color="indigo", linewidth=2)
        plt.title("Crescimento do Tempo vs. Variáveis de Decisão", fontweight="bold")
        plt.xlabel("Número de Variáveis Internas no Solver")
        plt.ylabel("Tempo Médio (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "G06_Scalability_Line_Vars.png"))

    # --- EXP 2: BEHAVIORAL NOISE ---
    df_exp2 = df[(df['group'] == 'Exp2_Behavioral') & (df['feasible'] == 1)]
    if not df_exp2.empty:
        plt.figure(figsize=(8, 5), dpi=300)
        sns.lineplot(data=df_exp2, x="Cenario", y="mksp", marker="o", color="darkorange", linewidth=2.5)
        plt.title("Evolução do Cronograma sob Ruído", fontweight="bold")
        plt.xlabel("Cenário Comportamental")
        plt.ylabel("Makespan Médio (Dias)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "G07_Behavioral_Line_Makespan.png"))

        plt.figure(figsize=(8, 5), dpi=300)
        sns.boxplot(data=df_exp2, x="Cenario", y="mksp", palette="Oranges")
        plt.title("Variância do Makespan nos Cenários Críticos", fontweight="bold")
        plt.xlabel("Cenário Comportamental")
        plt.ylabel("Makespan (Dias)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "G08_Behavioral_Boxplot_Makespan.png"))

        plt.figure(figsize=(8, 5), dpi=300)
        sns.barplot(data=df_exp2, x="Cenario", y="avg_EC", palette="autumn", ci="sd")
        plt.title("Crescimento do Erro Comportamental (EC)", fontweight="bold")
        plt.xlabel("Cenário Comportamental")
        plt.ylabel("Erro Comportamental (EC) Médio")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "G09_Behavioral_Bar_EC.png"))

    # --- EXP 3: BROOKS' LAW ---
    df_exp3 = df[(df['group'] == 'Exp3_Heatmap') & (df['feasible'] == 1)]
    if not df_exp3.empty:
        pivot_df = df_exp3.pivot_table(values="mksp", index="M", columns="N", aggfunc="mean").round(0)
        plt.figure(figsize=(10, 6), dpi=300)
        sns.heatmap(pivot_df, annot=True, fmt="g", cmap="YlOrRd", linewidths=.5, 
                    cbar_kws={'label': 'Makespan (Dias)', 'orientation': 'horizontal', 'location': 'bottom'})
        plt.title("Matriz de Planejamento Tático de Capacidade", fontweight="bold")
        plt.xlabel("Número de Tarefas (N)")
        plt.ylabel("Desenvolvedores Disponíveis (M)")
        plt.savefig(os.path.join(OUTPUT_DIR, "G10_Brooks_Heatmap.png"), bbox_inches='tight')

        df_exp3_n50 = df_exp3[df_exp3['N'] == 50]
        if not df_exp3_n50.empty:
            plt.figure(figsize=(8, 5), dpi=300)
            sns.lineplot(data=df_exp3_n50, x="M", y="mksp", marker="s", color="firebrick", linewidth=2.5)
            plt.title("A Assíntota da Lei de Brooks (N=50)", fontweight="bold")
            plt.xlabel("Número de Desenvolvedores (M)")
            plt.ylabel("Makespan (Dias)")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G11_Brooks_Asymptote.png"))

    # --- EXP 4: PRECEDENCE DENSITY ---
    df_exp4 = df[df['group'] == 'Exp4_Precedence']
    if not df_exp4.empty:
        df_viab = df_exp4.groupby("Density")["feasible"].mean().reset_index()
        df_viab["feasible"] *= 100
        
        plt.figure(figsize=(8, 5), dpi=300)
        sns.lineplot(data=df_exp4[df_exp4['feasible']==1], x="Density", y="mksp", marker="o", color="purple", linewidth=2)
        plt.title("Inflação do prazo devido a dependências estritas", fontweight="bold")
        plt.xlabel("Densidade de Precedência")
        plt.ylabel("Makespan (Dias)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "G12_Precedence_Makespan.png"))

        plt.figure(figsize=(8, 5), dpi=300)
        sns.barplot(data=df_viab, x="Density", y="feasible", color="mediumpurple")
        plt.title("Queda drástica na Viabilidade do Projeto", fontweight="bold")
        plt.xlabel("Densidade de Precedência")
        plt.ylabel("Soluções Viáveis (%)")
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "G13_Precedence_Feasibility.png"))

    # --- EXP 5: TECHNICAL SURPLUS ---
    df_exp5 = df[(df['group'] == 'Exp5_Surplus') & (df['feasible'] == 1)]
    if not df_exp5.empty:
        plt.figure(figsize=(8, 5), dpi=300)
        sns.lineplot(data=df_exp5, x="TechLvl", y="mksp", marker="D", color="teal", linewidth=2)
        plt.title("Redução de Makespan via Excedente Técnico", fontweight="bold")
        plt.xlabel("Nível de Proficiência Técnica")
        plt.ylabel("Makespan (Dias)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "G14_Surplus_Makespan.png"))
