# -*- coding: utf-8 -*-
"""
Módulo de Geração de Gráficos e Visualizações Analíticas (Bilíngue).

Este script é responsável por renderizar os gráficos utilizados no artigo (G01 a G15).
Ele gera duas baterias de imagens automaticamente:
1. Versão em Português (salva em docs/paper-pt/figures/)
2. Versão em Inglês (salva em docs/paper-en/figures/)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

# Suprime avisos do Pandas/Matplotlib para uma execução limpa no console
warnings.filterwarnings('ignore')

# Caminho de entrada do CSV
INPUT_CSV = os.path.join("data", "results", "metrics.csv")

# Configuração global de estética dos gráficos com font_scale ampliado (1.5)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.5)

# ==============================================================================
# DICIONÁRIO DE INTERNACIONALIZAÇÃO (i18n)
# ==============================================================================
LANG_STR = {
    'pt': {
        'out_dir': os.path.join("docs", "paper-pt", "figures"),
        # G01
        'g01_title': "Modelagem do Impacto Comportamental por Zonas de Risco",
        'g01_x': "Desalinhamento Comportamental (0.0 a 1.0)",
        'g01_y': "Multiplicador do Tempo de Execução",
        'z_acc': "Zona de Aceitação", 'z_crit': "Zona Crítica", 'z_ext': "Zona Extrema / Irreal",
        'l_lin': "Erro Linear (Ordem 1)", 'l_quad': "Erro Quadrático (Ordem 2)",
        'l_cub': "Erro Cúbico Proposto (Ordem 3)", 'l_quar': "Erro Quártico (Ordem 4)",
        # G02
        'g02_title': "Calibração Empírica do Fator de Penalidade (k)",
        'k2': "k=2 (Subestimado)", 'k5': "k=5 (Proposto)", 'k10': "k=10 (Superestimado)",
        't_acc': "Limite Aceitável (1.5x)", 't_crit': "Limite Crítico (3x)", 't_inv': "Limite Inviável (6x)",
        # G15
        'g15_title': "O Paradoxo do Especialista",
        'g15_y': "Multiplicador do Tempo Final",
        'l_jun': "Júnior (ET = 0.0)", 'l_mid': "Pleno (ET = 0.5)", 'l_sen': "Sênior (ET = 1.0)",
        't_nom': "Tempo Nominal (100%)", 't_null': "Vantagem Técnica\nAnulada",
        # G03 & G04
        'g03_title': "Esforço Computacional por Número de Tarefas",
        'g04_title': "Crescimento do Tempo vs. Tarefas (N)",
        'x_task': "Número de Tarefas (N)", 'y_time_s': "Tempo de Otimização (s)", 'y_time_avg': "Tempo Médio (s)",
        # G05 & G06
        'g05_title': "Distribuição do Tempo por Quantidade de Variáveis",
        'g06_title': "Crescimento do Tempo vs. Variáveis de Decisão",
        'x_var_b': "Faixas de Variáveis de Decisão", 'x_var_n': "Número de Variáveis Internas no Solver",
        # G07, G08, G09
        'g07_title': "Evolução do Cronograma sob Ruído",
        'g08_title': "Variância do Makespan nos Cenários Críticos",
        'g09_title': "Crescimento do Erro Comportamental (EC)",
        'x_scen': "Cenário Comportamental", 'y_mksp': "Makespan (Dias)", 
        'y_mksp_avg': "Makespan Médio (Dias)", 'y_ec': "Erro Comportamental (EC) Médio",
        # G10 & G11
        'g10_title': "Matriz de Planejamento Tático de Capacidade",
        'g11_title': "A Assíntota da Lei de Brooks (N=50)",
        'y_devs': "Desenvolvedores Disponíveis (M)", 'x_devs': "Número de Desenvolvedores (M)",
        # G12 & G13
        'g12_title': "Inflação do prazo devido a dependências estritas",
        'g13_title': "Queda drástica na Viabilidade do Projeto",
        'x_dens': "Densidade de Precedência", 'y_feas': "Soluções Viáveis (%)",
        # G14
        'g14_title': "Redução de Makespan via Excedente Técnico",
        'x_tech': "Nível de Proficiência Técnica"
    },
    'en': {
        'out_dir': os.path.join("docs", "paper-en", "figures"),
        # G01
        'g01_title': "Modeling Behavioral Impact by Risk Zones",
        'g01_x': "Behavioral Misalignment (0.0 to 1.0)",
        'g01_y': "Execution Time Multiplier",
        'z_acc': "Acceptance Zone", 'z_crit': "Critical Zone", 'z_ext': "Extreme / Unreal Zone",
        'l_lin': "Linear Error (Order 1)", 'l_quad': "Quadratic Error (Order 2)",
        'l_cub': "Proposed Cubic Error (Order 3)", 'l_quar': "Quartic Error (Order 4)",
        # G02
        'g02_title': "Empirical Calibration of the Penalty Factor (k)",
        'k2': "k=2 (Underestimated)", 'k5': "k=5 (Proposed)", 'k10': "k=10 (Overestimated)",
        't_acc': "Acceptable Limit (1.5x)", 't_crit': "Critical Limit (3x)", 't_inv': "Infeasible Limit (6x)",
        # G15
        'g15_title': "The Specialist Paradox",
        'g15_y': "Final Time Multiplier",
        'l_jun': "Junior (TS = 0.0)", 'l_mid': "Mid-level (TS = 0.5)", 'l_sen': "Senior (TS = 1.0)",
        't_nom': "Nominal Time (100%)", 't_null': "Technical Advantage\nNullified",
        # G03 & G04
        'g03_title': "Computational Effort by Number of Tasks",
        'g04_title': "Time Growth vs. Tasks (N)",
        'x_task': "Number of Tasks (N)", 'y_time_s': "Optimization Time (s)", 'y_time_avg': "Average Time (s)",
        # G05 & G06
        'g05_title': "Time Distribution by Quantity of Variables",
        'g06_title': "Time Growth vs. Decision Variables",
        'x_var_b': "Decision Variable Ranges", 'x_var_n': "Number of Internal Variables in Solver",
        # G07, G08, G09
        'g07_title': "Schedule Evolution under Noise",
        'g08_title': "Makespan Variance in Critical Scenarios",
        'g09_title': "Growth of Behavioral Error (BE)",
        'x_scen': "Behavioral Scenario", 'y_mksp': "Makespan (Days)", 
        'y_mksp_avg': "Average Makespan (Days)", 'y_ec': "Average Behavioral Error (BE)",
        # G10 & G11
        'g10_title': "Tactical Capacity Planning Matrix",
        'g11_title': "Brooks's Law Asymptote (N=50)",
        'y_devs': "Available Developers (M)", 'x_devs': "Number of Developers (M)",
        # G12 & G13
        'g12_title': "Schedule Inflation due to Strict Dependencies",
        'g13_title': "Drastic Drop in Project Feasibility",
        'x_dens': "Precedence Density", 'y_feas': "Feasible Solutions (%)",
        # G14
        'g14_title': "Makespan Reduction via Technical Surplus",
        'x_tech': "Technical Proficiency Level"
    }
}

print(f"\nIniciando pipeline de geração de gráficos Bilíngue...")

# Carrega o CSV apenas uma vez, se existir
df = None
if os.path.exists(INPUT_CSV):
    df = pd.read_csv(INPUT_CSV)
else:
    print(f"[AVISO] Arquivo '{INPUT_CSV}' não encontrado. Apenas gráficos teóricos serão gerados.")

# ==============================================================================
# LOOP PRINCIPAL PARA GERAR OS DOIS IDIOMAS
# ==============================================================================
for lang in ['pt', 'en']:
    S = LANG_STR[lang]
    OUTPUT_DIR = S['out_dir']
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n---> Renderizando gráficos em [{lang.upper()}] (Destino: '{OUTPUT_DIR}')")
    
    x = np.linspace(0, 1, 500)
    k_fixed = 5.0

    # --- G01 ---
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axvspan(0.0, 0.3, color='limegreen', alpha=0.15, label=S['z_acc'])
    plt.axvspan(0.3, 0.7, color='orange', alpha=0.15, label=S['z_crit'])
    plt.axvspan(0.7, 1.0, color='crimson', alpha=0.15, label=S['z_ext'])
    plt.plot(x, 1.0 + (k_fixed * x), label=S['l_lin'], color='gray', linestyle=':', linewidth=2)
    plt.plot(x, 1.0 + (k_fixed * (x**2)), label=S['l_quad'], color='steelblue', linestyle='--', linewidth=2)
    plt.plot(x, 1.0 + (k_fixed * (x**3)), label=S['l_cub'], color='indigo', linewidth=3.5)
    plt.plot(x, 1.0 + (k_fixed * (x**4)), label=S['l_quar'], color='teal', linestyle='-.', linewidth=2.5)
    plt.axhline(1.0, color='black', linewidth=1, linestyle='dashed')
    plt.title(S['g01_title'], fontweight='bold', fontsize=20)
    plt.xlabel(S['g01_x'], fontsize=18)
    plt.ylabel(S['g01_y'], fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0.8, 6.2)
    plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "G01_Justificativa_Matematica.png"))

    # --- G02 ---
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axvspan(0.0, 0.3, color='limegreen', alpha=0.15, label=S['z_acc'])
    plt.axvspan(0.3, 0.7, color='orange', alpha=0.15, label=S['z_crit'])
    plt.axvspan(0.7, 1.0, color='crimson', alpha=0.15, label=S['z_ext'])
    plt.plot(x, 1.0 + (2.0 * (x**3)), label=S['k2'], color='lightgray', linestyle='-.', linewidth=2)
    plt.plot(x, 1.0 + (4.0 * (x**3)), label='k=4', color='darkgray', linestyle='--', linewidth=2)
    plt.plot(x, 1.0 + (5.0 * (x**3)), label=S['k5'], color='indigo', linewidth=3.5)
    plt.plot(x, 1.0 + (6.0 * (x**3)), label='k=6', color='indianred', linestyle='--', linewidth=2)
    plt.plot(x, 1.0 + (10.0 * (x**3)), label=S['k10'], color='darkred', linestyle=':', linewidth=2.5)
    plt.axhline(1.0, color='black', linewidth=1, linestyle='dashed')
    plt.axhline(1.5, color='darkgreen', linewidth=1, linestyle='dashed')
    plt.text(0.02, 1.6, S['t_acc'], color='darkgreen', fontsize=14, fontweight='bold')
    plt.axhline(3.0, color='darkorange', linewidth=1, linestyle='dashed')
    plt.text(0.02, 3.1, S['t_crit'], color='darkorange', fontsize=14, fontweight='bold')
    plt.axhline(6.0, color='darkred', linewidth=1, linestyle='dashed')
    plt.text(0.02, 6.1, S['t_inv'], color='darkred', fontsize=14, fontweight='bold')
    plt.title(S['g02_title'], fontweight='bold', fontsize=20)
    plt.xlabel(S['g01_x'], fontsize=18)
    plt.ylabel(S['g01_y'], fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0.8, 11.5)
    plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "G02_Calibracao_K.png"))

    # --- G15 ---
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axvspan(0.0, 0.3, color='limegreen', alpha=0.15)
    plt.axvspan(0.3, 0.7, color='orange', alpha=0.15)
    plt.axvspan(0.7, 1.0, color='crimson', alpha=0.15)
    y_junior = 1.0 + (5.0 * (x**3)) - (0.5 * 0.0) 
    y_pleno = 1.0 + (5.0 * (x**3)) - (0.5 * 0.5)  
    y_senior = 1.0 + (5.0 * (x**3)) - (0.5 * 1.0) 
    plt.plot(x, y_junior, label=S['l_jun'], color='darkorange', linestyle='--', linewidth=2.5)
    plt.plot(x, y_pleno, label=S['l_mid'], color='steelblue', linestyle='-.', linewidth=2.5)
    plt.plot(x, y_senior, label=S['l_sen'], color='indigo', linewidth=3.5)
    plt.axhline(1.0, color='black', linewidth=1, linestyle='dashed')
    plt.text(0.02, 1.05, S['t_nom'], color='black', fontsize=16, fontweight='bold')
    cross_idx = np.where(y_senior >= 1.0)[0][0]
    cross_x = x[cross_idx]
    plt.plot(cross_x, 1.0, marker='o', markersize=8, color="red")
    plt.text(cross_x + 0.02, 0.82, f"{S['t_null']} (x={cross_x:.2f})", color='red', fontweight='bold', fontsize=15)
    plt.title(S['g15_title'], fontweight='bold', fontsize=20)
    plt.xlabel(S['g01_x'], fontsize=18)
    plt.ylabel(S['g15_y'], fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0.4, 6.2)
    plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "G15_Paradoxo_Especialista.png"))

    # ================= EMPÍRICOS =================
    if df is not None:
        # --- EXP 1 ---
        df_exp1 = df[(df['group'] == 'Exp1_Scalability') & (df['feasible'] == 1)]
        if not df_exp1.empty:
            plt.figure(figsize=(8, 5), dpi=300)
            sns.boxplot(data=df_exp1, x="N", y="runtime", palette="Blues_d")
            plt.title(S['g03_title'], fontweight="bold")
            plt.xlabel(S['x_task'])
            plt.ylabel(S['y_time_s'])
            plt.yscale("log") 
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G03_Scalability_Boxplot_Tasks.png"))

            plt.figure(figsize=(8, 5), dpi=300)
            sns.lineplot(data=df_exp1, x="N", y="runtime", marker="o", color="navy", linewidth=2)
            plt.title(S['g04_title'], fontweight="bold")
            plt.xlabel(S['x_task'])
            plt.ylabel(S['y_time_avg'])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G04_Scalability_Line_Tasks.png"))

            df_exp1['Vars_Bin'] = pd.qcut(df_exp1['NumVars'], q=5, precision=0, duplicates='drop')
            plt.figure(figsize=(8, 5), dpi=300)
            sns.boxplot(data=df_exp1, x="Vars_Bin", y="runtime", palette="Purples_d")
            plt.title(S['g05_title'], fontweight="bold")
            plt.xlabel(S['x_var_b'])
            plt.ylabel(S['y_time_s'])
            plt.xticks(rotation=15)
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G05_Scalability_Boxplot_Vars.png"))

            plt.figure(figsize=(8, 5), dpi=300)
            sns.lineplot(data=df_exp1, x="NumVars", y="runtime", marker="o", color="indigo", linewidth=2)
            plt.title(S['g06_title'], fontweight="bold")
            plt.xlabel(S['x_var_n'])
            plt.ylabel(S['y_time_avg'])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G06_Scalability_Line_Vars.png"))

        # --- EXP 2 ---
        df_exp2 = df[(df['group'] == 'Exp2_Behavioral') & (df['feasible'] == 1)]
        if not df_exp2.empty:
            plt.figure(figsize=(8, 5), dpi=300)
            sns.lineplot(data=df_exp2, x="Cenario", y="mksp", marker="o", color="darkorange", linewidth=2.5)
            plt.title(S['g07_title'], fontweight="bold")
            plt.xlabel(S['x_scen'])
            plt.ylabel(S['y_mksp_avg'])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G07_Behavioral_Line_Makespan.png"))

            plt.figure(figsize=(8, 5), dpi=300)
            sns.boxplot(data=df_exp2, x="Cenario", y="mksp", palette="Oranges")
            plt.title(S['g08_title'], fontweight="bold")
            plt.xlabel(S['x_scen'])
            plt.ylabel(S['y_mksp'])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G08_Behavioral_Boxplot_Makespan.png"))

            plt.figure(figsize=(8, 5), dpi=300)
            sns.barplot(data=df_exp2, x="Cenario", y="avg_EC", palette="autumn", ci="sd")
            plt.title(S['g09_title'], fontweight="bold")
            plt.xlabel(S['x_scen'])
            plt.ylabel(S['y_ec'])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G09_Behavioral_Bar_EC.png"))

        # --- EXP 3 ---
        df_exp3 = df[(df['group'] == 'Exp3_Heatmap') & (df['feasible'] == 1)]
        if not df_exp3.empty:
            pivot_df = df_exp3.pivot_table(values="mksp", index="M", columns="N", aggfunc="mean").round(0)
            plt.figure(figsize=(10, 6), dpi=300)
            sns.heatmap(pivot_df, annot=True, fmt="g", cmap="YlOrRd", linewidths=.5, 
                        cbar_kws={'label': S['y_mksp'], 'orientation': 'horizontal', 'location': 'bottom'})
            plt.title(S['g10_title'], fontweight="bold")
            plt.xlabel(S['x_task'])
            plt.ylabel(S['y_devs'])
            plt.savefig(os.path.join(OUTPUT_DIR, "G10_Brooks_Heatmap.png"), bbox_inches='tight')

            df_exp3_n50 = df_exp3[df_exp3['N'] == 50]
            if not df_exp3_n50.empty:
                plt.figure(figsize=(8, 5), dpi=300)
                sns.lineplot(data=df_exp3_n50, x="M", y="mksp", marker="s", color="firebrick", linewidth=2.5)
                plt.title(S['g11_title'], fontweight="bold")
                plt.xlabel(S['x_devs'])
                plt.ylabel(S['y_mksp'])
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, "G11_Brooks_Asymptote.png"))

        # --- EXP 4 ---
        df_exp4 = df[df['group'] == 'Exp4_Precedence']
        if not df_exp4.empty:
            df_viab = df_exp4.groupby("Density")["feasible"].mean().reset_index()
            df_viab["feasible"] *= 100
            
            plt.figure(figsize=(8, 5), dpi=300)
            sns.lineplot(data=df_exp4[df_exp4['feasible']==1], x="Density", y="mksp", marker="o", color="purple", linewidth=2)
            plt.title(S['g12_title'], fontweight="bold")
            plt.xlabel(S['x_dens'])
            plt.ylabel(S['y_mksp'])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G12_Precedence_Makespan.png"))

            plt.figure(figsize=(8, 5), dpi=300)
            sns.barplot(data=df_viab, x="Density", y="feasible", color="mediumpurple")
            plt.title(S['g13_title'], fontweight="bold")
            plt.xlabel(S['x_dens'])
            plt.ylabel(S['y_feas'])
            plt.ylim(0, 105)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G13_Precedence_Feasibility.png"))

        # --- EXP 5 ---
        df_exp5 = df[(df['group'] == 'Exp5_Surplus') & (df['feasible'] == 1)]
        if not df_exp5.empty:
            plt.figure(figsize=(8, 5), dpi=300)
            sns.lineplot(data=df_exp5, x="TechLvl", y="mksp", marker="D", color="teal", linewidth=2)
            plt.title(S['g14_title'], fontweight="bold")
            plt.xlabel(S['x_tech'])
            plt.ylabel(S['y_mksp'])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "G14_Surplus_Makespan.png"))
            
    # Fecha todos os plots da rodada para liberar memória antes do próximo idioma
    plt.close('all')

print("\n[OK] Pipeline finalizado. Gráficos gerados com sucesso.")
