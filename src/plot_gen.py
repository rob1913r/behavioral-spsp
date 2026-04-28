import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 30})

def gerar_brainstorming_completo():
    pastas_resultados = sorted(glob.glob("data/results/Cenario_*"))
    if not pastas_resultados:
        print("❌ Nenhum resultado Super CSV encontrado. Rode o optimizer.py novo.")
        return
        
    pasta_plots = "data/results/plots_brainstorming"
    os.makedirs(pasta_plots, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # DataFrame Mestre para análises comparativas globais
    df_global = pd.DataFrame()
    
    print(f"🚀 Iniciando Geração de ~30 Gráficos em {pasta_plots}...")

    # =========================================================================
    # PARTE 1: GRÁFICOS OBRIGATÓRIOS POR INSTÂNCIA (3x3 = 9 Gráficos)
    # =========================================================================
    for pasta in pastas_resultados:
        nome_cen_full = os.path.basename(pasta)
        nome_cen = nome_cen_full.replace("Cenario_", "")
        arquivo_csv = os.path.join(pasta, "allocations_super.csv")
        
        if os.path.exists(arquivo_csv):
            df = pd.read_csv(arquivo_csv)
            # Adiciona ao global para depois
            df["Cenario_Nome"] = nome_cen
            df_global = pd.concat([df_global, df], ignore_index=True)
            
            # --- 1. Gráfico de Gantt Tradicional ---
            plt.figure(figsize=(12, 6))
            cores_devs = sns.color_palette("husl", df["Dev"].nunique())
            for idx, dev in enumerate(sorted(df["Dev"].unique())):
                df_dev = df[df["Dev"] == dev]
                for _, row in df_dev.iterrows():
                    plt.barh(y=row["Dev"], width=row["T_Din"], left=row["Inicio"], 
                             color=cores_devs[idx], edgecolor="black", alpha=0.8)
                    plt.text(row["Inicio"] + row["T_Din"]/2, row["Dev"], row["Tarefa"].replace("T",""), 
                             ha='center', va='center', color='black', fontsize=7)
            
            plt.title(f"Gantt: {nome_cen}", fontweight='bold')
            plt.xlabel("Horas Úteis")
            plt.tight_layout()
            plt.savefig(f"{pasta_plots}/01_Gantt_{nome_cen}.png", dpi=150)
            plt.close()

            # --- 2. Decomposição de Tempo: Nominal vs. Dinâmico (Múltiplos e Bônus) ---
            # Vamos pegar as 10 tarefas mais longas para não poluir
            df_tarefa = df.groupby("Tarefa").agg({
                "T_Nominal_Original": "first",
                "T_Din": "sum",
                "F_Beh_EC": "mean",
                "P_Phi_Fadiga": "mean",
                "B_Apr_Acumulado": "mean"
            }).reset_index().nlargest(10, "T_Din")
            
            x = np.arange(len(df_tarefa))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width/2, df_tarefa["T_Nominal_Original"], width, label='Tempo Nominal', color='gray', alpha=0.5)
            ax.bar(x + width/2, df_tarefa["T_Din"], width, label='Tempo Dinâmico (Final)', color='darkred', alpha=0.8)
            
            ax.set_ylabel('Horas')
            ax.set_title(f"Decomposição de Tempo (Top 10): {nome_cen}", fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(df_tarefa["Tarefa"], rotation=45)
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{pasta_plots}/02_Tempo_Decomposicao_{nome_cen}.png", dpi=150)
            plt.close()

            # --- 3. Pizza: Distribuição de Tamanho de Equipe por Tarefa ---
            df_equipe = df.groupby("Tarefa")["Dev"].nunique().reset_index(name="Tam_Equipe")
            dist_equipe = df_equipe["Tam_Equipe"].value_counts().sort_index()
            
            plt.figure(figsize=(8, 8))
            labels = [f"{k} Devs" for k in dist_equipe.index]
            plt.pie(dist_equipe, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
            plt.title(f"Trabalho em Equipe: {nome_cen}", fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"{pasta_plots}/03_Pizza_Equipe_{nome_cen}.png", dpi=150)
            plt.close()

    # =========================================================================
    # PARTE 2: BRAINSTORMING CRIATIVO (Comparativos e Malucos)
    # =========================================================================
    if df_global.empty: return

    # --- 04. Barras: Comparação de Makespan Total (Dias Úteis) ---
    plt.figure(figsize=(10, 6))
    makespan = df_global.groupby("Cenario_Nome")["Fim"].max() / 8.0 # Convertendo p dias
    sns.barplot(x=makespan.index, y=makespan.values, palette="muted")
    plt.title("Makespan Total do Projeto (Dias Úteis)", fontweight='bold')
    plt.ylabel("Dias")
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/04_Global_Makespan_Dias.png", dpi=150)
    plt.close()

    # --- 05. Barras: Tempo de Execução do Solver (Segundos) ---
    plt.figure(figsize=(10, 6))
    runtime = df_global.groupby("Cenario_Nome")["Runtime"].first()
    sns.barplot(x=runtime.index, y=runtime.values, palette="dark")
    plt.title("Tempo de Execução do Solver Gurobi", fontweight='bold')
    plt.ylabel("Segundos")
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/05_Global_Runtime.png", dpi=150)
    plt.close()

    # --- 06. Barras: MIPGap Final alcançado ---
    plt.figure(figsize=(10, 6))
    gap = df_global.groupby("Cenario_Nome")["MIPGap"].first()
    sns.barplot(x=gap.index, y=gap.values, palette="deep")
    plt.title("MIPGap Final do Solver (%)", fontweight='bold')
    plt.ylabel("%")
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/06_Global_MIPGap.png", dpi=150)
    plt.close()

    # --- 07. Boxplot Comparativo: Penalidade Comportamental (F_Beh_EC) ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_global, x="Cenario_Nome", y="F_Beh_EC", palette="Set2")
    plt.axhline(1.0, color='red', linestyle='--')
    plt.title("Distribuição do Fator Comportamental ($F_{beh}$) por Cenário", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/07_Boxplot_EC_Comparativo.png", dpi=150)
    plt.close()

    # --- 08. Boxplot Comparativo: Bônus de Aprendizado Capturado ---
    plt.figure(figsize=(12, 6))
    # Filtra apenas alocações que tiveram bônus p não sujar o boxplot
    sns.boxplot(data=df_global[df_global["B_Apr_Acumulado"] > 0], x="Cenario_Nome", y="B_Apr_Acumulado", palette="crest")
    plt.title("Distribuição do Bônus de Aprendizado Capturado ($B^{apr}$)", fontweight='bold')
    plt.ylabel("Redução Percentual")
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/08_Boxplot_Aprendizado.png", dpi=150)
    plt.close()

    # --- 09. Stacked Bar: Composição Total de Horas de Projeto ---
    df_global_horas = df_global.groupby("Cenario_Nome").agg({
        "T_Nominal_Original": "sum",
        "T_Din": "sum"
    }).reset_index()
    # "Atrito Sistêmico" = Dinâmico - Nominal (pode ser negativo se bônus > penalidade)
    df_global_horas["Atrito Sistêmico (Púnição - Bônus)"] = df_global_horas["T_Din"] - df_global_horas["T_Nominal_Original"]
    
    plt.figure(figsize=(10, 6))
    plt.bar(df_global_horas["Cenario_Nome"], df_global_horas["T_Nominal_Original"], label='Horas Base', color='gray', alpha=0.6)
    # Mostra apenas sobrecarga positiva p não complicar o gráficoStacked
    plt.bar(df_global_horas["Cenario_Nome"], np.maximum(0, df_global_horas["Atrito Sistêmico (Púnição - Bônus)"]), 
            bottom=df_global_horas["T_Nominal_Original"], label='Sobrecarga Otimizada', color='darkred')
    plt.title("Composição Total de Horas Alocadas", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/09_Global_Composicao_Horas.png", dpi=150)
    plt.close()

    # --- 10. Mapa de Calor: % de Aproveitamento da Sprint (Trabalho Real vs. Capacidade) ---
    # Capacidade útil por Dev/Sprint = 92.4h (14 dias * 8h * 0.9 buffer - 8h cerimonia)
    capacidade_util = (14 * 8 - 8) * 0.9 
    df_util = df_global.groupby(["Cenario_Nome", "Sprint", "Dev"])["T_Din"].sum().reset_index()
    df_util["Utilizacao_%"] = (df_util["T_Din"] / capacidade_util) * 100
    
    # Vamos gerar um mapa de calor para CADA cenário p não poluir
    for cen in df_global["Cenario_Nome"].unique():
        df_cen = df_util[df_util["Cenario_Nome"] == cen]
        pivot_util = df_cen.pivot(index="Dev", columns="Sprint", values="Utilizacao_%").fillna(0)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_util, cmap="YlGnBu", annot=True, fmt=".0f", cbar_kws={'label': '% Utilização'})
        plt.title(f"Aproveitamento de Sprint (%): {cen}", fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{pasta_plots}/10_Heatmap_Utilizacao_{cen}.png", dpi=150)
        plt.close()

    # --- 11. Mapa de Calor Global: Multitarefa (Tarefas Simultâneas por Sprint) ---
    df_multi = df_global.groupby(["Cenario_Nome", "Sprint", "Dev"])["Tarefa"].nunique().reset_index(name="Qtd_Tarefas")
    # Média global de multitarefa p cada Dev/Cenário
    pivot_multi = df_multi.groupby(["Dev", "Cenario_Nome"])["Qtd_Tarefas"].mean().unstack().fillna(0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_multi, cmap="Reds", annot=True, fmt=".1f")
    plt.title("Média de Tarefas Simultâneas por Sprint (Limite Weinberg = 5)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/11_Global_Heatmap_Multitarefa.png", dpi=150)
    plt.close()

    # --- 12. Ranking: Top 5 Devs que mais capturaram Bônus de Aprendizado ---
    plt.figure(figsize=(10, 6))
    rank_apr = df_global.groupby("Dev")["B_Apr_Acumulado"].sum().nlargest(5)
    sns.barplot(x=rank_apr.index, y=rank_apr.values, palette="viridis")
    plt.title("Ranking: Top 5 Devs Ecoeficientes (Soma de Bônus $B^{apr}$)", fontweight='bold')
    plt.ylabel("Soma de Redução Percentual")
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/12_Ranking_Eco_Devs.png", dpi=150)
    plt.close()

    # --- 13. Ranking: Top 5 Devs que mais sofreram com EC (EC > 1.0) ---
    plt.figure(figsize=(10, 6))
    rank_ec = df_global[df_global["F_Beh_EC"] > 1.0].groupby("Dev")["T_Din"].count().nlargest(5)
    sns.barplot(x=rank_ec.index, y=rank_ec.values, palette="magma")
    plt.title("Ranking: Top 5 Devs com Maior Volume de Atrito EC", fontweight='bold')
    plt.ylabel("Qtd de Alocações com $F_{beh} > 1$")
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/13_Ranking_Atrito_Devs.png", dpi=150)
    plt.close()

    # --- 14. Histograma: Distribuição de todas as Durações de Tarefa Finais ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df_global["T_Din"], kde=True, bins=30, color="darkblue")
    plt.title("Distribuição Global de Durações de Tarefa Finais", fontweight='bold')
    plt.xlabel("Horas")
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/14_Global_Histograma_Duracao.png", dpi=150)
    plt.close()

    # --- 15. Scatter Plot Maluco: Complexidade Técnica vs. Atrito Comportamental ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_global, x="F_Tech_ET", y="F_Beh_EC", hue="Cenario_Nome", style="Dev", s=100, alpha=0.6)
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.axvline(1.0, color='gray', linestyle='--')
    plt.title("Relação Maluca: Complexidade Técnica vs. Atrito EC", fontweight='bold')
    plt.xlabel("Fator Técnico ($F_{tech}$: 1=Fácil, 0.1=Difícil)")
    plt.ylabel("Fator Comportamental ($F_{beh}$: 1=Alinhado, >1=Ruído)")
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/15_ScatterMalco_ET_vs_EC.png", dpi=150)
    plt.close()

    # --- 16. Boxplot Maluco: Distribuição de Fadiga acumulada na Sprint ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_global, x="Cenario_Nome", y="P_Phi_Fadiga", palette="coolwarm")
    plt.title("Distribuição da Penalidade de Fadiga ($P_{\phi}$) por Cenário", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/16_Boxplot_Fadiga.png", dpi=150)
    plt.close()

    # --- 17. Barras: Média global do Índice de Carga Cognitiva sustentável ---
    plt.figure(figsize=(10, 6))
    cg_idx = df_global.groupby("Cenario_Nome")["Cg_Idx_Sustentabilidade"].mean()
    sns.barplot(x=cg_idx.index, y=cg_idx.values, palette="plasma")
    plt.axhline(0.10, color='red', linestyle='--', label='Limiar L_fat')
    plt.title("Média do Índice de Carga Cognitiva ($Cg_{\%}$) (Sustentabilidade)", fontweight='bold')
    plt.ylabel("Índice (0 a 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/17_Global_Carga_Cognitiva.png", dpi=150)
    plt.close()

    # --- 18. Mapa de Calor Global: Média de Penalidade de Contexto por Dev/Cenário ---
    plt.figure(figsize=(10, 8))
    pivot_ctx = df_global.groupby(["Dev", "Cenario_Nome"])["P_Ctx"].mean().unstack().fillna(0)
    sns.heatmap(pivot_ctx, cmap="Oranges", annot=True, fmt=".2f")
    plt.title("Média de Penalidade por Contexto ($P_{ctx}$) por Dev", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/18_Global_Heatmap_Contexto.png", dpi=150)
    plt.close()

    # --- 19. Radar Plot (Maluco): Comparação de Bônus e Penalidades Médias ---
    # Este precisa de processamento extra p normalizar
    metricas = ["F_Beh_EC", "P_Phi_Fadiga", "B_Apr_Acumulado", "P_Ctx"]
    df_radar = df_global.groupby("Cenario_Nome")[metricas].mean().reset_index()
    # Normaliza B_Apr p ficar na mesma escala (0-1)
    df_radar["B_Apr_Acumulado"] = df_radar["B_Apr_Acumulado"] / df_radar["B_Apr_Acumulado"].max()
    # Normaliza Fadiga
    df_radar["P_Phi_Fadiga"] = df_radar["P_Phi_Fadiga"] / df_radar["P_Phi_Fadiga"].max()
    
    # Plotar Radar exige matplotlib puro, vamos pular esse p simplificar o brainstorming
    # p/ o SBPO e fazer um ViolinPlot comparativo de Multiplicadores Finais

    # --- 19. ViolinPlot Comparativo de Multiplicadores Finais de Tempo ---
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_global, x="Cenario_Nome", y="F_Total_Multiplicador", inner="quart", palette="pastel")
    plt.axhline(1.0, color='red', linestyle='--')
    plt.title("Distribuição do Multiplicador Final de Tempo ($F_{total}$)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/19_Violin_Multiplicador_Final.png", dpi=150)
    plt.close()

    # --- 20. Tabela Plot (Maluco): Ranking de Tarefas por Atrito Comportamental Médio ---
    tarefas_ranking_ec = df_global.groupby("Tarefa")["F_Beh_EC"].mean().nlargest(10).reset_index()
    tarefas_ranking_ec.columns = ["Tarefa", "Atrito EC Médio"]
    
    # Gera uma imagem de tabela
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    tbl = plt.table(cellText=tarefas_ranking_ec.values, colLabels=tarefas_ranking_ec.columns, loc='center', cellLoc='center')
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)
    plt.title("Ranking: Top 10 Tarefas com Maior Atrito Comportamental Médio", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{pasta_plots}/20_Tabela_Ranking_EC_Tarefas.png", dpi=150)
    plt.close()

    print(f"✅ Brainstorming Finalizado! Cerca de 20 gráficos comparativos gerados além dos 9 obrigatórios.")

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 100})
    gerar_brainstorming_completo()
