import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def gerar_graficos_classicos():
    # 1. Pega o diretório exato de onde ESTE script Python está salvo no seu PC
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Se o script estiver dentro da pasta 'src', ele sobe um nível para 'behavioral-spsp'
    if os.path.basename(script_dir) == 'src':
        raiz_projeto = os.path.dirname(script_dir)
    else:
        raiz_projeto = script_dir # Se já estiver na raiz, mantém.

    # 3. Monta o caminho exato baseado no seu print: behavioral-spsp/data/results/Cenario_*
    pasta_results = os.path.join(raiz_projeto, "data", "results")
    caminho_busca = os.path.join(pasta_results, "Cenario_*")
    
    pastas_resultados = sorted(glob.glob(caminho_busca))
    
    if not pastas_resultados:
        print(f"❌ Nenhuma pasta de cenário encontrada.")
        print(f"🔍 O script procurou EXATAMENTE neste caminho: {caminho_busca}")
        return

    # 4. Pasta para salvar os gráficos: behavioral-spsp/data/results/plots_classicos
    pasta_plots = os.path.join(pasta_results, "plots_classicos")
    os.makedirs(pasta_plots, exist_ok=True)
    
    sns.set_theme(style="white", palette="muted")
    print(f"🚀 Caminho encontrado! Gerando Gráficos Clássicos em: {pasta_plots}...\n")

    graficos_gerados = 0

    for pasta in pastas_resultados:
        nome_cenario = os.path.basename(pasta).replace("Cenario_", "")
        arquivo_csv = os.path.join(pasta, "allocations_super.csv")

        if not os.path.exists(arquivo_csv):
            print(f"⏩ Pulando '{nome_cenario}': Arquivo {os.path.basename(arquivo_csv)} não encontrado.")
            continue
            
        print(f"📊 Processando '{nome_cenario}'...")
        df = pd.read_csv(arquivo_csv)
        df = df.sort_values(by=["Inicio", "Tarefa"]).reset_index(drop=True)

        # =====================================================================
        # 1. GRÁFICO DE GANTT CLÁSSICO
        # =====================================================================
        plt.figure(figsize=(16, 7))
        tarefas_ordenadas = df["Tarefa"].unique()
        y_pos = np.arange(len(tarefas_ordenadas))
        tarefa_y_map = {t: i for i, t in enumerate(tarefas_ordenadas)}

        devs = sorted(df["Dev"].unique())
        cores_devs = sns.color_palette("Set2", len(devs))
        dev_cor_map = {dev: cores_devs[i] for i, dev in enumerate(devs)}

        for _, row in df.iterrows():
            y = tarefa_y_map[row["Tarefa"]]
            inicio_dias = row["Inicio"] / 8.0
            duracao_dias = row["T_Din"] / 8.0 
            plt.barh(y, duracao_dias, left=inicio_dias, color=dev_cor_map[row["Dev"]], 
                     edgecolor="black", alpha=0.9, height=0.6)
            dev_num = row["Dev"].replace("Dev_", "")
            plt.text(inicio_dias + duracao_dias/2, y, f"D{dev_num} ({row['Fracao']*100:.0f}%)", 
                     ha='center', va='center', color='black', fontsize=8, fontweight='bold')

        # Linhas de Sprint
        max_dias = (df["Inicio"] + df["T_Din"]).max() / 8.0
        for s in range(1, int(max_dias // 14) + 2):
            plt.axvline(x=s*14, color='red', linestyle='--', alpha=0.3)

        plt.yticks(y_pos, tarefas_ordenadas)
        plt.gca().invert_yaxis()
        plt.xlabel("Dias Úteis (1 Dia = 8h)", fontsize=12)
        plt.title(f"Gantt: {nome_cenario.replace('_', ' ')}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_plots, f"01_Gantt_{nome_cenario}.png"), dpi=300)
        plt.close()
        graficos_gerados += 1

        # =====================================================================
        # 2. DECOMPOSIÇÃO DO TEMPO CLÁSSICO
        # =====================================================================
        plt.figure(figsize=(20, 8))
        componentes = {
            "T. Nominal": [], "Bônus ET (-)": [], "Atraso EC": [], 
            "Contexto": [], "Comunicação": [], "Fadiga": [], 
            "Atrito Equipe": [], "Setup": [], "Bônus Apr. (-)": []
        }
        x_labels = []

        for _, row in df.iterrows():
            dev_str = row["Dev"].replace("Dev_", "")
            x_labels.append(f"S{row['Sprint']}-{row['Tarefa']}(D{dev_str})")
            frac = row["Fracao"]
            nom = row["T_Nominal_Original"] * frac
            base_calc = row["T_Base_Calculado"] * frac 
            componentes["T. Nominal"].append(nom)
            componentes["Bônus ET (-)"].append(nom * (row["F_Tech_ET"] - 1.0))
            componentes["Atraso EC"].append(nom * row["F_Tech_ET"] * (row["F_Beh_EC"] - 1.0))
            componentes["Contexto"].append(base_calc * row["P_Ctx"])
            componentes["Comunicação"].append(base_calc * row["P_Com"])
            componentes["Fadiga"].append(base_calc * row["P_Phi_Fadiga"])
            componentes["Atrito Equipe"].append(base_calc * row["P_Match"])
            componentes["Setup"].append(2.0)
            componentes["Bônus Apr. (-)"].append(-base_calc * row["B_Apr_Acumulado"])

        x = np.arange(len(x_labels))
        width = 0.08
        offsets = np.linspace(-4*width, 4*width, 9)
        cores = ["#34495e", "#2ecc71", "#e74c3c", "#9b59b6", "#3498db", "#d35400", "#f1c40f", "#95a5a6", "#1abc9c"]

        for i, (key, vals) in enumerate(componentes.items()):
            plt.bar(x + offsets[i], vals, width, label=key, color=cores[i])

        plt.axhline(0, color='black', linewidth=0.8)
        plt.xticks(x, x_labels, rotation=45, ha='right', fontsize=7)
        plt.ylabel("Horas de Esforço", fontsize=12)
        plt.xlabel("Alocações (Ordenadas Cronologicamente)", fontsize=12)
        plt.title(f"Decomposição de Tempo: {nome_cenario.replace('_', ' ')}", fontsize=16, fontweight='bold')
        plt.legend(ncol=3, loc="upper right", fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_plots, f"02_Decomposicao_{nome_cenario}.png"), dpi=300)
        plt.close()
        graficos_gerados += 1

    print(f"\n✅ Concluído! {graficos_gerados} imagens salvas em: {pasta_plots}")

if __name__ == "__main__":
    gerar_graficos_classicos()
