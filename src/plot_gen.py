# -*- coding: utf-8 -*-
"""
============================================================================
 plot_gen.py — geração unificada de TODOS os gráficos e análises
============================================================================

Reúne num único módulo o que antes estava espalhado em seis arquivos
(plot_gen, plot_gen_classico, plot_gen_artigo_eps, plot_comparativo,
plot_convergencia e analise_comparativa). Os gráficos em formato .eps para
o artigo NÃO são mais gerados aqui — a conversão para .eps é feita sob
demanda na hora de montar o PDF.

Funções públicas:
  plot_solution_charts(results_dir, plots_dir)  — gráficos de uma solução
      (Gantt, decomposição de tempo, pizza de equipe, heatmaps, boxplots,
       violino) lidos de results_dir/<Cenario_*>/allocations.csv.
  plot_convergence(log, name, lb, t_limit, plots_dir)      — convergência
      externa do GVNS (Tmax real + objetivo penalizado + gap).
  plot_vnd_convergence(vnd_log, name, plots_dir)           — convergência
      interna do VND (entrada vs saída + melhorias por vizinhança).
  plot_comparison(gurobi_dir, vns_dir, plots_dir)          — comparativos
      gráficos Gurobi vs VNS (Tmax, tempo, gap).
  report(base_dir)                                         — análise textual
      estrutural Gurobi vs VNS (mesma família que os comparativos, mas em
      forma de tabela no terminal).

Todos os gráficos de um solver são salvos em UMA pasta (plots_dir), em vez
das três pastas separadas (brainstorming/classicos/artigo) usadas antes.
"""
import os
import glob
import re
import json
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOC_CSV = "allocations.csv"            # nome padrão do CSV de alocações
CAP_UTIL = (14 * 8 - 8) * 0.9            # 93.6h — capacidade útil por dev/sprint
MULTIPLIER_COLS = ["F_Total_Multiplicador", "P_Ctx", "P_Com", "P_Phi_Fadiga",
                   "P_Match", "B_Apr_Acumulado", "F_Beh_EC", "F_Tech_ET"]


def _scenario_dirs(results_dir):
    """Subpastas Cenario_* de um diretório de resultados, em ordem."""
    return sorted(glob.glob(os.path.join(results_dir, "Cenario_*")))


def _short_name(scenario_dir):
    """'Cenario_2_Monolitos_Equipe' -> '2_Monolitos_Equipe'."""
    return os.path.basename(scenario_dir).replace("Cenario_", "")


# ===========================================================================
# 1. GRÁFICOS POR SOLUÇÃO (Gantt, decomposição, pizza, heatmaps, boxplots)
# ===========================================================================

def _plot_per_scenario(results_dir, plots_dir):
    """Gantt detalhado, decomposição de tempo e pizza de equipe por cenário,
    além de acumular o DataFrame global para os comparativos da seção 2."""
    sns.set_theme(style="white", palette="muted")
    df_global = pd.DataFrame()

    for scenario_dir in _scenario_dirs(results_dir):
        name = _short_name(scenario_dir)
        csv_path = os.path.join(scenario_dir, ALLOC_CSV)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path).sort_values(["Inicio", "Tarefa"]).reset_index(drop=True)
        df_global = pd.concat([df_global, df.assign(Cenario=name)], ignore_index=True)

        # --- Gantt (1 dia = 8h) ---
        plt.figure(figsize=(16, 7))
        tasks = df["Tarefa"].unique()
        task_y = {t: i for i, t in enumerate(tasks)}
        devs = sorted(df["Dev"].unique())
        dev_color = {d: c for d, c in zip(devs, sns.color_palette("Set2", len(devs)))}
        for _, row in df.iterrows():
            y = task_y[row["Tarefa"]]
            start_d, dur_d = row["Inicio"] / 8.0, row["T_Din"] / 8.0
            plt.barh(y, dur_d, left=start_d, color=dev_color[row["Dev"]],
                     edgecolor="black", alpha=0.9, height=0.6)
            plt.text(start_d + dur_d / 2, y,
                     f"D{row['Dev'].replace('Dev_', '')} ({row['Fracao'] * 100:.0f}%)",
                     ha="center", va="center", fontsize=8, fontweight="bold")
        max_days = (df["Inicio"] + df["T_Din"]).max() / 8.0
        for s in range(1, int(max_days // 14) + 2):
            plt.axvline(x=s * 14, color="red", linestyle="--", alpha=0.3)
        plt.yticks(range(len(tasks)), tasks)
        plt.gca().invert_yaxis()
        plt.xlabel("Dias Úteis (1 Dia = 8h)", fontsize=12)
        plt.title(f"Gantt: {name.replace('_', ' ')}", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"gantt_{name}.png"), dpi=200)
        plt.close()

        # --- Decomposição do tempo dinâmico em componentes ---
        comps = {"T. Nominal": [], "Bônus ET (-)": [], "Atraso EC": [],
                 "Contexto": [], "Comunicação": [], "Fadiga": [],
                 "Atrito Equipe": [], "Setup": [], "Bônus Apr. (-)": []}
        x_labels = []
        for _, row in df.iterrows():
            x_labels.append(f"S{row['Sprint']}-{row['Tarefa']}(D{row['Dev'].replace('Dev_', '')})")
            frac = row["Fracao"]
            nom = row["T_Nominal_Original"] * frac
            base = row["T_Base_Calculado"] * frac
            comps["T. Nominal"].append(nom)
            comps["Bônus ET (-)"].append(nom * (row["F_Tech_ET"] - 1.0))
            comps["Atraso EC"].append(nom * row["F_Tech_ET"] * (row["F_Beh_EC"] - 1.0))
            comps["Contexto"].append(base * row["P_Ctx"])
            comps["Comunicação"].append(base * row["P_Com"])
            comps["Fadiga"].append(base * row["P_Phi_Fadiga"])
            comps["Atrito Equipe"].append(base * row["P_Match"])
            comps["Setup"].append(2.0)
            comps["Bônus Apr. (-)"].append(-base * row["B_Apr_Acumulado"])
        n = len(x_labels)
        fig_w = max(20, n * 0.30)
        width = max(0.025, min(0.08, 2.0 / n))
        offsets = np.linspace(-4 * width, 4 * width, 9)
        cores = ["#34495e", "#2ecc71", "#e74c3c", "#9b59b6", "#3498db",
                 "#d35400", "#f1c40f", "#95a5a6", "#1abc9c"]
        plt.figure(figsize=(fig_w, 8))
        x = np.arange(n)
        for i, (key, vals) in enumerate(comps.items()):
            plt.bar(x + offsets[i], vals, width, label=key, color=cores[i])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(x, x_labels, rotation=45, ha="right",
                   fontsize=max(4, min(7, int(120 / n))))
        plt.ylabel("Horas de Esforço", fontsize=12)
        plt.title(f"Decomposição de Tempo: {name.replace('_', ' ')}",
                  fontsize=16, fontweight="bold")
        plt.legend(ncol=3, loc="upper right", fontsize="small")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"decomposition_{name}.png"), dpi=200)
        plt.close()

        # --- Pizza: distribuição de tamanho de equipe ---
        team_size = df.groupby("Tarefa")["Dev"].nunique().value_counts().sort_index()
        plt.figure(figsize=(8, 8))
        plt.pie(team_size, labels=[f"{k} Devs" for k in team_size.index],
                autopct="%1.1f%%", startangle=140,
                colors=sns.color_palette("pastel"))
        plt.title(f"Trabalho em Equipe: {name.replace('_', ' ')}", fontweight="bold")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"team_pie_{name}.png"), dpi=150)
        plt.close()

    return df_global


def _plot_global(df_global, plots_dir):
    """Comparativos entre cenários do mesmo solver (boxplots, heatmaps,
    violino) — equivalentes aos antigos gráficos de 'brainstorming'."""
    if df_global.empty:
        return
    sns.set_theme(style="whitegrid", palette="muted")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_global, x="Cenario", y="F_Beh_EC", palette="Set2")
    plt.axhline(1.0, color="red", linestyle="--")
    plt.title("Distribuição do Fator Comportamental ($F_{beh}$) por Cenário",
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "boxplot_behavioral.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_global[df_global["B_Apr_Acumulado"] > 0],
                x="Cenario", y="B_Apr_Acumulado", palette="crest")
    plt.title("Distribuição do Bônus de Aprendizado ($B^{apr}$)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "boxplot_learning.png"), dpi=150)
    plt.close()

    util = df_global.groupby(["Cenario", "Sprint", "Dev"])["T_Din"].sum().reset_index()
    util["Utilizacao"] = util["T_Din"] / CAP_UTIL * 100
    for cen in df_global["Cenario"].unique():
        pivot = (util[util["Cenario"] == cen]
                 .pivot(index="Dev", columns="Sprint", values="Utilizacao").fillna(0))
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f",
                    cbar_kws={"label": "% Utilização"})
        plt.title(f"Aproveitamento de Sprint (%): {cen}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"heatmap_utilization_{cen}.png"), dpi=150)
        plt.close()

    multi = (df_global.groupby(["Cenario", "Sprint", "Dev"])["Tarefa"]
             .nunique().reset_index(name="Qtd"))
    pivot = multi.groupby(["Dev", "Cenario"])["Qtd"].mean().unstack().fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="Reds", annot=True, fmt=".1f")
    plt.title("Média de Tarefas Simultâneas por Sprint (Limite Weinberg = 5)",
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "heatmap_multitask.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_global, x="Cenario", y="P_Phi_Fadiga", palette="coolwarm")
    plt.title("Distribuição da Penalidade de Fadiga ($P_{\\phi}$) por Cenário",
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "boxplot_fatigue.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    cg = df_global.groupby("Cenario")["Cg_Idx_Sustentabilidade"].mean()
    sns.barplot(x=cg.index, y=cg.values, palette="plasma")
    plt.axhline(0.10, color="red", linestyle="--", label="Limiar L_fat")
    plt.title("Média do Índice de Carga Cognitiva ($Cg_{\\%}$)", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cognitive_load.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_global, x="Cenario", y="F_Total_Multiplicador",
                   inner="quart", palette="pastel")
    plt.axhline(1.0, color="red", linestyle="--")
    plt.title("Distribuição do Multiplicador Final de Tempo ($F_{total}$)",
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "violin_multiplier.png"), dpi=150)
    plt.close()


def plot_solution_charts(results_dir, plots_dir):
    """Gera todos os gráficos de solução de um solver em plots_dir."""
    if not _scenario_dirs(results_dir):
        print(f"   [plots] Nenhum cenário em {results_dir}.")
        return
    os.makedirs(plots_dir, exist_ok=True)
    df_global = _plot_per_scenario(results_dir, plots_dir)
    _plot_global(df_global, plots_dir)
    print(f"   [plots] Gráficos de solução salvos em {plots_dir}")


# ===========================================================================
# 2. CONVERGÊNCIA (externa do GVNS e interna do VND)
# ===========================================================================

def plot_convergence(convergence_log, name, lb, t_limit, plots_dir):
    """Convergência externa: Tmax real (sem penalidade), objetivo penalizado
    e gap ao longo das iterações. Lê o convergence_log (lista de dicts ou
    DataFrame) cujas colunas seguem o esquema gravado pelo solver."""
    df = (pd.DataFrame(convergence_log)
          if isinstance(convergence_log, list) else convergence_log.copy())
    if df.empty:
        return
    os.makedirs(plots_dir, exist_ok=True)
    if "fp_iter" not in df.columns:
        df["fp_iter"] = df["fp_best"]
    if "rejeitada_infactivel" not in df.columns:
        df["rejeitada_infactivel"] = 0

    it = df["iteracao"]
    tmax_best = df["tmax_best_h"]
    gap_final = float(df["gap_pct"].iloc[-1])
    tmax_final = float(tmax_best.iloc[-1])
    n_rej = int(df["rejeitada_infactivel"].sum())

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(
        f"Convergência VNS — {name}\n"
        f"Tmax final: {tmax_final:.1f}h (objetivo REAL, sem penalidade)  |  "
        f"Gap: {gap_final:.1f}%  |  Rejeitadas (infactíveis): {n_rej}  |  "
        f"T_limit: {t_limit:.0f}s", fontsize=11)

    ax1.step(it, tmax_best, where="post", color="steelblue", linewidth=2,
             label="Tmax do melhor (REAL, sem penalidade)", zorder=3)
    idx_imp = df["melhoria"] == 1
    if idx_imp.any():
        ax1.scatter(it[idx_imp], tmax_best[idx_imp], color="steelblue",
                    s=60, zorder=5, label="Melhoria aceita (factível)")
    ax1.axhline(lb, color="red", linestyle="--", linewidth=1.2,
                label=f"Lower Bound = {lb:.1f}h")
    ax1.set_ylabel("Tmax real (h)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(it, df["fp_iter"], color="gray", linewidth=0.9, alpha=0.7,
             label="f_pen da candidata (Tmax + penalidades)")
    idx_rej = df["rejeitada_infactivel"] == 1
    if idx_rej.any():
        ax2.scatter(it[idx_rej], df["fp_iter"][idx_rej], color="red",
                    marker="x", s=70, zorder=5,
                    label="Candidata rejeitada (violou restrição)")
    ax2.step(it, df["fp_best"], where="post", color="darkgreen",
             linewidth=1.4, alpha=0.8, label="f_pen do melhor")
    ax2.set_ylabel("Objetivo penalizado", fontsize=10)
    ax2.set_yscale("log")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax3.step(it, df["gap_pct"], where="post", color="darkorange",
             linewidth=2, label="Gap do melhor (%)")
    ax3.axhline(20.0, color="green", linestyle="--", linewidth=1.2,
                label="Alvo: 20%")
    ax3.set_ylabel("Gap (%)", fontsize=10)
    ax3.set_xlabel("Iteração VNS", fontsize=10)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plots_dir, f"convergence_vns_{name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [convergência] {out}")


def plot_vnd_convergence(vnd_log, name, plots_dir):
    """Convergência INTERNA do VND, chamada a chamada.

    Mostra o que acontece DENTRO de cada chamada do VND: o valor da solução
    que entra (▼ vermelho), a descida do objetivo penalizado a cada melhoria
    interna (ponto colorido pela vizinhança L1–L5 que a produziu) e o valor
    de saída — e então a próxima solução que entra, sua descida, e assim por
    diante. O `vnd_log` registra um evento por entrada/melhoria/saída, em
    ordem cronológica; cada 'entrada' inicia uma nova chamada.

    Dois painéis: (1) todas as chamadas concatenadas (visão global do
    serrilhado entra-desce-sai); (2) zoom nas primeiras chamadas, onde a
    descida interna de cada uma fica nítida.
    """
    from matplotlib.lines import Line2D

    df = pd.DataFrame(vnd_log) if isinstance(vnd_log, list) else vnd_log.copy()
    if df.empty:
        return
    df = df.reset_index(drop=True)
    os.makedirs(plots_dir, exist_ok=True)

    # --- Segmenta o log em chamadas de VND (cada 'entrada' abre uma nova) ---
    segments = []
    cur = None
    step = 0
    for _, r in df.iterrows():
        if r["evento"] == "entrada":
            if cur is not None:
                segments.append(cur)
            cur = {"x": [], "fp": [], "evt": [], "l": []}
        if cur is None:
            continue  # eventos antes da 1ª entrada (não deve ocorrer)
        cur["x"].append(step)
        cur["fp"].append(float(r["fp"]))
        cur["evt"].append(r["evento"])
        cur["l"].append(int(r["l"]))
        step += 1
    if cur is not None:
        segments.append(cur)
    if not segments:
        return

    n_calls = len(segments)
    n_improv = int((df["evento"] == "melhoria").sum())
    n_timeout = int((df["evento"] == "timeout").sum())
    L_color = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#C44E52", 5: "#8172B3"}

    def draw(ax, segs):
        """Desenha cada chamada como uma descida ligada; entrada=▼, melhorias
        coloridas por vizinhança, saída interrompida por tempo = x cinza."""
        for seg in segs:
            ax.plot(seg["x"], seg["fp"], color="0.65", lw=0.8, alpha=0.7, zorder=1)
            ax.scatter(seg["x"][0], seg["fp"][0], marker="v", color="red",
                       s=30, zorder=4)                       # solução ENTRA
            for x, fp, evt, l in zip(seg["x"], seg["fp"], seg["evt"], seg["l"]):
                if evt == "melhoria":
                    ax.scatter(x, fp, color=L_color.get(l, "black"),
                               s=24, zorder=3)                # melhoria interna
                elif evt == "timeout":
                    ax.scatter(x, fp, marker="x", color="0.4",
                               s=28, zorder=3)                # saída por tempo
        ax.set_yscale("log")
        ax.set_ylabel("Objetivo penalizado (f_pen)", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(
        f"Convergência INTERNA do VND — {name}\n"
        f"Chamadas de VND: {n_calls}  |  Melhorias internas: {n_improv}  |  "
        f"Interrompidas por tempo: {n_timeout}", fontsize=11)

    draw(ax1, segments)
    ax1.set_title("Todas as chamadas (▼ = solução entra; ponto colorido = "
                  "melhoria interna por vizinhança)", fontsize=9)
    ax1.set_xlabel("Passo interno acumulado (todas as chamadas do VND)", fontsize=10)
    handles = [Line2D([0], [0], marker="v", color="w", markerfacecolor="red",
                      markersize=9, label="solução entra no VND")]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=L_color[l],
                       markersize=8, label=f"melhoria L{l}") for l in (1, 2, 3, 4, 5)]
    ax1.legend(handles=handles, fontsize=8, loc="upper right", ncol=3)

    n_zoom = min(12, n_calls)
    draw(ax2, segments[:n_zoom])
    ax2.set_title(f"Zoom: primeiras {n_zoom} chamadas — descida interna "
                  "detalhada de cada solução", fontsize=9)
    ax2.set_xlabel("Passo interno acumulado (zoom)", fontsize=10)

    plt.tight_layout()
    out = os.path.join(plots_dir, f"convergence_vnd_{name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [convergência-vnd] {out}")


# ===========================================================================
# 3. COMPARATIVO GUROBI vs VNS (Tmax, tempo, gap)
# ===========================================================================

def _read_summaries(results_dir):
    """Resumo (cenário, Tmax, gap, runtime) de cada cenário de um solver."""
    rows = []
    for scenario_dir in _scenario_dirs(results_dir):
        csv_path = os.path.join(scenario_dir, ALLOC_CSV)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        gap_col = "MIPGap" if "MIPGap" in df.columns else "VNS_Gap"
        rows.append({
            "scenario": "_".join(os.path.basename(scenario_dir).split("_")[:3]),
            "tmax": df["Fim"].max(),
            "gap": df[gap_col].iloc[0] if gap_col in df.columns else 0.0,
            "runtime": df["Runtime"].iloc[0] if "Runtime" in df.columns else 0.0,
        })
    return rows


def plot_comparison(gurobi_dir, vns_dir, plots_dir, label="VNS"):
    """Gera comparison_tmax.png, comparison_time.png e comparison_gap.png.
    `label` nomeia o solver não-Gurobi nos títulos/legendas (ex.: 'VNS-v2')."""
    g = _read_summaries(gurobi_dir)
    v = {r["scenario"]: r for r in _read_summaries(vns_dir)}
    if not g or not v:
        print("   [comparativo] Faltam CSVs de Gurobi ou VNS.")
        return
    os.makedirs(plots_dir, exist_ok=True)

    rows = []
    for rg in g:
        c = rg["scenario"]
        if c not in v:
            continue
        rv = v[c]
        rows.append({
            "scenario": c, "tmax_g": rg["tmax"], "tmax_v": rv["tmax"],
            "delta": (rv["tmax"] - rg["tmax"]) / rg["tmax"] * 100 if rg["tmax"] else 0,
            "gap_g": rg["gap"], "gap_v": rv["gap"],
            "rt_g": rg["runtime"], "rt_v": rv["runtime"],
            "speedup": rg["runtime"] / rv["runtime"] if rv["runtime"] else float("inf"),
        })
    if not rows:
        print("   [comparativo] Nenhum cenário em comum.")
        return
    df = pd.DataFrame(rows)
    x = range(len(df))
    w = 0.35
    labels = [c.replace("Cenario_", "C").replace("_", " ") for c in df["scenario"]]
    cg, cv = "steelblue", "darkorange"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], df["tmax_g"], w, label="Gurobi", color=cg)
    ax.bar([i + w / 2 for i in x], df["tmax_v"], w, label=label, color=cv)
    for i, row in df.iterrows():
        sign = "+" if row["delta"] >= 0 else ""
        ax.annotate(f"{sign}{row['delta']:.1f}%", xy=(i + w / 2, row["tmax_v"]),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8, color=cv)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Tmax (horas)")
    ax.set_title(f"Comparativo Tmax — Gurobi vs {label}")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparison_tmax.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], df["rt_g"], w, label="Gurobi", color=cg)
    ax.bar([i + w / 2 for i in x], df["rt_v"], w, label=label, color=cv)
    for i, row in df.iterrows():
        if row["speedup"] < float("inf"):
            ax.annotate(f"{row['speedup']:.0f}x",
                        xy=(i, max(row["rt_g"], row["rt_v"])),
                        xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=9, color="green", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Tempo (s) — escala log")
    ax.set_title(f"Comparativo Tempo — Gurobi vs {label}")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparison_time.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], df["gap_g"], w, label="MIPGap Gurobi", color=cg)
    ax.bar([i + w / 2 for i in x], df["gap_v"], w, label=f"{label}_Gap", color=cv)
    ax.axhline(20, color="green", linestyle="--", linewidth=1, label="Alvo: 20%")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Gap (%)")
    ax.set_title(f"Comparativo Gap — Gurobi vs {label}\n"
                 "(MIPGap e Gap da heurística medem conceitos distintos)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "comparison_gap.png"), dpi=150)
    plt.close()
    print(f"   [comparativo] 3 figuras (Gurobi vs {label}) salvas em {plots_dir}")


# ===========================================================================
# 4. ANÁLISE TEXTUAL ESTRUTURAL Gurobi vs VNS (relatório no terminal)
# ===========================================================================

def summarize_solution(df: pd.DataFrame) -> dict:
    """Extrai métricas estruturais de um CSV de alocações."""
    by_task = df.groupby("Tarefa").agg(
        n_devs=("Dev", "nunique"), sprint=("Sprint", "first"))
    load = df.groupby(["Dev", "Sprint"])["T_Din"].sum()
    return {
        "tmax": (df["Inicio"] + df["Duracao_Tarefa"]).max(),
        "sprints": [int(s) for s in sorted(df["Sprint"].unique())],
        "tasks_per_sprint": {int(k): int(v) for k, v
                             in by_task["sprint"].value_counts().sort_index().items()},
        "teams": {int(k): int(v) for k, v
                  in by_task["n_devs"].value_counts().sort_index().items()},
        "max_load": load.max(),
        "n_over_cap": int((load > CAP_UTIL + 1e-6).sum()),
        "mult": {c: (df[c].mean(), df[c].min(), df[c].max())
                 for c in MULTIPLIER_COLS if c in df.columns},
    }


def _print_side_by_side(gurobi: dict, vns: dict, label: str = "VNS"):
    print(f"  {'':24} {'GUROBI':>22} {label.upper():>22}")
    delta = (vns["tmax"] / gurobi["tmax"] - 1) * 100 if gurobi["tmax"] else 0
    print(f"  {'Tmax':24} {gurobi['tmax']:>21.1f}h {vns['tmax']:>21.1f}h"
          f"   (delta {vns['tmax'] - gurobi['tmax']:+.1f}h = {delta:+.1f}%)")
    print(f"  {'Sprints usados':24} {str(gurobi['sprints']):>22} {str(vns['sprints']):>22}")
    print(f"  {'Tarefas/sprint':24} {str(gurobi['tasks_per_sprint']):>22} "
          f"{str(vns['tasks_per_sprint']):>22}")
    print(f"  {'Equipes (d: n_tarefas)':24} {str(gurobi['teams']):>22} {str(vns['teams']):>22}")
    print(f"  {'Carga max (dev,sprint)':24} {gurobi['max_load']:>21.1f}h "
          f"{vns['max_load']:>21.1f}h")
    print(f"  {'Cargas > CapUtil':24} {gurobi['n_over_cap']:>22} {vns['n_over_cap']:>22}")
    print("  Multiplicadores (media [min, max]):")
    for c in MULTIPLIER_COLS:
        if c in gurobi["mult"] and c in vns["mult"]:
            g, v = gurobi["mult"][c], vns["mult"][c]
            print(f"    {c:32} {g[0]:6.3f} [{g[1]:6.3f},{g[2]:6.3f}]   "
                  f"{v[0]:6.3f} [{v[1]:6.3f},{v[2]:6.3f}]")


def report(base_dir: str = BASE_DIR, vns_subdir: str = "vns", label: str = "VNS",
           results_base: str = "results"):
    """Relatório textual comparativo Gurobi vs `vns_subdir` para cada cenário.
    `results_base` = "results" (modelo antigo) ou "results_v2" (modelo corrigido)."""
    for inst_path in sorted(glob.glob(os.path.join(base_dir, "data", "instances", "*.json"))):
        with open(inst_path) as f:
            name = json.load(f)["_params"]["Nome"]
        p_gurobi = os.path.join(base_dir, "data", results_base, "gurobi", name, ALLOC_CSV)
        p_vns = os.path.join(base_dir, "data", results_base, vns_subdir, name, ALLOC_CSV)
        print("\n" + "=" * 100)
        print(f"  {name}")
        print("=" * 100)
        if not (os.path.exists(p_gurobi) and os.path.exists(p_vns)):
            print(f"  [faltam CSVs de Gurobi e/ou {label}]")
            continue
        _print_side_by_side(summarize_solution(pd.read_csv(p_gurobi)),
                            summarize_solution(pd.read_csv(p_vns)), label)


if __name__ == "__main__":
    report()
