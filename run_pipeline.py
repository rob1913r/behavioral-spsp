# -*- coding: utf-8 -*-
"""
============================================================================
 run_pipeline.py — orquestrador (versão do MODELO × solver)
============================================================================

Menu em 2 telas:
  Tela 1 — versão do modelo:
    [1] ANTIGO  → solvers em src/optimizer/            → data/results/
    [2] NOVO    → solvers em src/optimizer_corrigido/  → data/results_v2/
  Tela 2 — solver: Gurobi | VNS | VNS-v2 | só-gráficos

A versão ANTIGA roda EXATAMENTE como antes (optimizer.*, data/results/,
validation.py). A versão NOVA roda os solvers CORRIGIDOS (modelo do artigo,
Eq. 1–67), grava em data/results_v2/ (mesma estrutura) e valida com
validation_v2.py — assim os dados antigos ficam intactos como backup.

A limpeza remove APENAS a pasta do solver escolhido DENTRO da versão escolhida
(ex.: rodar o VNS novo limpa só data/results_v2/vns/; nunca toca data/results/).

Uso:
  python run_pipeline.py                 # menu interativo (2 telas)
  python run_pipeline.py new vns         # versão NOVA, VNS
  python run_pipeline.py old gurobi      # versão ANTIGA, Gurobi
  python run_pipeline.py vns             # (compat) versão ANTIGA, VNS
  python run_pipeline.py plots           # (compat) versão ANTIGA, só gráficos
"""
import os
import sys
import glob
import shutil
import importlib

# Garante UTF-8 no Windows (acentos/emojis no console).
if os.name == "nt" and not sys.flags.utf8_mode:
    import subprocess
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    sys.exit(subprocess.run([sys.executable, "-X", "utf8"] + sys.argv, env=env).returncode)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

INSTANCES_DIR = os.path.join(BASE_DIR, "data", "instances")

LINE = "─" * 64


# ===========================================================================
# Configuração por versão de modelo
# ===========================================================================

def _make_version(label, results_base, opt_pkg, validation_mod):
    return {
        "label": label,
        "results_base": results_base,
        "opt": opt_pkg,
        "validation": validation_mod,
        "gurobi": os.path.join(BASE_DIR, "data", results_base, "gurobi"),
        "vns":    os.path.join(BASE_DIR, "data", results_base, "vns"),
        "vns_v2": os.path.join(BASE_DIR, "data", results_base, "vns_v2"),
    }

VERSIONS = {
    "old": _make_version("ANTIGO",           "results",    "optimizer",           "validation"),
    "new": _make_version("NOVO (corrigido)", "results_v2", "optimizer_corrigido", "validation_v2"),
}


def banner(title):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def step(msg):
    print(f"\n{LINE}\n  {msg}\n{LINE}")


def ensure_instances():
    """Gera as instâncias apenas se a pasta estiver vazia (não sobrescreve)."""
    os.makedirs(INSTANCES_DIR, exist_ok=True)
    found = sorted(glob.glob(os.path.join(INSTANCES_DIR, "*.json")))
    if found:
        print(f"        instâncias: {len(found)} encontradas (mantidas)")
    else:
        from instance_gen import generate_instances
        generate_instances()
        found = sorted(glob.glob(os.path.join(INSTANCES_DIR, "*.json")))
        print(f"        instâncias: {len(found)} geradas")
    return found


def clean_solver_results(solver_dir, label):
    """Remove os resultados de UM solver (mantém os demais intactos)."""
    if os.path.exists(solver_dir):
        shutil.rmtree(solver_dir)
    os.makedirs(os.path.join(solver_dir, "plots"), exist_ok=True)
    print(f"        resultados anteriores de {label}: limpos")


# ===========================================================================
# Fluxo VNS
# ===========================================================================

def run_vns(ver):
    solve_vns = importlib.import_module(f"{ver['opt']}.vns.solver").solve_vns
    validation = importlib.import_module(ver["validation"])
    import plot_gen

    VNS_DIR, GUROBI_DIR = ver["vns"], ver["gurobi"]
    plots_dir = os.path.join(VNS_DIR, "plots")

    step(f"[1/4] Preparando dados do VNS  (modelo {ver['label']})")
    instances = ensure_instances()
    clean_solver_results(VNS_DIR, "VNS")

    step("[2/4] Resolvendo com VNS")
    results = []
    for fp in instances:
        name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n  ► {name}")
        res = solve_vns(fp, show_console=False)
        results.append(res)
        ok = "factível" if res["Status"] == "Sucesso" else "INFACTÍVEL"
        print(f"        ✓ Tmax={res['Tmax']:.1f}h  gap={res['VNS_Gap (%)']}  "
              f"{ok}  (t={res['Tempo (s)']:.0f}s, {res['iteracoes']} iter)")
        plot_gen.plot_convergence(res["convergence_log"], name, res["LB"],
                                  res["T_limit"], plots_dir)
        plot_gen.plot_vnd_convergence(res["vnd_log"], name, plots_dir)

    step("[3/4] Validando factibilidade (LP do Gurobi)")
    validation.validate_all(INSTANCES_DIR, VNS_DIR)

    step("[4/4] Gráficos")
    plot_gen.plot_solution_charts(VNS_DIR, plots_dir)
    if glob.glob(os.path.join(GUROBI_DIR, "Cenario_*")):
        plot_gen.plot_comparison(GUROBI_DIR, VNS_DIR, plots_dir)
        plot_gen.report(BASE_DIR, results_base=ver["results_base"])
    else:
        print("        (sem baseline Gurobi — comparativo pulado)")

    _summary("VNS", results, plots_dir)


# ===========================================================================
# Fluxo VNS-v2 (matheurística híbrida VNS + Gurobi)
# ===========================================================================

def _has_precedence(fp):
    import json
    with open(fp) as f:
        return any(json.load(f)["Pred"])


def _solve_elite_pool(fp, name, seeds, solve_vns, vcfg, vns_v2_dir):
    """Roda solve_vns para várias seeds (pool de elite) e mantém a MELHOR
    solução factível. Usado em instâncias com precedências (C3 é bimodal)."""
    out_dir = os.path.join(vns_v2_dir, name)
    stash = out_dir + "__best"
    best = None
    for sd in seeds:
        vcfg.RANDOM_SEED = sd
        res = solve_vns(fp, show_console=False)
        feas = res["Status"] == "Sucesso"
        better = (best is None
                  or (feas and best["Status"] != "Sucesso")
                  or (feas == (best["Status"] == "Sucesso")
                      and res["Tmax"] < best["Tmax"]))
        print(f"        seed {sd}: Tmax={res['Tmax']:.1f}h "
              f"{'factível' if feas else 'INFACTÍVEL'}"
              f"{'   <-- melhor do pool' if better else ''}")
        if better:
            if os.path.exists(stash):
                shutil.rmtree(stash)
            shutil.copytree(out_dir, stash)
            best = res
    if os.path.exists(stash):       # restaura a melhor do pool no out_dir
        shutil.rmtree(out_dir)
        shutil.move(stash, out_dir)
    return best


def run_vns_v2(ver):
    solve_vns = importlib.import_module(f"{ver['opt']}.vns_v2.solver").solve_vns
    vcfg = importlib.import_module(f"{ver['opt']}.vns_v2.config")
    validation = importlib.import_module(ver["validation"])
    import plot_gen

    VNS_V2_DIR, GUROBI_DIR = ver["vns_v2"], ver["gurobi"]
    plots_dir = os.path.join(VNS_V2_DIR, "plots")

    step(f"[1/4] Preparando dados do VNS-v2 (matheurística)  (modelo {ver['label']})")
    instances = ensure_instances()
    clean_solver_results(VNS_V2_DIR, "VNS-v2")

    step("[2/4] Resolvendo com VNS-v2 (VNS + Gurobi em subproblemas)")
    results = []
    for fp in instances:
        name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n  ► {name}")
        if _has_precedence(fp) and len(vcfg.MH_POOL_SEEDS_PRED) > 1:
            res = _solve_elite_pool(fp, name, vcfg.MH_POOL_SEEDS_PRED,
                                    solve_vns, vcfg, VNS_V2_DIR)
        else:
            vcfg.RANDOM_SEED = vcfg.MH_BASE_SEED
            res = solve_vns(fp, show_console=False)
        results.append(res)
        ok = "factível" if res["Status"] == "Sucesso" else "INFACTÍVEL"
        print(f"        ✓ Tmax={res['Tmax']:.1f}h  gap={res['VNS_Gap (%)']}  "
              f"{ok}  (t={res['Tempo (s)']:.0f}s, {res['iteracoes']} iter, "
              f"MIP {res['mh_improves']}/{res['mh_calls']})")
        plot_gen.plot_convergence(res["convergence_log"], name, res["LB"],
                                  res["T_limit"], plots_dir)
        plot_gen.plot_vnd_convergence(res["vnd_log"], name, plots_dir)

    step("[3/4] Validando factibilidade (LP do Gurobi)")
    validation.validate_all(INSTANCES_DIR, VNS_V2_DIR)

    step("[4/4] Gráficos")
    plot_gen.plot_solution_charts(VNS_V2_DIR, plots_dir)
    if glob.glob(os.path.join(GUROBI_DIR, "Cenario_*")):
        plot_gen.plot_comparison(GUROBI_DIR, VNS_V2_DIR, plots_dir, label="VNS-v2")
        plot_gen.report(BASE_DIR, vns_subdir="vns_v2", label="VNS-v2",
                        results_base=ver["results_base"])
    else:
        print("        (sem baseline Gurobi — comparativo pulado)")

    _summary("VNS-v2", results, plots_dir)


# ===========================================================================
# Fluxo Gurobi
# ===========================================================================

def run_gurobi(ver):
    solve_gurobi = importlib.import_module(f"{ver['opt']}.gurobi.solver").solve_gurobi
    import plot_gen

    GUROBI_DIR = ver["gurobi"]
    plots_dir = os.path.join(GUROBI_DIR, "plots")

    print("\n  ATENÇÃO: o Gurobi é EXATO e pode levar HORAS por instância.")
    print(f"  Isto vai APAGAR e regerar o baseline em {os.path.relpath(GUROBI_DIR, BASE_DIR)}.")
    if input("  Continuar? [s/N] ").strip().lower() not in ("s", "sim", "y", "yes"):
        print("  Cancelado.")
        return

    step(f"[1/3] Preparando dados do Gurobi  (modelo {ver['label']})")
    instances = ensure_instances()
    clean_solver_results(GUROBI_DIR, "Gurobi")

    step("[2/3] Resolvendo com Gurobi (MILP exato)")
    results = []
    for fp in instances:
        name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n  ► {name}")
        res = solve_gurobi(fp)
        results.append({"Cenario": res["Cenario"], "Status": res["Status"],
                        "Gap": res["MIPGap (%)"], "Tempo (s)": res["Tempo (s)"]})
        print(f"        ✓ {res['Status']}  MIPGap={res['MIPGap (%)']}  "
              f"(t={res['Tempo (s)']:.0f}s)")

    step("[3/3] Gráficos")
    plot_gen.plot_solution_charts(GUROBI_DIR, plots_dir)

    banner("RESUMO GUROBI")
    for r in results:
        print(f"  {r['Cenario']:<34} {r['Status']:<10} gap={r['Gap']:>8}  "
              f"t={r['Tempo (s)']:.0f}s")
    print(f"\n  ✓ Pipeline Gurobi concluído. Gráficos em {plots_dir}\n")


# ===========================================================================
# Fluxo só-gráficos (não resolve; usa os CSVs existentes)
# ===========================================================================

def _regen_heuristic_plots(solver_dir, gurobi_dir, subdir, label, results_base):
    """Regenera convergência + comparativo vs Gurobi de uma heurística
    (VNS ou VNS-v2) a partir dos CSVs/logs já salvos."""
    import pandas as pd
    import plot_gen
    if not glob.glob(os.path.join(solver_dir, "Cenario_*")):
        return
    plots_dir = os.path.join(solver_dir, "plots")
    for scenario_dir in sorted(glob.glob(os.path.join(solver_dir, "Cenario_*"))):
        name = os.path.basename(scenario_dir)
        conv = os.path.join(scenario_dir, "convergence_log.csv")
        vndl = os.path.join(scenario_dir, "vnd_log.csv")
        if os.path.exists(conv):
            df = pd.read_csv(conv)
            r0 = df.iloc[0]
            lb = r0["tmax_best_h"] / (1 + r0["gap_pct"] / 100.0) if r0["gap_pct"] > -100 else 0.0
            t_limit = float(df["tempo_s"].max())
            plot_gen.plot_convergence(df, name, lb, t_limit, plots_dir)
        if os.path.exists(vndl):
            plot_gen.plot_vnd_convergence(pd.read_csv(vndl), name, plots_dir)
    if glob.glob(os.path.join(gurobi_dir, "Cenario_*")):
        plot_gen.plot_comparison(gurobi_dir, solver_dir, plots_dir, label=label)
        plot_gen.report(BASE_DIR, vns_subdir=subdir, label=label, results_base=results_base)


def run_plots_only(ver):
    import plot_gen
    step(f"Regenerando gráficos a partir dos CSVs existentes  (modelo {ver['label']})")
    for label, d in (("Gurobi", ver["gurobi"]), ("VNS", ver["vns"]), ("VNS-v2", ver["vns_v2"])):
        if glob.glob(os.path.join(d, "Cenario_*")):
            print(f"\n  ► {label}")
            plot_gen.plot_solution_charts(d, os.path.join(d, "plots"))

    _regen_heuristic_plots(ver["vns"], ver["gurobi"], "vns", "VNS", ver["results_base"])
    _regen_heuristic_plots(ver["vns_v2"], ver["gurobi"], "vns_v2", "VNS-v2", ver["results_base"])
    print("\n  ✓ Gráficos regenerados.\n")


def _summary(label, results, plots_dir):
    banner(f"RESUMO {label}")
    print(f"  {'Cenário':<34} {'Tmax':>9}  {'gap':>8}  {'factível':>9}")
    for r in results:
        ok = "sim" if r["Status"] == "Sucesso" else "NÃO"
        print(f"  {r['Cenario']:<34} {r['Tmax']:>8.1f}h  "
              f"{r['VNS_Gap (%)']:>8}  {ok:>9}")
    print(f"\n  ✓ Pipeline {label} concluído. Gráficos em {plots_dir}\n")


# ===========================================================================
# Menu
# ===========================================================================

RUNNERS = {"gurobi": run_gurobi, "vns": run_vns, "vns_v2": run_vns_v2,
           "plots": run_plots_only}


def _ask_version():
    print("  Qual versão do MODELO?")
    print("    [1] ANTIGO  (src/optimizer/ → data/results/)")
    print("    [2] NOVO    (src/optimizer_corrigido/, modelo do artigo → data/results_v2/)")
    return {"1": "old", "2": "new"}.get(input("  > ").strip())


def _ask_solver():
    print("\n  Escolha o solver:")
    print("    [1] Gurobi   (exato, lento — horas)")
    print("    [2] VNS      (metaheurística pura, minutos)")
    print("    [3] VNS-v2   (matheurística híbrida VNS + Gurobi)")
    print("    [4] Só gráficos (não resolve; usa os CSVs atuais)")
    return {"1": "gurobi", "2": "vns", "3": "vns_v2", "4": "plots"}.get(input("  > ").strip())


def main():
    banner("BEHAVIORAL SPSP — PIPELINE")

    args = [a.lower().replace("-", "_") for a in sys.argv[1:]]
    ver_key, solver = None, None
    for a in args:
        if a in ("old", "antigo", "v1"):
            ver_key = "old"
        elif a in ("new", "novo", "v2", "corrigido"):
            ver_key = "new"
        elif a in RUNNERS:
            solver = a
    # Compat: `python run_pipeline.py vns` (sem versão) → versão ANTIGA.
    if solver is not None and ver_key is None:
        ver_key = "old"

    if ver_key is None:
        ver_key = _ask_version()
        if ver_key is None:
            print("  Opção inválida. Encerrando.")
            return
    if solver is None:
        solver = _ask_solver()
        if solver is None:
            print("  Opção inválida. Encerrando.")
            return

    ver = VERSIONS[ver_key]
    banner(f"MODELO {ver['label']}  ×  {solver.upper()}")
    RUNNERS[solver](ver)


if __name__ == "__main__":
    main()
