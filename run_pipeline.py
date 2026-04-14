# -*- coding: utf-8 -*-
"""
Pipeline de Otimizacao - Behavioral SPSP

Este script é multiplataforma (Windows, macOS, Linux). Ele automatiza a limpeza 
de diretórios, instalação de dependências e a execução sequencial dos módulos do projeto.
"""

import os
import sys
import shutil
import subprocess

# Caminho absoluto da raiz do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_directories():
    """Remove pastas de dados dinâmicos para garantir uma execução limpa."""
    dirs_to_clean = [
        os.path.join(BASE_DIR, "data", "instances"),
        os.path.join(BASE_DIR, "data", "results"),
        os.path.join(BASE_DIR, "docs", "paper", "figures")
    ]
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            
def run_command(command, step_name):
    """Executa um comando no terminal e trava a execução em caso de erro."""
    print("-" * 60)
    print(f"{step_name}")
    
    # O shell=True é evitado por segurança, mas usado se for comando do sistema (como pip)
    result = subprocess.run(command, cwd=BASE_DIR)
    
    if result.returncode != 0:
        print(f"\n[ERRO FATAL] Ocorreu uma falha na etapa: {step_name}")
        print("Verifique os logs acima para corrigir o problema.")
        sys.exit(1)
    print(f"[OK] Etapa concluída com sucesso.")

def main():
    print("\n" + "=" * 60)
    print("       PIPELINE DE OTIMIZACAO - BEHAVIORAL SPSP")
    print("=" * 60)

    print("\n[1/5] Preparando o ambiente e limpando dados antigos...")
    clean_directories()
    print("[OK] Limpeza concluída.")

    # O comando sys.executable garante que ele use o mesmo Python que rodou o script
    python_exe = sys.executable

    run_command(
        [python_exe, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
        "[2/5] Verificando e instalando dependencias..."
    )

    run_command(
        [python_exe, os.path.join("src", "instance_gen.py")],
        "[3/5] Gerando as instancias parametricas para o modelo..."
    )

    run_command(
        [python_exe, os.path.join("src", "optimizer.py")],
        "[4/5] Resolvendo os modelos matematicos..."
    )

    run_command(
        [python_exe, os.path.join("src", "plot_gen.py")],
        "[5/5] Renderizando graficos e analises visuais..."
    )

    print("\n" + "=" * 60)
    print("[SUCESSO] Pipeline finalizado!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
