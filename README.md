# Behavioral SPSP: Otimização de Alocação de Tarefas em Projetos de Software

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Gurobi Optimizer](https://img.shields.io/badge/solver-Gurobi-red)
![License](https://img.shields.io/badge/license-MIT-green)

Este repositório contém o código-fonte e a formulação matemática de um projeto de pesquisa focado na resolução do clássico **Problema de Escalonamento de Projetos de Software (SPSP)** através da Programação Linear Inteira Mista (MILP).

O grande diferencial deste modelo é o rompimento com a visão tradicional que considera apenas restrições técnicas. Aqui, o tempo de conclusão de uma tarefa é uma variável dinâmica calculada com base no **Excedente Técnico** e no **Erro Comportamental** do desenvolvedor, integrando a metodologia psicométrica DISC (Dominância, Influência, Estabilidade, Conformidade) diretamente às restrições do modelo.

**Principais Contribuições do Modelo:**
- 📉 Simulação matemática e comprovação tática da empírica **Lei de Brooks**.
- 🧩 Comprovação do *Trade-off* Técnico-Comportamental (um desenvolvedor Júnior alinhado pode superar um Sênior sob alto ruído cognitivo).
- 🔥 Geração de Mapas de Calor para planejamento de capacidade tática de equipes ágeis.

---

## 📁 Estrutura do Repositório

A arquitetura do projeto segue o seguinte padrão para facilitar a reprodutibilidade dos experimentos. Alguns diretórios e arquivos de saída não acompanham o repositório no momento do clone, pois são gerados dinamicamente durante a execução do código:

```text
behavioral-spsp/
│
├── data/                    # Diretório estruturado dinamicamente durante a execução do pipeline
│   ├── instances/           # Instâncias geradas sinteticamente divididas por experimento (exp1_scalability, exp2_behavioral, etc.)
│   └── results/             # Resultados consolidados do otimizador (metrics.csv) e pastas de gráficos exportados (figures-en, figures-pt)
│
├── src/                     # Código-fonte
│   ├── instance_gen.py      # Gerador paramétrico de instâncias JSON
│   ├── optimizer.py         # Solver MILP (Modelagem via Gurobi)
│   └── plot_gen.py          # Gerador de gráficos em português e inglês
│
├── .gitignore               # Regras de arquivos e pastas ignorados pelo Git
├── LICENSE                  # Licença MIT
├── README.md                # Documentação principal e instruções de uso
├── requirements.txt         # Bibliotecas que o projeto necessita que precisam ser instaladas previamente
└── run_pipeline.py          # Pipeline de automação (Limpeza -> Instalação de Dependências -> Geração de Instâncias -> Resolução -> Geração de Gráficos)
```

## ⚙️ Pré-requisitos e Instalação

1. Certifique-se de ter o [Python 3.8+](https://www.python.org/downloads/) instalado.
2. É estritamente necessário possuir uma licença válida do **Gurobi Optimizer**.
3. Clone o repositório:

```bash
git clone https://github.com/rob1913r/behavioral-spsp.git
cd behavioral-spsp
```

## 🚀 Como Executar

O projeto possui um orquestrador em Python (`run_pipeline.py`) projetado para automatizar o fluxo de trabalho de ponta a ponta em qualquer sistema operacional (Windows, macOS ou Linux). Basta executá-lo na raiz do projeto:

```bash
python run_pipeline.py
```

**O que o pipeline automatizado faz:**
1. Prepara e limpa os diretórios de dados dinâmicos (`data/`);
2. Instala as bibliotecas necessárias descritas em `requirements.txt`;
3. Executa `src/instance_gen.py` para criar instâncias controladas (grupos de escalabilidade, ruído, etc.);
4. Executa `src/optimizer.py`, que carrega os dados no Gurobi, processa as restrições sociotécnicas e exporta as métricas para o arquivo `data/results/metrics.csv`;
5. Executa `src/plot_gen.py` para plotar os gráficos nas versões em Português (`data/results/figures-pt/`) e Inglês (`data/results/figures-en/`).

## 📄 Licença

Distribuído sob a **licença MIT**. Veja o arquivo **LICENSE** para mais informações.
