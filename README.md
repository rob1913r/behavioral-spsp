# Behavioral SPSP: Otimização de Alocação de Tarefas em Projetos de Software

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Gurobi Optimizer](https://img.shields.io/badge/solver-Gurobi-red)
![License](https://img.shields.io/badge/license-MIT-green)

Este repositório contém o código-fonte, a documentação e a formulação matemática de um projeto de pesquisa focado na resolução do clássico **Problema de Escalonamento de Projetos de Software (SPSP)** através da Programação Linear Inteira Mista (MILP).

O grande diferencial deste modelo é o rompimento com a visão tradicional que considera apenas restrições técnicas. Aqui, o tempo de conclusão de uma tarefa é uma variável dinâmica calculada com base no **Excedente Técnico** e no **Erro Comportamental** do desenvolvedor, integrando a metodologia psicométrica Solides (Executor, Comunicador, Planejador, Analista) diretamente à função objetivo.

**Principais Contribuições do Modelo:**
- 📉 Simulação matemática e comprovação tática da empírica **Lei de Brooks**.
- 🧩 Comprovação do *Trade-off* Técnico-Comportamental (um desenvolvedor Júnior alinhado pode superar um Sênior sob alto ruído cognitivo).
- 🔥 Geração de Mapas de Calor para planejamento de capacidade tática de equipes ágeis.

---

## 📁 Estrutura do Repositório

A arquitetura do projeto segue os padrões da engenharia de software para facilitar a reprodutibilidade dos experimentos:

```text
behavioral-spsp/
│
├── README.md                # Documentação principal
├── LICENSE                  # Licença MIT
├── requirements.txt         # Dependências do projeto (Python)
├── run_pipeline.py         # Pipeline de automação (Geração -> Otimização -> Gráficos)
│
├── src/                     # Código-fonte
│   ├── instance_gen.py      # Gerador paramétrico de instâncias JSON
│   ├── optimizer.py         # Solver MILP (Modelagem via Gurobi)
│   └── plot_gen.py          # Gerador de gráficos e análises visuais
│
├── data/                    # Dados isolados do código
│   ├── instances/           # Subpastas com instâncias sintéticas geradas
│   └── results/             # Métricas e resultados extraídos do otimizador
│
└── docs/                    # Documentação e Artigo Científico
    └── paper/               # Código-fonte LaTeX do artigo e bibliografia
        └── figures/         # Gráficos renderizados consumidos pelo PDF

## ⚙️ Pré-requisitos e Instalação

1. Certifique-se de ter o [Python 3.8+](https://www.python.org/downloads/) instalado.
2. É estritamente necessário possuir uma licença válida do **Gurobi Optimizer**.
3. Clone o repositório e instale as dependências:

```bash
git clone https://github.com/rob1913r/behavioral-spsp.git
cd behavioral-spsp
pip install -r requirements.txt
```

## 🚀 Como Executar

O projeto possui um orquestrador em Python (`run_pipeline.py`) projetado para automatizar o fluxo de trabalho de ponta a ponta em qualquer sistema operacional (Windows, macOS ou Linux). Basta executá-lo na raiz do projeto:

```bash
python run_pipeline.py
```

**O que o pipeline automatizado faz:**
1. Prepara e limpa os diretórios de dados dinâmicos (`data/` e `docs/paper/figures/`);
2. Executa `src/instance_gen.py` para criar instâncias controladas (grupos de escalabilidade, ruído, etc.);
3. Executa `src/optimizer.py`, que carrega os dados no Gurobi, processa as restrições sociotécnicas e exporta as métricas para um arquivo unificado em `data/results/metrics.csv`;
4. Executa `src/plot_gen.py` para analisar os resultados e plotar os gráficos.

Para compilar o PDF final da pesquisa, basta acessar a pasta `docs/paper/` e processar o arquivo `main.tex` em qualquer compilador **LaTeX** (como o *TeXworks* ou *Overleaf*).

## 📄 Licença

Distribuído sob a **licença MIT**. Veja o arquivo **LICENSE** para mais informações.
