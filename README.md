# Behavioral SPSP: Otimização de Alocação de Tarefas em Projetos de Software

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Gurobi Optimizer](https://img.shields.io/badge/solver-Gurobi-red)
![Metaheuristic](https://img.shields.io/badge/metaheuristic-GVNS-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Este repositório contém o código-fonte e a formulação matemática de um projeto
de pesquisa focado na resolução do clássico **Problema de Escalonamento de
Projetos de Software (SPSP)** através da **Programação Linear Inteira Mista
(MILP)**.

O grande diferencial deste modelo é o rompimento com a visão tradicional, que
considera apenas restrições técnicas. Aqui, o tempo de conclusão de uma tarefa
é uma **variável dinâmica**, calculada a partir de fatores humanos e
sociotécnicos do desenvolvedor alocado, integrados diretamente às restrições do
modelo:

- **Excedente Técnico** — folga entre a proficiência do dev e o requisito da tarefa;
- **Erro Comportamental** — desalinhamento de perfil psicométrico (metodologia Solides/DISC);
- **Curva de Aprendizado** — ganho de eficiência ao repetir tarefas do mesmo eixo temático ou herdadas de predecessoras;
- **Fadiga por Carga Cognitiva** — penalidade não-linear por multitarefa excessiva na mesma *sprint*;
- **Overhead de Comunicação (Lei de Brooks)** — atrito que cresce com o tamanho da equipe.

Esta versão (**v2**) acrescenta uma **metaheurística GVNS** (*General Variable
Neighborhood Search*) como alternativa ao Gurobi. O solver exato não escala para
instâncias realistas (dias por instância); o GVNS entrega soluções com o **mesmo
padrão estrutural** do Gurobi em **minutos**, abrindo caminho para a aplicação
prática do modelo.

**Principais Contribuições do Modelo:**
- 📉 Simulação matemática e comprovação tática da empírica **Lei de Brooks**.
- 🧩 *Trade-off* Técnico-Comportamental (um dev Júnior alinhado pode superar um Sênior sob alto ruído cognitivo).
- 🔁 Curva de aprendizado e fadiga modeladas de forma endógena no tempo das tarefas.
- ⚡ Transição de solver exato (Gurobi) para metaheurística (GVNS) escalável, com verificação integral de factibilidade.

---

## 📁 Estrutura do Repositório

A arquitetura segue um padrão que facilita a reprodutibilidade dos experimentos.
A pasta `data/` **não acompanha o repositório no clone**: instâncias, resultados
e gráficos são gerados dinamicamente durante a execução do pipeline.

```text
behavioral-spsp/
│
├── data/                       # Gerado dinamicamente pelo pipeline (não versionado)
│   ├── instances/              # Instâncias sintéticas (*.json)
│   └── results/
│       ├── gurobi/             # Baseline exato: <Cenario>/allocations.csv + plots/
│       └── vns/                # Metaheurística: <Cenario>/(allocations.csv + logs) + plots/
│
├── src/                        # Código-fonte
│   ├── optimizer/              # Implementações dos SOLVERS
│   │   ├── gurobi/
│   │   │   └── solver.py       # Solver MILP exato (Gurobi)
│   │   └── vns/
│   │       ├── config.py       # Todos os parâmetros (modelo + busca + debug)
│   │       ├── constraints.py  # Avaliação + as 43 restrições do artigo (Eq. 6–48)
│   │       └── solver.py       # Busca GVNS (gulosa, shaking, VND, loop)
│   ├── instance_gen.py         # Gerador determinístico de instâncias JSON
│   ├── plot_gen.py             # Geração de gráficos + análise comparativa textual
│   └── validation.py           # Valida a factibilidade do VNS via LP do Gurobi
│
├── .gitignore                  # Arquivos/pastas ignorados pelo Git
├── LICENSE                     # Licença MIT
├── README.md                   # Este arquivo
├── requirements.txt            # Dependências Python
└── run_pipeline.py             # Pipeline único (escolhe Gurobi OU VNS no terminal)
```

> **Identificadores de cenário** (`Cenario_1_Maratona_Multitarefa`, etc.) são
> nomes de dados: aparecem no JSON da instância, no campo `_params.Nome` e nas
> pastas de resultados, mantendo a correspondência instância ↔ baseline.

---

## ⚙️ Pré-requisitos e Instalação

1. Tenha o [Python 3.10+](https://www.python.org/downloads/) instalado.
2. É necessária uma licença válida do **Gurobi Optimizer** (para o solver exato e
   para a validação de factibilidade do VNS).
3. Clone o repositório e instale as dependências:

```bash
git clone https://github.com/rob1913r/behavioral-spsp.git
cd behavioral-spsp
pip install -r requirements.txt
```

---

## 🚀 Como Executar

O projeto possui um orquestrador único (`run_pipeline.py`), multiplataforma
(Windows, macOS, Linux). Basta executá-lo na raiz do projeto:

```bash
python run_pipeline.py
```

Ele pergunta no terminal qual fluxo executar:

| Opção | O que faz |
|-------|-----------|
| **`[1] Gurobi`** (lento, horas) | Limpa `data/results/gurobi/`, gera as instâncias (se faltarem), resolve o MILP exato e gera os gráficos. Pede confirmação, pois sobrescreve o baseline. |
| **`[2] VNS`** (rápido, minutos) | Limpa `data/results/vns/`, resolve com a metaheurística, valida a factibilidade no Gurobi (LP) e gera gráficos de solução, convergência e comparativos vs Gurobi. |
| **`[3] Só gráficos`** | Regenera todos os gráficos a partir dos CSVs já existentes, **sem** resolver nada. |

Atalhos não-interativos: `python run_pipeline.py vns | gurobi | plots`.

> A limpeza remove **apenas** os resultados do solver escolhido — rodar o VNS
> nunca apaga o baseline do Gurobi (que leva horas para regerar). As instâncias
> só são geradas se a pasta estiver vazia.

---

## 🔬 Sobre a Metaheurística (GVNS)

- **`optimizer/vns/config.py`** concentra todos os "botões": parâmetros do modelo
  (idênticos ao Gurobi), parâmetros de busca (balanço convergência × diversidade)
  e o modo de verificação de cada restrição (ENFORCE / WARN / OFF).
- **`optimizer/vns/constraints.py`** é a única fonte de verdade do modelo: a função
  `evaluate()` reconstrói todas as variáveis do MILP, e as **43 restrições**
  (Eq. 6–48 do artigo) estão escritas explicitamente como funções padronizadas
  `r06_…` a `r48_…`, no formato `h(x)=0` / `g(x)≤0`.
- **`optimizer/vns/solver.py`** contém apenas a busca. A cada transição de
  incumbente, a solução candidata passa por uma **verificação integral das 43
  restrições**; se violar qualquer restrição em modo ENFORCE (mesmo após reparo),
  é **rejeitada** e o motivo é registrado. O incumbente é, portanto, sempre 100%
  factível.
- Cada execução do VNS grava em `data/results/vns/<Cenario>/`: `allocations.csv`,
  `convergence_log.csv`, `vnd_log.csv`, `debug_log.txt` (transparência completa da
  busca) e `constraints_report.txt` (relatório final das 43 restrições).

O histórico detalhado do desenvolvimento está em `../planning/` (`changes.md` e
`planos/plano_metaheuristica_v*.md`).

---

## 📄 Licença

Distribuído sob a **licença MIT**. Veja o arquivo **LICENSE** para mais informações.
