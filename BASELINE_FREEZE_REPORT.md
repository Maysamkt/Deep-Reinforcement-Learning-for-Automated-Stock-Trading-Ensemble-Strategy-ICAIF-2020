# BASELINE_FREEZE_REPORT.md — Relatório Final de Congelamento do Baseline

---

### Status Geral do Congelamento
* **Status:** **CONGELADO E AUDITADO COM SUCESSO**
* **Data:** 15 de Agosto de 2026
* **Estágio:** Baseline Yang et al. (2020) Fechado para Início da Adaptação B3.

---

### Resumo dos Componentes Verificados

| Componente | Quantidade / Status | Observação |
| :--- | :---: | :--- |
| **Checkpoints Treinados** | 54 modelos `.zip` | 18 trimestres $\times$ 3 algoritmos (PPO, A2C, DDPG) |
| **Bases de Dados** | `done_data.csv`, `dow_30`, `^DJI` | Todas presentes e validadas |
| **Scripts de Avaliação** | 6 scripts em `backtest_eval/` | Executados e validados |
| **Tabela I (Seleção Agentes)** | 18 linhas auditadas | CSV salvo em `baseline_freeze/results/` |
| **Tabela II (Performance)** | 6 estratégias | CSV salvo em `baseline_freeze/results/` |
| **Gráficos em Alta Resolução** | 4 imagens PNG | Salvas em `baseline_freeze/plots/` |
| **Hashes SHA-256** | 78 arquivos críticos | Gravados em `baseline_freeze/manifests/files_sha256.txt` |

---

### Métricas Finais do Baseline Congelado (04/01/2016 - 08/05/2020)

* **Ensemble (Ours):** Retorno: **64.29%** | Sharpe: **1.44** | Volatilidade: **8.20%** | Max Drawdown: **-9.15%**
* **PPO Isolado:** Retorno: **53.15%** | Sharpe: **1.32** | Volatilidade: **7.69%** | Max Drawdown: **-7.04%**
* **A2C Isolado:** Retorno: **49.50%** | Sharpe: **1.22** | Volatilidade: **7.83%** | Max Drawdown: **-5.42%**
* **DDPG Isolado:** Retorno: **74.52%** | Sharpe: **1.63** | Volatilidade: **8.08%** | Max Drawdown: **-6.60%**
* **Min-Variance:** Retorno: **28.61%** | Sharpe: **0.40** | Volatilidade: **19.03%** | Max Drawdown: **-35.33%**
* **DJIA (^DJI):** Retorno: **41.88%** | Sharpe: **0.50** | Volatilidade: **20.06%** | Max Drawdown: **-37.09%**

---

### Inconsistências e Riscos Remanescentes
* **Inexistência de Inconsistências Críticas:** Todos os 54 modelos estão operacionais e reprodutíveis.
* **Reserva de Modificação:** Nenhuma alteração foi realizada nas funções originais de recompensa, hiperparâmetros ou código de ambiente.

---

### Confirmação Final de Integridade
Confirmamos que o baseline experimental **NÃO FOI ALTERADO** e permanece intocado em estado de somente leitura.
