# BASELINE_FREEZE.md — Congelamento do Baseline Yang et al. (2020)

> **AVISO DE CONGELAMENTO:**  
> Este diretório representa o baseline experimental congelado utilizado como referência antes da adaptação da estratégia ao mercado brasileiro (B3).  
> **Status:** SOMENTE LEITURA (*READ-ONLY*). Nenhuma alteração deve ser feita nestes arquivos.

---

### 1. Identificação do Projeto
* **Nome:** Reprodução Modernizada do Baseline DRL Ensemble Strategy (Yang et al., 2020)
* **Objetivo:** Estabelecer uma linha de base (*baseline*) experimental rigorosa, auditada e reprodutível no universo Dow Jones (DJIA 30) antes da expansão para o mercado brasileiro.

### 2. Referência Científica
* **Artigo Original:** Yang, Hongyang, et al. *"Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy."* NeurIPS Workshop / ICAIF 2020 (SSRN 3690996).
* **Autores:** Hongyang Yang, Xiao-Yang Liu, Shan Zhong, Anwar Walid (Columbia University & Nokia-Bell Labs).

### 3. Framework Utilizado
* **Reimplementação Modernizada:** Python 3.12+ / PyTorch 2.x / Stable-Baselines3 (SB3) 2.9+ / Gymnasium 1.x.
* **Nota sobre a Reprodução:** Esta é uma reprodução modernizada e não uma cópia bit a bit do ambiente original do artigo. O artigo original de 2020 utilizou o framework obsoleto **OpenAI Baselines / TensorFlow 1.x**, enquanto nossa implementação utiliza a infraestrutura moderna **PyTorch + SB3 + Gymnasium**.

### 4. Versão do Python e Ambiente
* **Python:** `3.14.0a3` / `3.12.x`
* **OS:** Windows x86_64

### 5. Versões das Principais Bibliotecas
* `stable-baselines3 == 2.9.0`
* `gymnasium == 1.3.0`
* `torch == 2.9.1`
* `pandas == 2.3.3`
* `numpy == 2.4.1`
* `scipy == 1.17.0`
* `scikit-learn == 1.8.0`
* `stockstats == 0.6.8`
* `matplotlib == 3.11.1`

### 6. Arquitetura dos Agentes (PPO / A2C / DDPG)
* **A2C (Advantage Actor Critic):** Policy `MlpPolicy`, `pi=[64, 64]`, `vf=[64, 64]`, `learning_rate=7e-4`, `ent_coef=0.01`, `vf_coef=0.25`, `timesteps=30.000` por janela.
* **PPO (Proximal Policy Optimization):** Policy `MlpPolicy`, `pi=[64, 64]`, `vf=[64, 64]`, `learning_rate=2.5e-4`, `batch_size=16`, `n_epochs=4`, `clip_range=0.2`, `timesteps=100.000` por janela.
* **DDPG (Deep Deterministic Policy Gradient):** Policy `MlpPolicy`, `pi=[64, 64]`, `qf=[64, 64]`, `buffer_size=50.000`, `batch_size=128`, `tau=0.001`, com ruído de ação `OrnsteinUhlenbeckActionNoise(sigma=0.5)`, `timesteps=10.000` por janela.

### 7. Regras do Ensemble e Critério de Seleção por Sharpe
* Em cada um dos 18 trimestres (janelas rolantes de 63 dias úteis), os 3 algoritmos (A2C, PPO, DDPG) são treinados.
* Cada algoritmo é avaliado na janela de validação prévia de 63 dias úteis.
* O Sharpe de validação é calculado como: $\text{Sharpe} = \sqrt{4} \times \frac{\bar{r}_d}{\sigma_d}$.
* O algoritmo que obtém o maior Sharpe na validação é o único selecionado para realizar o trading real no trimestre seguinte.

### 8. Janelas Temporais de Treino, Validação e Trading
* **Treinamento:** Janela expansiva iniciada em `01/01/2009` até `t - 126` dias.
* **Validação:** Janela móvel de 63 dias úteis de `t - 126` a `t - 63` dias.
* **Trading Out-of-Sample:** Janela móvel de 63 dias úteis de `t - 63` a `t` (totalizando 18 trimestres de 04/01/2016 a 07/07/2020, cortados rigorosamente em 08/05/2020 para comparação de artigo).

### 9. Índice de Turbulência
* Medida de estresse de mercado baseada na distância de Mahalanobis: $\text{turbulence}_t = (y_t - \mu) \Sigma^{-1} (y_t - \mu)^T$.
* Se a média da turbulência histórica ultrapassa o quantil de 90% in-sample, a trava de risco atua, vendendo todas as posições e interrompendo novas compras até a normalização.

### 10. Custos de Transação e Liquidez
* Custo fixo de **0,1% (10 bps)** sobre o valor bruto de cada transação de compra e venda.
* Slippage de **0%** (assumindo liquidez total ao preço ajustado de fechamento `Adj Close`).

### 11. Universo e Dados
* **Universo:** 30 ações constituintes do Dow Jones Industrial Average (DJIA).
* **Fonte:** `done_data.csv` (período 2009 a 2020).

### 12. Capital Inicial
* **Capital Inicial ($b_0$):** **\$1.000.000,00** em 04/01/2016.

### 13. Resultados Finais Auditados (04/01/2016 a 08/05/2020 - 1.095 dias úteis)
* **Ensemble (Ours):** Retorno Acumulado: **64.29%** | Retorno Anualizado: **12.10%** | Volatilidade Anual: **8.20%** | Sharpe (252): **1.44** | Max Drawdown: **-9.15%**
* **PPO Isolado:** Retorno Acumulado: **53.15%** | Retorno Anualizado: **10.31%** | Volatilidade Anual: **7.69%** | Sharpe (252): **1.32** | Max Drawdown: **-7.04%**
* **A2C Isolado:** Retorno Acumulado: **49.50%** | Retorno Anualizado: **9.70%** | Volatilidade Anual: **7.83%** | Sharpe (252): **1.22** | Max Drawdown: **-5.42%**
* **DDPG Isolado:** Retorno Acumulado: **74.52%** | Retorno Anualizado: **13.67%** | Volatilidade Anual: **8.08%** | Sharpe (252): **1.63** | Max Drawdown: **-6.60%**
* **Min-Variance (Markowitz):** Retorno Acumulado: **28.61%** | Retorno Anualizado: **5.96%** | Volatilidade Anual: **19.03%** | Sharpe (252): **0.40** | Max Drawdown: **-35.33%**
* **DJIA (^DJI):** Retorno Acumulado: **41.88%** | Retorno Anualizado: **8.38%** | Volatilidade Anual: **20.06%** | Sharpe (252): **0.50** | Max Drawdown: **-37.09%**

### 14. Tabela I (Seleção dos Agentes por Trimestre)
* Ver arquivo auditado completo em `baseline_freeze/results/tabela_1_selecao_agentes.csv`.
* Agentes escolhidos na validação: DDPG (10 trimestres), PPO (5 trimestres), A2C (3 trimestres).

### 15. Tabela II (Performance Evaluation Comparison)
* Ver arquivo completo em `baseline_freeze/results/tabela_2_performance_comparison.csv`.

### 16. Diferenças Conhecidas em Relação ao Artigo Original
1. **Framework:** SB3/PyTorch em vez de OpenAI Baselines/TF 1.x.
2. **Semente Aleatória:** Ausência de semente determinística estática gera oscilação de $\pm 5\%$ nos retornos finais em relação ao paper (64,3% vs 70,4% no artigo).

### 17. Limitações Conhecidas
* Universo restrito a 30 ações americanas de altíssima liquidez (DJIA).
* Ausência de taxas de aluguel de ações, custos de retenção ou impostos locais.

### 18. Riscos Metodológicos Investigados
* **Look-Ahead Bias nos Indicadores:** Investigado e descartado. Todos os indicadores (MACD, RSI, CCI, ADX) são digitalmente 100% causais.

### 19. Localização dos Checkpoints e Dados
* **Checkpoints:** `trained_models/20260815_014853/*.zip` (54 arquivos).
* **Dados:** `done_data.csv`, `data/dow_30_2009_2020.csv`, `data/^DJI.csv`.

### 20. Localização dos Resultados e Artefatos
* **Resultados e Scripts de Avaliação:** `backtest_eval/`
* **Snapshot Congelado:** `baseline_freeze/`

### 21. Hashes SHA-256
* Registrados na íntegra em `baseline_freeze/manifests/files_sha256.txt` (78 arquivos críticos verificados).
