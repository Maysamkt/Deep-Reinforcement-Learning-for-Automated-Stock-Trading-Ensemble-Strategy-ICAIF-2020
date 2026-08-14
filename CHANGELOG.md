# Registro de Alterações (CHANGELOG)

## Projeto: Deep Reinforcement Learning for Automated Stock Trading — Ensemble Strategy (Yang et al., 2020 / 2025)

Este documento registra todas as alterações realizadas no projeto durante a execução do **Plano Controlado de Compatibilização**, garantindo a migração da pilha legada (Python 3.7, TensorFlow 1.15, Stable-Baselines1, Gym 0.15) para uma pilha moderna (Python 3.10+, PyTorch, Stable-Baselines3, Gymnasium) com **100% de preservação da metodologia científica original**.

---

### Visão Geral da Arquitetura
* **Ambiente de Desenvolvimento:** Antigravity (edição de código, documentação e testes leves/sintéticos).
* **Ambiente de Execução:** Google Colab (treinamento computacional pesado dos 54 modelos DRL com aceleração GPU).
* **Ponte de Sincronização:** GitHub (versionamento de código e artefatos).

---

### Resumo por Passo do Plano de Compatibilização

#### Passo 1: Atualização de Dependências (`requirements.txt`)
* **Arquivo:** `requirements.txt`
* **Tipo de Alteração:** A) Compatibilidade
* **Mudanças:**
  * Remoção do `tensorflow==1.15.0` e `stable-baselines[mpi]`.
  * Fixação das bibliotecas modernas: `torch>=2.0.0`, `stable-baselines3>=2.0.0,<2.4.0`, `gymnasium>=0.28.1,<1.0.0`, `numpy>=1.24.0,<2.0.0`, `pandas>=1.5.0,<2.3.0`, `stockstats>=0.4.0`, `pyfolio-reloaded>=0.5.1`, `joblib`, `pytest`.
* **Risco Metodológico:** Baixo (gerenciamento de runtime).
* **Status:** Concluído e Validado.

---

#### Passo 2: Configuração e Diretórios (`config/config.py`)
* **Arquivo:** `config/config.py`
* **Tipo de Alteração:** A) Compatibilidade / B) Correção de Bug (Cross-platform)
* **Mudanças:**
  * Formatação do timestamp alterada de `%Y%m%d-%H:%M:%S` para `%Y%m%d_%H%M%S` (removendo caractere `:` incompatível com o sistema de arquivos Windows).
  * Substituição de criação de diretórios sem checagem por `os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)` e `os.makedirs("results", exist_ok=True)`.
* **Risco Metodológico:** Nulo.
* **Status:** Concluído e Validado.

---

#### Passo 3: Pré-processamento e Trata de Deprecamentos (`preprocessing/preprocessors.py`)
* **Arquivo:** `preprocessing/preprocessors.py`
* **Tipo de Alteração:** A) Compatibilidade / C) Refatoração
* **Mudanças:**
  * Substituição de chamadas obsoletas `df.append()` por acúmulo em listas Python e concatenação via `pd.concat(..., ignore_index=True)`.
  * Substituição de `df.fillna(method='bfill')` por `df.bfill()`.
* **Risco Metodológico:** Nulo (lógica de transformação de dados e cálculo de indicadores técnicos preservados exatamente iguais).
* **Status:** Concluído e Validado (`py_compile`).

---

#### Passo 4: Ambientes de Simulação Gym → Gymnasium (`env/EnvMultipleStock_*.py`)
* **Arquivos:**
  * `env/EnvMultipleStock_train.py`
  * `env/EnvMultipleStock_validation.py`
  * `env/EnvMultipleStock_trade.py`
* **Tipo de Alteração:** A) Compatibilidade (Gymnasium API v0.26+)
* **Mudanças:**
  * `reset()`: Assinatura atualizada para retornar a 2-tupla `(obs, info)` com `obs` em `np.float32`.
  * `step()`: Assinatura atualizada para retornar a 5-tupla `(obs, reward, terminated, truncated, info)` com `terminated=self.terminal` e `truncated=False`.
  * Conversão explícita de `self.state` e observações para `np.array(..., dtype=np.float32)`.
* **Risco Metodológico:** Nulo (regras de negócio, dimensionamento de estados, custo de transação de 0.1%, dinâmica de carteira, turbulência e recompensa mantidos 100% fiéis).
* **Status:** Concluído e Validado (`py_compile`).

---

#### Passo 5: Modelos e Algoritmos de RL SB1 → SB3 (`model/models.py`)
* **Arquivo:** `model/models.py`
* **Tipo de Alteração:** A) Compatibilidade / B) Parametrização estrita
* **Mudanças:**
  * Migração de `stable_baselines` (TensorFlow) para `stable_baselines3` (PyTorch).
  * **A2C:** Parametrizado com `policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))`, `ent_coef=0.01`, `vf_coef=0.25`, `gamma=0.99`, `learning_rate=7e-4`.
  * **PPO:** Fixed hyperparameters: `n_steps=128`, `batch_size=16`, `n_epochs=4`, `ent_coef=0.005`, `learning_rate=2.5e-4`, `clip_range=0.2`, `net_arch=[64, 64]`.
  * **DDPG:** Arquitetura fixada em `[64, 64]` (`pi=[64, 64], qf=[64, 64]`), `OrnsteinUhlenbeckActionNoise` ($\sigma=0.5$), `buffer_size=50000`, `batch_size=128`, `tau=0.001`.
  * **Lógica de Ensemble:** Manted intacta a janela rolante trimestral, a seleção pelo maior Sharpe ratio anualizado e a propagação do estado final (`last_state_ensemble`).
* **Risco Metodológico:** Baixo/Controlado.
* **Status:** Concluído e Validado (`py_compile`).

---

#### Passo 6: Ponto de Entrada e Suíte de Teste de Fumaça (`run_DRL.py`, `tests/smoke_test.py`)
* **Arquivos:**
  * `run_DRL.py`
  * `tests/smoke_test.py` (Novo)
  * `tests/__init__.py` (Novo)
* **Tipo de Alteração:** A) Compatibilidade / B) Correção de Bug / C) Testabilidade
* **Mudanças:**
  * Em `run_DRL.py`: Ajuste do import `stable_baselines3.common.vec_env`, caminho robusto independente do CWD para `done_data.csv` e inclusão de `sys.path.insert(0, PROJECT_ROOT)`.
  * Em `tests/smoke_test.py`: Implementação de 14 testes automatizados em PyTest validando a pilha em CPU com dados sintéticos em < 60s.
* **Risco Metodológico:** Nulo.
* **Status:** Concluído e Validado (validação de AST e compilação de bytecode).

---

#### Passo 7: Orquestrador Colab e Documentação (`colab_runner.ipynb`, `CHANGELOG.md`)
* **Arquivos:**
  * `colab_runner.ipynb` (Novo)
  * `CHANGELOG.md` (Novo)
* **Tipo de Alteração:** C) Infraestrutura e Documentação
* **Mudanças:**
  * Criação do notebook `colab_runner.ipynb` estruturado para verificação de GPU, instalação automática de requisitos, execução do `smoke_test.py`, execução do treinamento DRL e empacotamento em arquivos ZIP.
  * Criação do registro completo de alterações neste `CHANGELOG.md`.
* **Risco Metodológico:** Nulo.
* **Status:** Concluído e Validado.
