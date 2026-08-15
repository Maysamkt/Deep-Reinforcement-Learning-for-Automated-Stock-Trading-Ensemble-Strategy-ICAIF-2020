# BASELINE_CHANGELOG.md — Registro de Congelamento do Baseline

Este registro documenta todas as etapas e arquivos criados exclusivamente para a formalização do congelamento do baseline experimental (Yang et al., 2020).

---

### [Unreleased Baseline Freeze] — 2026-08-15

#### Alterações Realizadas
* **NENHUM CÓDIGO EXPERIMENTAL OU CHECKPOINT FOI ALTERADO OU RETREINADO.**
* Criada a estrutura de diretórios `baseline_freeze/` para armazenar o instantâneo de documentação, manifests e resultados.
* Gerado o manifesto de integridade com hashes SHA-256 de todos os 54 checkpoints e 24 arquivos de código/dados críticos (`baseline_freeze/manifests/files_sha256.txt`).
* Criados os scripts isolados de avaliação em `backtest_eval/` para reprodução do período do artigo (04/01/2016 a 08/05/2020), geração das Tabelas I e II e exportação dos gráficos comparativos em alta resolução.
* Criado o documento formal `BASELINE_FREEZE.md` detalhando as 24 premissas metodológicas.
* Criado o relatório final de congelamento `BASELINE_FREEZE_REPORT.md`.

#### Preservação
* O código de treinamento original (`run_DRL.py`, `model/models.py`, `preprocessing/preprocessors.py`, `env/*.py`) foi mantido intacto.
* A base pré-processada `done_data.csv` permanece inalterada.
* Os 54 modelos compactados em `trained_models/20260815_014853/` foram 100% preservados em estado de somente leitura (*read-only*).
