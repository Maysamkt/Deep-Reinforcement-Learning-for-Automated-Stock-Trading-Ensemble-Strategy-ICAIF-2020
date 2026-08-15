import pandas as pd
import numpy as np
import os

PROJECT_ROOT = r"d:\ProjetoOriginal\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DONE_DATA_PATH = os.path.join(PROJECT_ROOT, "done_data.csv")
DJI_PATH = os.path.join(PROJECT_ROOT, "data", "^DJI.csv")

# 1. Load done_data to get exact date sequence for trading
df_done = pd.read_csv(DONE_DATA_PATH)
unique_trade_dates = sorted(df_done[(df_done.datadate > 20151001) & (df_done.datadate <= 20200707)].datadate.unique())

# Rebalance windows used during run_ensemble_strategy:
# range(126, len(unique_trade_dates), 63)
trade_indices = [126, 189, 252, 315, 378, 441, 504, 567, 630, 693, 756, 819, 882, 945, 1008, 1071, 1134, 1197]

trade_dfs = []
for idx in trade_indices:
    fname = os.path.join(RESULTS_DIR, f"account_value_trade_ensemble_{idx}.csv")
    if os.path.exists(fname):
        df_w = pd.read_csv(fname)
        val_col = '0' if '0' in df_w.columns else df_w.columns[-1]
        trade_dfs.append(df_w[[val_col]].rename(columns={val_col: 'account_value'}))

ensemble_full = pd.concat(trade_dfs, ignore_index=True)

# Attach dates starting from trade_date[63] (2016-01-04)
# unique_trade_dates[63] is 2016-01-04
trade_dates_seq = unique_trade_dates[63: 63 + len(ensemble_full)]
ensemble_full['datadate'] = trade_dates_seq
ensemble_full['Date'] = pd.to_datetime(ensemble_full['datadate'].astype(str), format='%Y%m%d')

# Filter exactly up to 2020-05-08 (paper cutoff)
paper_cutoff_date = pd.to_datetime('2020-05-08')
ensemble_paper = ensemble_full[ensemble_full['Date'] <= paper_cutoff_date].copy().reset_index(drop=True)

print("=== FASE 1: AVALIAÇÃO DO ENSEMBLE NO PERÍODO EXATO DO ARTIGO (04/01/2016 a 08/05/2020) ===")
print(f"Número de dias de negociação no período do artigo: {len(ensemble_paper)}")
print(f"Data Inicial: {ensemble_paper['Date'].iloc[0].strftime('%Y-%m-%d')}")
print(f"Data Final:   {ensemble_paper['Date'].iloc[-1].strftime('%Y-%m-%d')}")

initial_val = ensemble_paper['account_value'].iloc[0]
final_val = ensemble_paper['account_value'].iloc[-1]
cum_return = (final_val - initial_val) / initial_val

daily_ret = ensemble_paper['account_value'].pct_change().dropna()
ann_return = (1 + cum_return)**(252 / len(ensemble_paper)) - 1
ann_vol = daily_ret.std() * (252**0.5)
sharpe_252 = (252**0.5) * daily_ret.mean() / daily_ret.std()

cum_max = np.maximum.accumulate(ensemble_paper['account_value'].values)
drawdowns = (ensemble_paper['account_value'].values - cum_max) / cum_max
max_dd = drawdowns.min()

print("\n--- RESULTADOS OBTIDOS PARA O ENSEMBLE (04/01/2016 - 08/05/2020) ---")
print(f"Patrimônio Inicial:  ${initial_val:,.2f}")
print(f"Patrimônio Final:    ${final_val:,.2f}")
print(f"Retorno Acumulado:   {cum_return:.2%}")
print(f"Retorno Anualizado:  {ann_return:.2%}")
print(f"Volatilidade Anual:  {ann_vol:.2%}")
print(f"Sharpe Ratio (252):  {sharpe_252:.2f}")
print(f"Maximum Drawdown:    {max_dd:.2%}")

print("\n--- VALORES PUBLICADOS POR YANG ET AL. (2020) TABELA II ---")
print("Retorno Acumulado:   70.4%")
print("Retorno Anualizado:  13.0%")
print("Volatilidade Anual:  9.7%")
print("Sharpe Ratio:        1.30")
print("Maximum Drawdown:    -9.7%")

# Save CSV of ensemble paper series
os.makedirs(os.path.join(PROJECT_ROOT, "backtest_eval"), exist_ok=True)
ensemble_paper.to_csv(os.path.join(PROJECT_ROOT, "backtest_eval", "ensemble_daily_account_value_paper.csv"), index=False)
