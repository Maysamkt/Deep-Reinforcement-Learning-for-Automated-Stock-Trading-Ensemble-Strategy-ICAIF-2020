import pandas as pd
import numpy as np
import os
import glob

PROJECT_ROOT = r"d:\ProjetoOriginal\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DONE_DATA_PATH = os.path.join(PROJECT_ROOT, "done_data.csv")

df_done = pd.read_csv(DONE_DATA_PATH)
unique_trade_dates = sorted(df_done[(df_done.datadate > 20151001) & (df_done.datadate <= 20200707)].datadate.unique())

trade_indices = [126, 189, 252, 315, 378, 441, 504, 567, 630, 693, 756, 819, 882, 945, 1008, 1071, 1134, 1197]
paper_cutoff_date = pd.to_datetime('2020-05-08')

def compile_isolated_csvs(algo_name):
    dfs = []
    for idx in trade_indices:
        fname = os.path.join(RESULTS_DIR, f"account_value_trade_{algo_name}_isolated_{idx}.csv")
        if os.path.exists(fname):
            df_w = pd.read_csv(fname)
            val_col = '0' if '0' in df_w.columns else df_w.columns[-1]
            dfs.append(df_w[[val_col]].rename(columns={val_col: 'account_value'}))
        else:
            print(f"Missing {fname}")
            
    df_full = pd.concat(dfs, ignore_index=True)
    trade_dates_seq = unique_trade_dates[63: 63 + len(df_full)]
    df_full['datadate'] = trade_dates_seq
    df_full['Date'] = pd.to_datetime(df_full['datadate'].astype(str), format='%Y%m%d')
    
    df_paper = df_full[df_full['Date'] <= paper_cutoff_date].copy().reset_index(drop=True)
    
    init_v = df_paper['account_value'].iloc[0]
    final_v = df_paper['account_value'].iloc[-1]
    cum_ret = (final_v - init_v) / init_v
    d_ret = df_paper['account_value'].pct_change().dropna()
    ann_ret = (1 + cum_ret)**(252 / len(df_paper)) - 1
    ann_vol = d_ret.std() * (252**0.5)
    sharpe = (252**0.5) * d_ret.mean() / d_ret.std()
    
    cum_m = np.maximum.accumulate(df_paper['account_value'].values)
    mdd = ((df_paper['account_value'].values - cum_m) / cum_m).min()
    
    print(f"\n============================================")
    print(f"=== {algo_name.upper()} ISOLADO (04/01/2016 - 08/05/2020) ===")
    print(f"============================================")
    print(f"Patrimônio Inicial: ${init_v:,.2f}")
    print(f"Patrimônio Final:   ${final_v:,.2f}")
    print(f"Retorno Acumulado:  {cum_ret:.2%}")
    print(f"Retorno Anualizado: {ann_ret:.2%}")
    print(f"Volatilidade Anual: {ann_vol:.2%}")
    print(f"Sharpe Ratio (252): {sharpe:.2f}")
    print(f"Maximum Drawdown:   {mdd:.2%}")
    
    out_path = os.path.join(PROJECT_ROOT, "backtest_eval", f"{algo_name.lower()}_daily_account_value_paper.csv")
    df_paper.to_csv(out_path, index=False)
    return df_paper

if __name__ == "__main__":
    compile_isolated_csvs("PPO")
    compile_isolated_csvs("A2C")
    compile_isolated_csvs("DDPG")
