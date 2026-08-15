import sys
import os

PROJECT_ROOT = r"d:\ProjetoOriginal\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import time
import gymnasium as gym
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

from preprocessing.preprocessors import data_split
from env.EnvMultipleStock_trade import StockEnvTrade

DONE_DATA_PATH = os.path.join(PROJECT_ROOT, "done_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models", "20260815_014853")

df_done = pd.read_csv(DONE_DATA_PATH)
unique_trade_dates = sorted(df_done[(df_done.datadate > 20151001) & (df_done.datadate <= 20200707)].datadate.unique())

rebalance_window = 63
validation_window = 63
trade_indices = [126, 189, 252, 315, 378, 441, 504, 567, 630, 693, 756, 819, 882, 945, 1008, 1071, 1134, 1197]

# Calculate turbulence threshold lookback exactly like run_ensemble_strategy
insample_turbulence = df_done[(df_done.datadate < 20151000) & (df_done.datadate >= 20090000)].drop_duplicates(subset=['datadate'])
insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.90)

paper_cutoff_date = pd.to_datetime('2020-05-08')

def run_isolated_backtest(algo_type, model_prefix, loader_cls):
    print(f"\n============================================")
    print(f"=== FASE 2: BACKTEST OUT-OF-SAMPLE - {algo_type} ISOLADO ===")
    print(f"============================================")
    
    last_state = []
    daily_account_values = []
    
    for idx_count, i in enumerate(trade_indices):
        initial = (idx_count == 0)
        
        # Turbulence threshold calculation
        end_date_index = df_done.index[df_done["datadate"] == unique_trade_dates[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window * 30 + 1
        historical_turbulence = df_done.iloc[start_date_index:(end_date_index + 1), :].drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)
        
        if historical_turbulence_mean > insample_turbulence_threshold:
            turbulence_threshold = insample_turbulence_threshold
        else:
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
            
        trade_data = data_split(df_done, start=unique_trade_dates[i - rebalance_window], end=unique_trade_dates[i])
        
        model_filename = f"{model_prefix}_{i}.zip"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        # Load model checkpoint
        model = loader_cls.load(model_path)
        
        env_trade = DummyVecEnv([lambda: StockEnvTrade(
            trade_data,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
            previous_state=last_state,
            model_name=f"{algo_type}_isolated",
            iteration=i
        )])
        
        obs_trade = env_trade.reset()
        
        for k in range(len(trade_data.index.unique())):
            action, _states = model.predict(obs_trade)
            obs_trade, rewards, dones, info = env_trade.step(action)
            if k == (len(trade_data.index.unique()) - 2):
                last_state = env_trade.envs[0].render()
                
        # Asset memory of this window (excluding duplicate initial state on subsequent windows)
        env_instance = env_trade.envs[0]
        window_assets = env_instance.asset_memory
        if initial:
            daily_account_values.extend(window_assets)
        else:
            daily_account_values.extend(window_assets[1:])
            
    # Attach dates
    trade_dates_seq = unique_trade_dates[63: 63 + len(daily_account_values)]
    df_res = pd.DataFrame({
        'datadate': trade_dates_seq,
        'Date': pd.to_datetime(pd.Series(trade_dates_seq).astype(str), format='%Y%m%d'),
        'account_value': daily_account_values
    })
    
    df_paper = df_res[df_res['Date'] <= paper_cutoff_date].copy().reset_index(drop=True)
    
    # Calculate metrics
    init_v = df_paper['account_value'].iloc[0]
    final_v = df_paper['account_value'].iloc[-1]
    cum_ret = (final_v - init_v) / init_v
    d_ret = df_paper['account_value'].pct_change().dropna()
    ann_ret = (1 + cum_ret)**(252 / len(df_paper)) - 1
    ann_vol = d_ret.std() * (252**0.5)
    sharpe = (252**0.5) * d_ret.mean() / d_ret.std()
    cum_m = np.maximum.accumulate(df_paper['account_value'].values)
    mdd = ((df_paper['account_value'].values - cum_m) / cum_m).min()
    
    print(f"\n--- MÉTRICAS PARA {algo_type} ISOLADO (04/01/2016 - 08/05/2020) ---")
    print(f"Patrimônio Inicial: ${init_v:,.2f}")
    print(f"Patrimônio Final:   ${final_v:,.2f}")
    print(f"Retorno Acumulado:  {cum_ret:.2%}")
    print(f"Retorno Anualizado: {ann_ret:.2%}")
    print(f"Volatilidade Anual: {ann_vol:.2%}")
    print(f"Sharpe Ratio (252): {sharpe:.2f}")
    print(f"Maximum Drawdown:   {mdd:.2%}")
    
    df_paper.to_csv(os.path.join(PROJECT_ROOT, "backtest_eval", f"{algo_type.lower()}_daily_account_value_paper.csv"), index=False)
    return df_paper

if __name__ == "__main__":
    df_ppo = run_isolated_backtest("PPO", "PPO_100k_dow", PPO)
    df_a2c = run_isolated_backtest("A2C", "A2C_30k_dow", A2C)
    df_ddpg = run_isolated_backtest("DDPG", "DDPG_10k_dow", DDPG)
