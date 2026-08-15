import pandas as pd
import numpy as np
import os

PROJECT_ROOT = r"d:\ProjetoOriginal\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DONE_DATA_PATH = os.path.join(PROJECT_ROOT, "done_data.csv")

df_done = pd.read_csv(DONE_DATA_PATH)
unique_trade_dates = sorted(df_done[(df_done.datadate > 20151001) & (df_done.datadate <= 20200707)].datadate.unique())

rebalance_window = 63
validation_window = 63

trade_indices = [126, 189, 252, 315, 378, 441, 504, 567, 630, 693, 756, 819, 882, 945, 1008, 1071, 1134, 1197]

# Table I from Yang et al. (2020) paper for comparison:
paper_table_1 = {
    "2016/01-2016/03": {"PPO": 0.06, "A2C": 0.03, "DDPG": 0.05, "Picked": "PPO"},
    "2016/04-2016/06": {"PPO": 0.31, "A2C": 0.53, "DDPG": 0.61, "Picked": "DDPG"},
    "2016/07-2016/09": {"PPO": -0.02, "A2C": 0.01, "DDPG": 0.05, "Picked": "DDPG"},
    "2016/10-2016/12": {"PPO": 0.11, "A2C": 0.01, "DDPG": 0.09, "Picked": "PPO"},
    "2017/01-2017/03": {"PPO": 0.53, "A2C": 0.44, "DDPG": 0.13, "Picked": "PPO"},
    "2017/04-2017/06": {"PPO": 0.29, "A2C": 0.44, "DDPG": 0.12, "Picked": "A2C"},
    "2017/07-2017/09": {"PPO": 0.40, "A2C": 0.32, "DDPG": 0.15, "Picked": "PPO"},
    "2017/10-2017/12": {"PPO": -0.05, "A2C": -0.04, "DDPG": 0.12, "Picked": "DDPG"},
    "2018/01-2018/03": {"PPO": 0.71, "A2C": 0.63, "DDPG": 0.62, "Picked": "PPO"},
    "2018/04-2018/06": {"PPO": -0.08, "A2C": -0.02, "DDPG": -0.01, "Picked": "DDPG"},
    "2018/07-2018/09": {"PPO": -0.17, "A2C": 0.21, "DDPG": -0.03, "Picked": "A2C"},
    "2018/10-2018/12": {"PPO": 0.30, "A2C": 0.48, "DDPG": 0.39, "Picked": "A2C"},
    "2019/01-2019/03": {"PPO": -0.26, "A2C": -0.25, "DDPG": -0.18, "Picked": "DDPG"},
    "2019/04-2019/06": {"PPO": 0.38, "A2C": 0.29, "DDPG": 0.25, "Picked": "PPO"},
    "2019/07-2019/09": {"PPO": 0.53, "A2C": 0.47, "DDPG": 0.52, "Picked": "PPO"},
    "2019/10-2019/12": {"PPO": -0.22, "A2C": 0.11, "DDPG": -0.22, "Picked": "A2C"},
    "2020/01-2020/03": {"PPO": -0.36, "A2C": -0.13, "DDPG": -0.22, "Picked": "A2C"},
    "2020/04-2020/05": {"PPO": -0.42, "A2C": -0.15, "DDPG": -0.58, "Picked": "A2C"},
}

rows_t1 = []
print("=== FASE 4: TABELA I - SELEÇÃO TRIMESTRAL DE AGENTES ===")

quarter_names = list(paper_table_1.keys())

for idx_num, idx in enumerate(trade_indices):
    val_file = os.path.join(RESULTS_DIR, f"account_value_validation_{idx}.csv")
    quarter_str = quarter_names[idx_num] if idx_num < len(quarter_names) else f"Quarter_{idx}"
    
    val_sharpe_calc = np.nan
    if os.path.exists(val_file):
        df_v = pd.read_csv(val_file, index_col=0)
        df_v.columns = ['account_value']
        d_ret = df_v['account_value'].pct_change().dropna()
        # get_validation_sharpe formula in models.py uses (4**0.5) * mean / std
        val_sharpe_calc = (4 ** 0.5) * d_ret.mean() / d_ret.std()
    
    paper_info = paper_table_1.get(quarter_str, {})
    picked_paper = paper_info.get("Picked", "N/A")
    
    rows_t1.append({
        "Quarter Index": idx,
        "Trading Quarter": quarter_str,
        "Validation Sharpe (Final Saved)": round(val_sharpe_calc, 4) if not np.isnan(val_sharpe_calc) else "N/A",
        "Paper PPO Sharpe": paper_info.get("PPO", "N/A"),
        "Paper A2C Sharpe": paper_info.get("A2C", "N/A"),
        "Paper DDPG Sharpe": paper_info.get("DDPG", "N/A"),
        "Paper Picked Agent": picked_paper
    })

df_t1 = pd.DataFrame(rows_t1)
print(df_t1.to_string(index=False))

df_t1.to_csv(os.path.join(PROJECT_ROOT, "backtest_eval", "tabela_1_selecao_agentes.csv"), index=False)
