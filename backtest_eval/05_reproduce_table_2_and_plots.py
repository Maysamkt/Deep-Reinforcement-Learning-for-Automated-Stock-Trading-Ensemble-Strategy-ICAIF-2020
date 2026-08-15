import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = r"d:\ProjetoOriginal\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020"
EVAL_DIR = os.path.join(PROJECT_ROOT, "backtest_eval")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load 6 strategies daily account values
files = {
    'Ensemble (Ours)': 'ensemble_daily_account_value_paper.csv',
    'PPO': 'ppo_daily_account_value_paper.csv',
    'A2C': 'a2c_daily_account_value_paper.csv',
    'DDPG': 'ddpg_daily_account_value_paper.csv',
    'DJIA': 'djia_daily_account_value.csv',
    'Min-Variance': 'minvar_daily_account_value.csv'
}

data_dict = {}
for name, fname in files.items():
    df = pd.read_csv(os.path.join(EVAL_DIR, fname))
    df['Date'] = pd.to_datetime(df['Date'])
    data_dict[name] = df.sort_values('Date').reset_index(drop=True)

# Align dates on common index
df_merged = data_dict['Ensemble (Ours)'][['Date', 'account_value']].rename(columns={'account_value': 'Ensemble (Ours)'})
for name in ['PPO', 'A2C', 'DDPG', 'DJIA', 'Min-Variance']:
    df_temp = data_dict[name][['Date', 'account_value']].rename(columns={'account_value': name})
    df_merged = pd.merge(df_merged, df_temp, on='Date', how='inner')

# Calculate Table II Metrics
table_2_rows = []

for col in ['Ensemble (Ours)', 'PPO', 'A2C', 'DDPG', 'Min-Variance', 'DJIA']:
    series = df_merged[col]
    init_v = series.iloc[0]
    final_v = series.iloc[-1]
    cum_ret = (final_v - init_v) / init_v
    
    d_ret = series.pct_change().dropna()
    ann_ret = (1 + cum_ret)**(252 / len(series)) - 1
    ann_vol = d_ret.std() * (252**0.5)
    sharpe = (252**0.5) * d_ret.mean() / d_ret.std()
    
    cum_m = np.maximum.accumulate(series.values)
    mdd = ((series.values - cum_m) / cum_m).min()
    
    table_2_rows.append({
        'Strategy': col,
        'Cumulative Return': f"{cum_ret * 100:.1f}%",
        'Annual Return': f"{ann_ret * 100:.1f}%",
        'Annual Volatility': f"{ann_vol * 100:.1f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{mdd * 100:.1f}%"
    })

df_table_2 = pd.DataFrame(table_2_rows)
print("=== FASE 5: TABELA II - PERFORMANCE EVALUATION COMPARISON (04/01/2016 - 08/05/2020) ===")
print(df_table_2.to_string(index=False))
df_table_2.to_csv(os.path.join(EVAL_DIR, "tabela_2_performance_comparison.csv"), index=False)

# --- FASE 6: GRÁFICOS EM ALTA RESOLUÇÃO ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

colors = {
    'PPO': '#1f77b4',
    'Ensemble (Ours)': '#ff7f0e',
    'A2C': '#2ca02c',
    'DDPG': '#d62728',
    'DJIA': '#17becf',
    'Min-Variance': '#9467bd'
}

# 1. FIGURA 5A: Comparative Portfolio Value ($)
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
for col in ['PPO', 'Ensemble (Ours)', 'A2C', 'DDPG', 'DJIA', 'Min-Variance']:
    ax.plot(df_merged['Date'], df_merged[col], label=col, color=colors[col], linewidth=1.8 if 'Ensemble' in col else 1.2)

ax.set_title("Portfolio Value Over Time ($1.0M Initial, 2016/01/04 - 2020/05/08)", fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel("Portfolio Value ($)", fontsize=12)
ax.yaxis.set_major_formatter('${x:,.0f}')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig5_comparative_portfolio_value.png"), dpi=300)
plt.close()

# 2. FIGURA 5B: Comparative Cumulative Return (%)
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
for col in ['PPO', 'Ensemble (Ours)', 'A2C', 'DDPG', 'DJIA', 'Min-Variance']:
    cum_ret_series = (df_merged[col] - df_merged[col].iloc[0]) / df_merged[col].iloc[0]
    ax.plot(df_merged['Date'], cum_ret_series, label=col, color=colors[col], linewidth=2.0 if 'Ensemble' in col else 1.2)

ax.set_title("Cumulative Return Curves with Transaction Cost (2016/01/04 - 2020/05/08)", fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel("Cumulative Return", fontsize=12)
ax.yaxis.set_major_formatter('{x:.0%}')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig5_comparative_cumulative_return.png"), dpi=300)
plt.close()

# 3. FIGURA 6: Market Crash Focus (Q1 2020) with Turbulence Overlay
df_crash = df_merged[(df_merged['Date'] >= '2020-01-02') & (df_merged['Date'] <= '2020-05-08')].copy().reset_index(drop=True)

# Fetch turbulence data for Q1 2020 from done_data.csv
done_data = pd.read_csv(os.path.join(PROJECT_ROOT, "done_data.csv"))
done_turb = done_data.drop_duplicates('datadate')[['datadate', 'turbulence']].copy()
done_turb['Date'] = pd.to_datetime(done_turb['datadate'].astype(str), format='%Y%m%d')

df_crash = pd.merge(df_crash, done_turb[['Date', 'turbulence']], on='Date', how='left')

fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
ax2 = ax1.twinx()

for col in ['PPO', 'Ensemble (Ours)', 'A2C', 'DDPG', 'DJIA', 'Min-Variance']:
    cum_ret_series = (df_crash[col] - df_merged[col].iloc[0]) / df_merged[col].iloc[0]
    ax1.plot(df_crash['Date'], cum_ret_series, label=col, color=colors[col], linewidth=2.0 if 'Ensemble' in col else 1.2)

ax2.plot(df_crash['Date'], df_crash['turbulence'], label='Turbulence Index', color='red', linestyle='--', linewidth=1.5, alpha=0.7)

ax1.set_title("Performance During Stock Market Crash (Q1 2020)", fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("Cumulative Return (Base 2016)", fontsize=12)
ax2.set_ylabel("Turbulence Index", fontsize=12, color='red')
ax1.yaxis.set_major_formatter('{x:.0%}')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig6_market_crash_march_2020.png"), dpi=300)
plt.close()

# 4. DRAWDOWN CHART
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
for col in ['Ensemble (Ours)', 'PPO', 'A2C', 'DDPG', 'DJIA', 'Min-Variance']:
    cum_m = np.maximum.accumulate(df_merged[col].values)
    dd_series = (df_merged[col].values - cum_m) / cum_m
    ax.plot(df_merged['Date'], dd_series, label=col, color=colors[col], linewidth=1.8 if 'Ensemble' in col else 1.0)

ax.set_title("Drawdown Analysis Over Time (2016/01/04 - 2020/05/08)", fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel("Drawdown", fontsize=12)
ax.yaxis.set_major_formatter('{x:.0%}')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig_drawdowns.png"), dpi=300)
plt.close()

print(f"\nTodos os 4 gráficos em alta resolução foram salvos em: {PLOTS_DIR}")
