import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

PROJECT_ROOT = r"d:\ProjetoOriginal\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020"
DONE_DATA_PATH = os.path.join(PROJECT_ROOT, "done_data.csv")
DJI_PATH = os.path.join(PROJECT_ROOT, "data", "^DJI.csv")

df_done = pd.read_csv(DONE_DATA_PATH)
unique_trade_dates = sorted(df_done[(df_done.datadate > 20151001) & (df_done.datadate <= 20200707)].datadate.unique())

# Exact paper cutoff: 2016-01-04 to 2020-05-08
# unique_trade_dates[63] is 2016-01-04
trade_dates_seq = [d for d in unique_trade_dates if 20160104 <= d <= 20200508]

print("=== FASE 3: CÁLCULO DOS BENCHMARKS (DJIA E MIN-VARIANCE) ===")
print(f"Período de avaliação: {trade_dates_seq[0]} a {trade_dates_seq[-1]} ({len(trade_dates_seq)} dias úteis)")

# --- 1. DJIA BENCHMARK ---
df_dji = pd.read_csv(DJI_PATH)
df_dji['datadate'] = pd.to_datetime(df_dji['Date']).dt.strftime('%Y%m%d').astype(int)
df_dji_sub = df_dji[(df_dji.datadate >= trade_dates_seq[0]) & (df_dji.datadate <= trade_dates_seq[-1])].copy().sort_values('datadate').reset_index(drop=True)

dji_daily_ret = df_dji_sub['Adj Close'].pct_change().fillna(0)
dji_account_val = 1000000.0 * (1 + dji_daily_ret).cumprod()

df_dji_result = pd.DataFrame({
    'datadate': df_dji_sub['datadate'],
    'Date': df_dji_sub['Date'],
    'account_value': dji_account_val
})

# --- 2. MINIMUM VARIANCE BENCHMARK ---
# We pivot done_data to get adjusted close prices per stock and date
df_prices = df_done.pivot(index='datadate', columns='tic', values='adjcp').sort_index()

# For rebalancing, we rebalance every 63 days using preceding 252 days of historical data
def get_min_variance_weights(cov_matrix):
    num_assets = cov_matrix.shape[0]
    
    def portfolio_var(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_weights = np.ones(num_assets) / num_assets
    
    res = minimize(portfolio_var, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x if res.success else init_weights

min_var_account_values = [1000000.0]
rebalance_window = 63
current_val = 1000000.0

# Align stock tickers
tickers = sorted(df_done.tic.unique())
df_prices = df_prices[tickers]

# Process day-by-day across trade_dates_seq
current_weights = None

for i in range(len(trade_dates_seq)):
    d = trade_dates_seq[i]
    
    # Rebalance every 63 days
    if i % rebalance_window == 0:
        # Preceding historical data (past 252 days up to d-1, strictly no lookahead)
        hist_dates = [dt for dt in df_prices.index if dt < d]
        if len(hist_dates) >= 63:
            lookback_dates = hist_dates[-252:]
            hist_prices = df_prices.loc[lookback_dates]
            returns_mat = hist_prices.pct_change().dropna()
            cov_mat = returns_mat.cov().values
            current_weights = get_min_variance_weights(cov_mat)
        else:
            current_weights = np.ones(len(tickers)) / len(tickers)
            
    if i > 0:
        prev_d = trade_dates_seq[i - 1]
        today_prices = df_prices.loc[d].values
        prev_prices = df_prices.loc[prev_d].values
        stock_rets = (today_prices - prev_prices) / prev_prices
        
        port_ret = np.sum(current_weights * stock_rets)
        current_val = current_val * (1.0 + port_ret)
        min_var_account_values.append(current_val)

df_minvar_result = pd.DataFrame({
    'datadate': trade_dates_seq,
    'Date': pd.to_datetime(pd.Series(trade_dates_seq).astype(str), format='%Y%m%d'),
    'account_value': min_var_account_values
})

# Save benchmark CSVs
os.makedirs(os.path.join(PROJECT_ROOT, "backtest_eval"), exist_ok=True)
df_dji_result.to_csv(os.path.join(PROJECT_ROOT, "backtest_eval", "djia_daily_account_value.csv"), index=False)
df_minvar_result.to_csv(os.path.join(PROJECT_ROOT, "backtest_eval", "minvar_daily_account_value.csv"), index=False)

# Print Summary Metrics for Benchmarks
def calc_metrics(df_res, name):
    init_v = df_res['account_value'].iloc[0]
    final_v = df_res['account_value'].iloc[-1]
    cum_ret = (final_v - init_v) / init_v
    d_ret = df_res['account_value'].pct_change().dropna()
    ann_ret = (1 + cum_ret)**(252 / len(df_res)) - 1
    ann_vol = d_ret.std() * (252**0.5)
    sharpe = (252**0.5) * d_ret.mean() / d_ret.std()
    
    cum_m = np.maximum.accumulate(df_res['account_value'].values)
    mdd = ((df_res['account_value'].values - cum_m) / cum_m).min()
    
    print(f"\n--- {name} BENCHMARK METRICS (04/01/2016 - 08/05/2020) ---")
    print(f"Patrimônio Inicial: ${init_v:,.2f}")
    print(f"Patrimônio Final:   ${final_v:,.2f}")
    print(f"Retorno Acumulado:  {cum_ret:.2%}")
    print(f"Retorno Anualizado: {ann_ret:.2%}")
    print(f"Volatilidade Anual: {ann_vol:.2%}")
    print(f"Sharpe Ratio (252): {sharpe:.2f}")
    print(f"Maximum Drawdown:   {mdd:.2%}")

calc_metrics(df_dji_result, "DJIA (^DJI)")
calc_metrics(df_minvar_result, "Min-Variance (Markowitz)")
