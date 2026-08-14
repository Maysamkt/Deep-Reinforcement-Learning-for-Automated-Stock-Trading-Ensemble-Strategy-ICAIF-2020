# common library
import sys
import os

# ---------------------------------------------------------------------------
# Resolve o diretório raiz do projeto a partir do local deste arquivo,
# garantindo que os imports funcionem independentemente do CWD.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv  # SB3 (PyTorch)

# preprocessor
from preprocessing.preprocessors import *
# config
from config import config
# model
from model.models import *


def run_model() -> None:
    """Train the ensemble strategy (A2C + PPO + DDPG)."""

    # Caminho absoluto do dataset pré-processado (independente do CWD)
    preprocessed_path = os.path.join(PROJECT_ROOT, "done_data.csv")
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[
        (data.datadate > 20151001) & (data.datadate <= 20200707)
    ].datadate.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63

    ## Ensemble Strategy
    run_ensemble_strategy(
        df=data,
        unique_trade_date=unique_trade_date,
        rebalance_window=rebalance_window,
        validation_window=validation_window,
    )


if __name__ == "__main__":
    run_model()
