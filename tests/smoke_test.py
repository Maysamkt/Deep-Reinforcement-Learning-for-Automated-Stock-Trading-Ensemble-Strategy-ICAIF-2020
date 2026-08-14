"""
Smoke Test — Passo 6 do Plano de Compatibilização Yang et al. (2025)
====================================================================
Objetivo: validar a integração entre as camadas do projeto **sem dados externos**
e em tempo inferior a 60 s em CPU (Antigravity / CI).

Testes cobertos
---------------
T01  Imports — todos os módulos do projeto importam sem erro.
T02  Config   — TRAINED_MODEL_DIR é criado; timestamp sem ':'.
T03  Preprocessors — as funções data_split e add_turbulence importam.
T04  Envs (Gymnasium) — reset/step retornam assinaturas 2/5-tupla.
T05  Modelos (SB3) — A2C, PPO e DDPG treinam 10 passos sem erro.
T06  VecEnv reset — DummyVecEnv retorna obs 2D.
T07  run_DRL import — run_DRL.py importa sem lançar run_model().

Executar com:
    pytest tests/smoke_test.py -v          # no diretório raiz do projeto
    python -m pytest tests/smoke_test.py   # alternativa
"""

import sys
import os
import time

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Garante que o diretório raiz do projeto está no sys.path,
# independentemente de onde o pytest é chamado.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Fixture: dataset mínimo sintético (30 stocks, 5 datas de negociação)
# Replica o schema exato do done_data.csv para que os envs instanciem.
# ---------------------------------------------------------------------------
STOCK_DIM = 30
N_DATES = 5


@pytest.fixture(scope="session")
def synthetic_df():
    """DataFrame com o schema do done_data.csv, mas 5 datas e 30 ativos."""
    rng = np.random.default_rng(42)
    dates = [20200101 + i for i in range(N_DATES)]
    rows = []
    for d in dates:
        for t in range(STOCK_DIM):
            rows.append(
                {
                    "datadate": d,
                    "tic": f"STK{t:02d}",
                    "adjcp": float(rng.uniform(10, 200)),
                    "open": float(rng.uniform(10, 200)),
                    "high": float(rng.uniform(10, 200)),
                    "low": float(rng.uniform(10, 200)),
                    "volume": float(rng.uniform(1e5, 1e7)),
                    "macd": float(rng.uniform(-5, 5)),
                    "rsi": float(rng.uniform(20, 80)),
                    "cci": float(rng.uniform(-100, 100)),
                    "adx": float(rng.uniform(10, 50)),
                    "turbulence": float(rng.uniform(0, 50)),
                }
            )
    df = pd.DataFrame(rows)
    # Indexado por data (como data_split retorna)
    df = df.sort_values(["datadate", "tic"]).reset_index(drop=True)
    df.index = df["datadate"]
    return df


# ===========================================================================
# T01 — Imports
# ===========================================================================
def test_T01_imports_config():
    from config import config  # noqa: F401
    assert hasattr(config, "TRAINED_MODEL_DIR")


def test_T01_imports_preprocessors():
    from preprocessing.preprocessors import data_split, add_turbulence  # noqa: F401
    assert callable(data_split)
    assert callable(add_turbulence)


def test_T01_imports_envs():
    from env.EnvMultipleStock_train import StockEnvTrain  # noqa: F401
    from env.EnvMultipleStock_validation import StockEnvValidation  # noqa: F401
    from env.EnvMultipleStock_trade import StockEnvTrade  # noqa: F401


def test_T01_imports_models():
    from model.models import train_A2C, train_PPO, train_DDPG  # noqa: F401
    assert callable(train_A2C)
    assert callable(train_PPO)
    assert callable(train_DDPG)


def test_T01_imports_sb3():
    from stable_baselines3 import A2C, PPO, DDPG  # noqa: F401
    from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: F401
    from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise  # noqa: F401


def test_T01_imports_gymnasium():
    import gymnasium as gym  # noqa: F401
    assert gym.__version__


# ===========================================================================
# T02 — Config
# ===========================================================================
def test_T02_config_model_dir_exists():
    from config import config
    assert os.path.isdir(config.TRAINED_MODEL_DIR), (
        f"TRAINED_MODEL_DIR não foi criado: {config.TRAINED_MODEL_DIR}"
    )


def test_T02_config_timestamp_no_colon():
    from config import config
    # Windows não aceita ':' em nomes de diretório
    assert ":" not in config.TRAINED_MODEL_DIR, (
        f"Timestamp contém ':': {config.TRAINED_MODEL_DIR}"
    )


def test_T02_results_dir_exists():
    assert os.path.isdir(os.path.join(PROJECT_ROOT, "results")), (
        "Diretório 'results' não foi criado pelo config.py"
    )


# ===========================================================================
# T03 — Preprocessors
# ===========================================================================
def test_T03_data_split(synthetic_df):
    from preprocessing.preprocessors import data_split
    start = 20200101
    end = 20200103
    split = data_split(synthetic_df, start=start, end=end)
    assert len(split) > 0, "data_split retornou DataFrame vazio"
    assert all(split["datadate"] >= start)
    assert all(split["datadate"] < end)


# ===========================================================================
# T04 — Envs (Gymnasium API)
# ===========================================================================
def _make_train_env(synthetic_df):
    """Retorna um StockEnvTrain com dados sintéticos."""
    from preprocessing.preprocessors import data_split
    from env.EnvMultipleStock_train import StockEnvTrain
    # Usa todas as datas disponíveis
    df = data_split(synthetic_df, start=20200101, end=20200999)
    return StockEnvTrain(df)


def test_T04_train_env_reset(synthetic_df):
    env = _make_train_env(synthetic_df)
    result = env.reset()
    assert isinstance(result, tuple), "reset() deve retornar uma tupla (obs, info)"
    assert len(result) == 2, f"reset() deve retornar 2-tupla; retornou {len(result)}"
    obs, info = result
    assert isinstance(obs, np.ndarray), "obs deve ser np.ndarray"
    assert obs.dtype == np.float32, f"obs.dtype deve ser float32; é {obs.dtype}"
    assert obs.shape == (181,), f"obs.shape deve ser (181,); é {obs.shape}"
    assert isinstance(info, dict)


def test_T04_train_env_step(synthetic_df):
    env = _make_train_env(synthetic_df)
    env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert isinstance(result, tuple), "step() deve retornar uma tupla"
    assert len(result) == 5, f"step() deve retornar 5-tupla; retornou {len(result)}"
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_T04_train_env_action_space(synthetic_df):
    env = _make_train_env(synthetic_df)
    assert env.action_space.shape == (STOCK_DIM,)


def test_T04_train_env_obs_space(synthetic_df):
    env = _make_train_env(synthetic_df)
    assert env.observation_space.shape == (181,)


# ===========================================================================
# T05 — Modelos SB3 (treinamento mínimo: 10 steps, CPU, < 60 s)
# ===========================================================================
def _make_vec_env(synthetic_df):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from preprocessing.preprocessors import data_split
    from env.EnvMultipleStock_train import StockEnvTrain
    df = data_split(synthetic_df, start=20200101, end=20200999)
    return DummyVecEnv([lambda: StockEnvTrain(df)])


def test_T05_a2c_trains_10_steps(synthetic_df):
    from stable_baselines3 import A2C
    env = _make_vec_env(synthetic_df)
    model = A2C(
        "MlpPolicy",
        env,
        verbose=0,
        ent_coef=0.01,
        vf_coef=0.25,
        gamma=0.99,
        learning_rate=7e-4,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    )
    t0 = time.time()
    model.learn(total_timesteps=10)
    elapsed = time.time() - t0
    assert elapsed < 60, f"A2C demorou {elapsed:.1f}s (limite: 60s)"


def test_T05_ppo_trains_10_steps(synthetic_df):
    from stable_baselines3 import PPO
    env = _make_vec_env(synthetic_df)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        ent_coef=0.005,
        n_steps=128,
        batch_size=16,
        n_epochs=4,
        learning_rate=2.5e-4,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    )
    t0 = time.time()
    # PPO coleta n_steps antes de atualizar; 10 steps < n_steps → collect phase
    model.learn(total_timesteps=10)
    elapsed = time.time() - t0
    assert elapsed < 60, f"PPO demorou {elapsed:.1f}s (limite: 60s)"


def test_T05_ddpg_trains_10_steps(synthetic_df):
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
    env = _make_vec_env(synthetic_df)
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=float(0.5) * np.ones(n_actions),
    )
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        buffer_size=50000,
        batch_size=128,
        tau=0.001,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[64, 64])),
        verbose=0,
        learning_starts=0,   # permite aprender mesmo com poucas amostras
    )
    t0 = time.time()
    model.learn(total_timesteps=10)
    elapsed = time.time() - t0
    assert elapsed < 60, f"DDPG demorou {elapsed:.1f}s (limite: 60s)"


# ===========================================================================
# T06 — VecEnv reset retorna obs 2D
# ===========================================================================
def test_T06_vecenv_reset_shape(synthetic_df):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from preprocessing.preprocessors import data_split
    from env.EnvMultipleStock_train import StockEnvTrain
    df = data_split(synthetic_df, start=20200101, end=20200999)
    vec = DummyVecEnv([lambda: StockEnvTrain(df)])
    obs = vec.reset()
    # DummyVecEnv.reset() retorna obs com shape (n_envs, obs_dim)
    assert isinstance(obs, np.ndarray), "obs do VecEnv deve ser np.ndarray"
    assert obs.ndim == 2, f"obs do VecEnv deve ser 2D; shape={obs.shape}"
    assert obs.shape == (1, 181), f"shape esperado (1, 181); obtido {obs.shape}"


# ===========================================================================
# T07 — run_DRL importa sem disparar run_model()
# ===========================================================================
def test_T07_run_drl_importable(monkeypatch):
    """Garante que importar run_DRL.py não executa run_model() nem falha."""
    import importlib
    # Já teremos importado antes; força reload para cobrir o caminho de import
    if "run_DRL" in sys.modules:
        del sys.modules["run_DRL"]
    module = importlib.import_module("run_DRL")
    assert hasattr(module, "run_model"), "run_DRL deve exportar run_model()"
