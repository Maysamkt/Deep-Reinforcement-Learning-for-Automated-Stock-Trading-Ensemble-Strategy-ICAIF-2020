import pathlib
import pandas as pd
import datetime
import os

# data
TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
TESTING_DATA_FILE = "test.csv"

# trained models and results directories (safe timestamp for Windows and Linux)
now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)
