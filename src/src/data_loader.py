import pandas as pd
from pathlib import Path


def load_churn_data():
    data_path = Path(__file__).resolve().parents[1] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    return pd.read_csv(data_path)
