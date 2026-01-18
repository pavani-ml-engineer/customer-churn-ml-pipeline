import pandas as pd
from pathlib import Path


def load_churn_data():
    """
    Load the Telco Customer Churn dataset.
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)
    return df


if __name__ == "__main__":
    df = load_churn_data()
    print(df.head())
    print(df.info())
