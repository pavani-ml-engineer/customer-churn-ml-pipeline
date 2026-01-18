import pandas as pd


def preprocess_telco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for Telco churn dataset:
    - Convert TotalCharges to numeric
    - Handle missing values
    - Normalize Yes/No fields
    - Create binary target for churn
    """
    df = df.copy()

    # Strip spaces just in case
    df.columns = [c.strip() for c in df.columns]

    # Convert TotalCharges to numeric (it is sometimes stored as string with blanks)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID (identifier, not a predictive feature)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target variable
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Fill missing numeric values with median
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df
