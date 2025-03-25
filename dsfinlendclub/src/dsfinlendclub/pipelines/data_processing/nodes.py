"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.8
"""

import pandas as pd


def check_and_remove_duplicates(df, key="id"):
    """
    Check and remove duplicate entries based on a key column.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    key (str): The column to check for duplicates (default is 'id')

    Returns:
    pd.DataFrame: A new DataFrame with duplicates removed
    """
    duplicate_ids = df[df.duplicated(key, keep=False)][key].unique()

    if len(duplicate_ids) > 0:
        print(f"Found {len(duplicate_ids)} duplicated '{key}' values:")
        print(duplicate_ids)
    else:
        print(f"No duplicates found in column '{key}'.")

    # Drop duplicates, keeping the first occurrence
    df_cleaned = df.drop_duplicates(subset=key, keep="first")

    return df_cleaned


def drop_unwanted_columns(x: pd.DataFrame, drop_list=None) -> pd.DataFrame:
    """
    Drops unwanted columns from the DataFrame.

    Parameters:
        :param drop_list: List of column names to drop. Defaults to ['Unnamed: 0.1', 'Unnamed: 0'].
        :param x: df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns dropped.

    """
    if drop_list is None:
        drop_list = ["Unnamed: 0.1", "Unnamed: 0"]

    return x.drop(columns=[col for col in drop_list if col in x.columns], axis=1)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to lowercase and strip whitespace."""
    df.columns = df.columns.str.lower().str.strip()
    return df


def fix_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert types of term, int_rate, issue_d."""
    df['term'] = df['term'].astype(str)
    df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float)
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid or nonsensical values."""
    df = df[df['loan_amnt'] > 0]
    df = df[df['annual_inc'] > 0]
    df = df[(df['dti'] >= 0) & (df['dti'] <= 100)]
    return df


def filter_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only loans with clear outcomes for supervised learning."""
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    return df[df['loan_status'].isin(valid_statuses)].copy()


def clean_string_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and clean string-based fields."""
    df['emp_length'] = df['emp_length'].replace('n/a', None)
    df['purpose'] = df['purpose'].str.lower().str.replace('_', ' ', regex=False)
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap extreme values to the 99th percentile."""
    df['annual_inc'] = df['annual_inc'].clip(upper=df['annual_inc'].quantile(0.99))
    return df
