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
    Tuple[pd.DataFrame, bool]: Cleaned DataFrame and a flag indicating if any duplicates were found
    """
    duplicate_ids = df[df.duplicated(key, keep=False)][key].unique()

    found_duplicates = len(duplicate_ids) > 0

    if found_duplicates:
        print(f"Found {len(duplicate_ids)} duplicated '{key}' values:")
        print(duplicate_ids)
    else:
        print(f"No duplicates found in column '{key}'.")

    df_cleaned = df.drop_duplicates(subset=key, keep="first")

    return df_cleaned, found_duplicates


def drop_unwanted_columns(x: pd.DataFrame, drop_list=None) -> pd.DataFrame:
    """
    Drops unwanted columns from the DataFrame.

    Parameters:
        :param drop_list: List of column names to drop. Defaults to ['Unnamed: 0.1', 'Unnamed: 0'].
        :param x: df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns dropped.

    """
    print("Dropping columns:", drop_list)
    if drop_list is None:
        drop_list = ["Unnamed: 0.1", "Unnamed: 0"]

    return x.drop(columns=[col for col in drop_list if col in x.columns], axis=1)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to lowercase and strip whitespace."""
    df.columns = df.columns.str.lower().str.strip()
    return df


def fix_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert types of term, int_rate, issue_d, etc."""
    df['term'] = df['term'].str.extract(r'(\d+)').astype(int)
    df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    df["revol_util"] = pd.to_numeric(df["revol_util"].str.rstrip("%"), errors="coerce")

    df["initial_list_status"] = df["initial_list_status"].astype(str).str.lower().str.strip()

    df["sec_app_earliest_cr_line"] = pd.to_datetime(df["sec_app_earliest_cr_line"], format="%b-%Y", errors="coerce")

    df["home_ownership"] = (
        df["home_ownership"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "none": "other",
            "any": "other",
            "unknown": "other"
        })
    )

    # Converting remaining object columns to clean strings (for later feature encoding)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df


def filter_and_flag_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with statuses not present in 'valid' list and encode the rest in binary format."""
    valid = ["Fully Paid", "Charged Off", "Default"]
    df = df[df["loan_status"].isin(valid)].copy()
    df["loan_status_binary"] = df["loan_status"].map({
        "Fully Paid": 0,
        "Charged Off": 1,
        "Default": 1
    })
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid or nonsensical values."""
    df = df[df["loan_amnt"] > 0]
    df = df[df["annual_inc"] > 0]
    df = df[(df["dti"] >= 0) & (df["dti"] <= 100)]
    return df


def filter_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only loans with clear outcomes for supervised learning."""
    valid_statuses = ["Fully Paid", "Charged Off", "Default"]
    return df[df["loan_status"].isin(valid_statuses)].copy()


def clean_string_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and clean string-based fields."""
    df["emp_length"] = df["emp_length"].replace("n/a", None)
    df["purpose"] = df["purpose"].str.lower().str.replace("_", " ", regex=False)
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap extreme values to the 99th percentile."""
    df["annual_inc"] = df["annual_inc"].clip(upper=df["annual_inc"].quantile(0.99))
    return df
