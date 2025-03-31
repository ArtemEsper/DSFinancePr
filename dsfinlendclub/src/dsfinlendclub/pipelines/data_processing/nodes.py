"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.8
"""

import pandas as pd


def check_and_remove_duplicates(df, key="id"):
    """
    Check and remove duplicate entries based on a key column, if it exists.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        key (str): The column to check for duplicates (default is 'id').

    Returns:
        Tuple[pd.DataFrame, bool]: Cleaned DataFrame and a flag indicating deduplication was performed successfully.
    """
    if key not in df.columns:
        print(f"Column '{key}' not found. Skipping deduplication.")
        return df, True  # ✅ Consider deduplication 'successful' if key is missing

    duplicate_ids = df[df.duplicated(key, keep=False)][key].unique()
    found_duplicates = len(duplicate_ids) > 0

    if found_duplicates:
        print(f"Found {len(duplicate_ids)} duplicated '{key}' values:")
        print(duplicate_ids)
    else:
        print(f"No duplicates found in column '{key}'.")

    df_cleaned = df.drop_duplicates(subset=key, keep="first")

    return df_cleaned, True  # ✅ Always return True if function completes


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
    df["term"] = pd.to_numeric(
        df["term"].str.extract(r"(\d+)")[0], errors="coerce"
    ).astype(
        "Int64"
    )  # nullable integer dtype

    df["int_rate"] = pd.to_numeric(df["int_rate"].str.rstrip("%"), errors="coerce")

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    df["hardship_loan_status"] = df["hardship_loan_status"].apply(
        lambda x: x.strip().title() if isinstance(x, str) else x
    )
    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    df["revol_util"] = pd.to_numeric(
        df["revol_util"].astype(str).str.rstrip("%"), errors="coerce"
    )

    df["pub_rec"] = pd.to_numeric(df["pub_rec"], errors="coerce")  # Ensures numeric

    df["revol_util"] = df["revol_util"].clip(
        upper=100
    )  # set values more than 100 to NaN

    df["initial_list_status"] = df["initial_list_status"].apply(
        lambda x: x.lower().strip() if isinstance(x, str) else x
    )

    df["sec_app_earliest_cr_line"] = pd.to_datetime(
        df["sec_app_earliest_cr_line"], format="%b-%Y", errors="coerce"
    )

    df["mths_since_last_major_derog"] = df["mths_since_last_major_derog"].fillna(999)
    df["mths_since_last_major_derog"] = df["mths_since_last_major_derog"].clip(
        upper=999
    )

    # Ensure annual_inc_joint is numeric (may contain nulls)
    df["annual_inc_joint"] = pd.to_numeric(df["annual_inc_joint"], errors="coerce")

    df["dti_joint"] = pd.to_numeric(df["dti_joint"], errors="coerce")

    df["hardship_dpd"] = pd.to_numeric(df["hardship_dpd"], errors="coerce")

    df["mths_since_last_record"] = pd.to_numeric(
        df["mths_since_last_record"], errors="coerce"
    ).astype("Float64")

    # # Clean and standardize verification_status fields
    for col in ["verification_status", "verification_status_joint"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.strip().lower() if isinstance(x, str) else x
            )

    for col in ["revol_bal", "revol_bal_joint"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["home_ownership"] = df["home_ownership"].apply(
        lambda x: (
            "other"
            if isinstance(x, str) and x.strip().lower() in {"none", "any", "unknown"}
            else x.strip().lower() if isinstance(x, str) else x
        )
    )

    return df


def clean_remaining_object_columns(
    df: pd.DataFrame, exclude: list = None
) -> pd.DataFrame:
    """
    Convert all object-type columns to clean lowercase strings,
    excluding columns listed in `exclude`. NaNs are preserved.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        exclude (list): Optional list of columns to skip.

    Returns:
        pd.DataFrame: Updated DataFrame with cleaned object columns.
    """
    if exclude is None:
        exclude = []

    text_cols_to_clean = [
        col for col in df.select_dtypes(include="object").columns if col not in exclude
    ]

    for col in text_cols_to_clean:
        df[col] = df[col].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

    return df


def encode_joint_application_flag(df):
    """
    Encodes a binary flag for joint applications from 'application_type' column,
    if it is present. If 'application_type' is missing but 'is_joint_app' exists,
    the function does nothing. If both are missing, defaults to 0.

    Parameters:
        df (pd.DataFrame): DataFrame possibly containing 'application_type' or 'is_joint_app'.

    Returns:
        pd.DataFrame: Updated DataFrame with 'is_joint_app' column.
    """
    if "application_type" in df.columns:
        df["is_joint_app"] = df["application_type"].apply(
            lambda x: (
                1 if isinstance(x, str) and x.strip().lower() == "joint app" else 0
            )
        )

    elif "is_joint_app" in df.columns:
        # Already present — do nothing
        pass

    else:
        # Neither column present — default to 0
        df["is_joint_app"] = 0

    return df


def filter_and_encode_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame to only include rows with clearly resolved loan statuses
    and encodes them into a binary target column.

    - Keeps: 'Fully Paid', 'Charged Off', 'Default'
    - Encodes:
        'Fully Paid' → 0
        'Charged Off' / 'Default' → 1
    - Drops all other statuses.

    Parameters:
        df (pd.DataFrame): Input data with 'loan_status'

    Returns:
        pd.DataFrame: Cleaned and encoded DataFrame
    """
    valid_statuses = ["fully paid", "charged off", "default"]
    df["loan_status"] = df["loan_status"].str.strip().str.lower()
    df["loan_status_binary"] = df["loan_status"].map(
        {"fully Paid": 0, "charged Off": 1, "default": 1}
    )
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid or nonsensical values."""
    df = df[df["loan_amnt"] > 0]
    df = df[df["annual_inc"] > 0]
    df = df[(df["dti"] >= 0) & (df["dti"] <= 100)]
    df = df[
        (df["dti_joint"].isna()) | ((df["dti_joint"] >= 0) & (df["dti_joint"] <= 100))
    ]
    return df


def clean_string_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and clean string-based fields while handling NaNs properly."""
    # Handle emp_length
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].replace("n/a", None)

    # Handle purpose
    if "purpose" in df.columns:
        df["purpose"] = df["purpose"].apply(
            lambda x: x.lower().replace("_", " ") if isinstance(x, str) else x
        )

    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap extreme values to the 99th percentile."""
    for col in [
        "revol_bal",
        "revol_bal_joint",
        "annual_inc",
        "annual_inc_joint",
        "hardship_dpd",
    ]:
        if col in df.columns:
            df[col] = df[col].clip(upper=df[col].quantile(0.99))

    return df
