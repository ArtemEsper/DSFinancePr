"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.8
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple


# debug function
def print_columns(df, stage_name):
    """Debug function to print column names at each stage."""
    print(f"Columns at {stage_name}: {list(df.columns)}")
    return df


def drop_unwanted_columns(x: pd.DataFrame, drop_list=None) -> pd.DataFrame:
    """
    Drops unwanted columns from the DataFrame and returns a copy for forked downstream use.

    Parameters:
        x: Input DataFrame.
        drop_list: List of columns to drop.

    Returns:
        Tuple of two identical DataFrames (post-drop), for forked pipeline paths.
    """
    print("Dropping columns:", drop_list)
    if drop_list is None:
        drop_list = ["Unnamed: 0.1", "Unnamed: 0"]

    return x.drop(columns=[col for col in drop_list if col in x.columns], axis=1)


def create_has_hardship_flag(df):
    """
    Creates a numeric hardship indicator from 'hardship_flag'.

    Parameters:
        df (pd.DataFrame): Input dataframe containing 'hardship_flag' column

    Returns:
        pd.DataFrame: Updated dataframe with 'has_hardship' column and 'hardship_flag' (will be dropped later)
    """
    if "hardship_flag" in df.columns:
        df["has_hardship"] = (df["hardship_flag"].str.lower() == "y").astype(int)
        # df.drop(columns=["hardship_flag"], inplace=True)
    else:
        print(
            "Warning: 'hardship_flag' column not found. Skipping creation of 'has_hardship'."
        )

    return df


def engineer_emp_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer model-specific 'emp_length' features for tree-based and regression models.

    Args:
        df (pd.DataFrame): Raw DataFrame with 'emp_length' field.

    Returns:
        pd.DataFrame: DataFrame with new engineered columns:
            - emp_length_clean
            - emp_length_clean_tree
            - emp_length_clean_reg
    """

    # Map textual employment lengths to numeric
    emp_length_mapping = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
        "n/a": None,
    }

    # Map to numeric values
    df["emp_length_clean"] = df["emp_length"].map(emp_length_mapping)

    # For tree-based models: fill missing with -1
    df["emp_length_clean_tree"] = df["emp_length_clean"].fillna(-1)

    # For regression/SVM models: fill missing with median
    imputer = SimpleImputer(strategy="median")
    df[["emp_length_clean_reg"]] = imputer.fit_transform(df[["emp_length_clean"]])

    # df.drop(columns=["emp_length_clean"], inplace=True)
    # df.drop(columns=["emp_length"], inplace=True)

    return df


def create_was_late_before_hardship(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary flag indicating whether the borrower's hardship started while the loan was already in a late status.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'hardship_loan_status'.

    Returns:
        pd.DataFrame: DataFrame with new 'was_late_before_hardship' feature.
    """
    if "hardship_loan_status" in df.columns:
        df["was_late_before_hardship"] = (
            df["hardship_loan_status"].str.contains("late", na=False).astype(int)
        )
        # df.drop(columns=["hardship_loan_status"], inplace=True)
    else:
        df["was_late_before_hardship"] = 0  # default to 0 if field is missing

    return df


def encode_interest_and_grade_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode interest rate and LendingClub classification fields: 'grade' and 'sub_grade'.

    - Converts 'int_rate' from percentage to decimal
    - Maps 'grade' to ordinal values
    - Maps 'sub_grade' to ordered numeric levels

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'int_rate', 'grade', and 'sub_grade'

    Returns:
        pd.DataFrame: Updated DataFrame with 'int_rate', 'grade_encoded', and 'sub_grade_encoded'
    """
    # Convert interest rate from percent to decimal
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"] / 100

    # Encode grade
    grade_mapping = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}
    if "grade" in df.columns:
        df["grade_encoded"] = df["grade"].map(grade_mapping)
        # df.drop(columns=["grade"], inplace=True)

    # Encode sub_grade
    subgrades = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
    ]
    subgrade_mapping = {sub: i + 1 for i, sub in enumerate(subgrades)}
    if "sub_grade" in df.columns:
        df["sub_grade_encoded"] = df["sub_grade"].map(subgrade_mapping)
        # df.drop(columns=["sub_grade"], inplace=True)

    return df


def create_joint_income_feature(df):
    """
    Creates a unified income feature ('annual_inc_final') that takes into account joint applications.
    If the application is joint, use 'annual_inc_joint', otherwise use 'annual_inc'.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'annual_inc', 'annual_inc_joint', and 'is_joint_app'.

    Returns:
        pd.DataFrame: Updated DataFrame with 'annual_inc_final'.
    """
    df["annual_inc_final"] = df["annual_inc"]
    if "is_joint_app" in df.columns and "annual_inc_joint" in df.columns:
        df.loc[df["is_joint_app"] == 1, "annual_inc_final"] = df.loc[
            df["is_joint_app"] == 1, "annual_inc_joint"
        ]
    return df


def create_income_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-specific income features.
    """
    if "annual_inc_final" in df.columns:
        # Tree-based features
        df["income_band_tree"] = pd.qcut(
            df["annual_inc_final"].clip(upper=df["annual_inc_final"].quantile(0.99)),
            q=10,
            labels=False,
            duplicates="drop",
        )

        # High income flag for trees
        df["is_high_income_tree"] = (
                df["annual_inc_final"] > df["annual_inc_final"].median()
        ).astype(int)

        # Regression features
        df["income_log_reg"] = np.log1p(df["annual_inc_final"])

        # Income to loan amount ratio for regression
        if "loan_amnt" in df.columns:
            df["income_to_loan_reg"] = (df["annual_inc_final"] / df["loan_amnt"]).clip(
                upper=df["annual_inc_final"].quantile(0.99)
            )

    return df


def create_joint_dti_feature(df):
    """
    Creates a unified DTI feature ('dti_final') that accounts for joint applications.
    If the application is joint, use 'dti_joint', otherwise use 'dti'.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'dti', 'dti_joint', and 'is_joint_app'.

    Returns:
        pd.DataFrame: Updated DataFrame with 'dti_final'.
    """
    df["dti_final"] = df["dti"]
    if "is_joint_app" in df.columns and "dti_joint" in df.columns:
        df.loc[df["is_joint_app"] == 1, "dti_final"] = df.loc[
            df["is_joint_app"] == 1, "dti_joint"
        ]
    return df


def create_dti_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-specific DTI (Debt-to-Income) features.
    """
    if "dti_final" in df.columns:
        # Tree-based features
        df["dti_band_tree"] = pd.qcut(
            df["dti_final"].clip(upper=df["dti_final"].quantile(0.99)),
            q=10,
            labels=False,
            duplicates="drop",
        )

        # High DTI flag for trees
        df["is_high_dti_tree"] = (df["dti_final"] > df["dti_final"].median()).astype(
            int
        )

        # Regression features
        df["dti_normalized_reg"] = df["dti_final"] / 100  # Convert to 0-1 scale

        # Log transform for regression
        df["dti_log_reg"] = np.log1p(df["dti_final"])

    return df


def create_term_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-specific loan term features.
    """
    if "term" in df.columns:
        # Tree-based features - already categorical
        df["term_tree"] = df["term"].astype("category")

        # Regression features - normalized
        df["term_normalized_reg"] = df["term"] / df["term"].max()

        # Create interaction terms for regression
        if "int_rate" in df.columns:
            df["term_rate_interaction_reg"] = df["term_normalized_reg"] * df["int_rate"]

    return df


def create_joint_verification_feature(df):
    """
    Creates a unified verification status feature ('verification_status_final')
    that combines individual and joint loan information.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'verification_status',
                           'verification_status_joint', and 'is_joint_app'.

    Returns:
        pd.DataFrame: Updated DataFrame with 'verification_status_final'.
    """
    df["verification_status_final"] = df["verification_status"]
    if "is_joint_app" in df.columns and "verification_status_joint" in df.columns:
        df.loc[df["is_joint_app"] == 1, "verification_status_final"] = df.loc[
            df["is_joint_app"] == 1, "verification_status_joint"
        ]

    # # Drop original columns if present
    # df.drop(
    #     columns=[
    #         col
    #         for col in ["verification_status", "verification_status_joint"]
    #         if col in df.columns
    #     ],
    #     inplace=True,
    # )

    return df


def create_joint_revol_bal_feature(df):
    """
    Creates a unified revolving balance feature ('revol_bal_final')
    that accounts for joint applications.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'revol_bal',
                           'revol_bal_joint', and 'is_joint_app'.

    Returns:
        pd.DataFrame: Updated DataFrame with 'revol_bal_final'.
    """
    df["revol_bal_final"] = df["revol_bal"]
    if "is_joint_app" in df.columns and "revol_bal_joint" in df.columns:
        df.loc[df["is_joint_app"] == 1, "revol_bal_final"] = df.loc[
            df["is_joint_app"] == 1, "revol_bal_joint"
        ]

    # # Drop original columns if present
    # df.drop(
    #     columns=[col for col in ["revol_bal", "revol_bal_joint"] if col in df.columns],
    #     inplace=True,
    # )

    return df


def create_hardship_features(df):
    """
    Feature engineering for hardship-related columns.
    """
    df["hardship_dpd_filled"] = df["hardship_dpd"].fillna(0)
    # df.drop(columns=["hardship_dpd"], inplace=True)
    return df


def create_mths_since_last_record_feature(df):
    """
    Feature engineering for 'mths_since_last_record'.
    Converts missing values to 999 (no derogatory record) and caps extreme values.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'mths_since_last_record'.

    Returns:
        pd.DataFrame: Updated DataFrame with 'mths_since_last_record_filled'.
    """
    if "mths_since_last_record" in df.columns:
        df["mths_since_last_record_filled"] = df["mths_since_last_record"].fillna(999)
        df["mths_since_last_record_filled"] = df["mths_since_last_record_filled"].clip(
            upper=300
        )
    else:
        df["mths_since_last_record_filled"] = 999

    # df.drop(columns=["mths_since_last_record"], inplace=True)

    return df


def create_revol_util_features(df):
    """
    Create model-specific features from 'revol_util' field:
    - Fills with -1 for tree-based models
    - Fills with median for regression-based models

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'revol_util'

    Returns:
        pd.DataFrame: Updated DataFrame with 'revol_util_tree' and 'revol_util_reg'
    """
    if "revol_util" in df.columns:
        # Tree-based models: fill missing with -1
        df["revol_util_tree"] = df["revol_util"].fillna(-1)

        # Regression models: fill missing with median
        imputer = SimpleImputer(strategy="median")
        df[["revol_util_reg"]] = imputer.fit_transform(df[["revol_util"]])

    else:
        df["revol_util_tree"] = -1
        df["revol_util_reg"] = -1

    return df


def create_utilization_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-specific credit utilization features.

    Args:
        df (pd.DataFrame): Input DataFrame containing revol_util column

    Returns:
        pd.DataFrame: DataFrame with added utilization features
    """
    if "revol_util" in df.columns:
        # Tree-based features
        df["util_band_tree"] = pd.qcut(
            df["revol_util"].clip(upper=df["revol_util"].quantile(0.99)),
            q=10,
            labels=False,
            duplicates="drop",
        )

        # High utilization flag for trees
        df["is_high_util_tree"] = (df["revol_util"] > df["revol_util"].median()).astype(
            int
        )

        # Regression features
        df["util_normalized_reg"] = df["revol_util"] / 100

        # Create utilization buckets for regression
        # Convert series to float type first
        util_series = df["revol_util"].astype(float)
        bins = np.array([0, 20, 40, 60, 80, 100], dtype=float)
        labels = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)

        df["util_buckets_reg"] = pd.cut(
            util_series, bins=bins, labels=labels, include_lowest=True
        )

    return df


def create_initial_list_status_flag(df):
    """
    Create a binary flag for 'initial_list_status' where 'w' becomes 1 and others become 0.
    Drops the original column after encoding.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'initial_list_status'.

    Returns:
        pd.DataFrame: Updated DataFrame with 'initial_list_status_flag'.
    """
    if "initial_list_status" in df.columns:
        df["initial_list_status_flag"] = (df["initial_list_status"] == "w").astype(int)
        # df.drop(columns=["initial_list_status"], inplace=True)

    return df


def create_credit_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a unified feature 'earliest_cr_line_final' combining individual and joint applicants'
    credit line information, and calculates credit age in months.

    Parameters:
        df (pd.DataFrame): DataFrame with 'earliest_cr_line', 'sec_app_earliest_cr_line',
                           'issue_d', and 'is_joint_app'.

    Returns:
        pd.DataFrame: Updated DataFrame with 'earliest_cr_line_final' and 'credit_age_months'.
    """
    # Ensure date columns are properly parsed as datetime
    if "earliest_cr_line" in df.columns and not pd.api.types.is_datetime64_dtype(df["earliest_cr_line"]):
        df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")

    if "sec_app_earliest_cr_line" in df.columns and not pd.api.types.is_datetime64_dtype(
            df["sec_app_earliest_cr_line"]):
        df["sec_app_earliest_cr_line"] = pd.to_datetime(df["sec_app_earliest_cr_line"], errors="coerce")

    if "issue_d" in df.columns and not pd.api.types.is_datetime64_dtype(df["issue_d"]):
        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")

    # Default to primary applicant's earliest credit line
    df["earliest_cr_line_final"] = df["earliest_cr_line"]

    # Use secondary applicant's date if it's a joint application
    if "is_joint_app" in df.columns and "sec_app_earliest_cr_line" in df.columns:
        joint_mask = (df["is_joint_app"] == 1) & df["sec_app_earliest_cr_line"].notna()
        df.loc[joint_mask, "earliest_cr_line_final"] = df.loc[joint_mask, "sec_app_earliest_cr_line"]

    # Calculate credit age in months
    if "issue_d" in df.columns and "earliest_cr_line_final" in df.columns:
        valid_dates_mask = df["issue_d"].notna() & df["earliest_cr_line_final"].notna()
        df["credit_age_months"] = np.nan

        if valid_dates_mask.any():
            df.loc[valid_dates_mask, "credit_age_months"] = (
                    (df.loc[valid_dates_mask, "issue_d"] - df.loc[valid_dates_mask, "earliest_cr_line_final"])
                    .dt.total_seconds() / (30 * 24 * 60 * 60)
            )

            # Clip to avoid negative durations
            df["credit_age_months"] = df["credit_age_months"].clip(lower=0)

    return df


def handle_pub_rec_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Handles missing values in the 'pub_rec' column using a specified strategy.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        strategy (str): Strategy to fill NaNs, either "median" or "negative_one".

    Returns:
        pd.DataFrame: DataFrame with 'pub_rec' column updated.
    """
    if "pub_rec" not in df.columns:
        print("Column 'pub_rec' not found — skipping.")
        return df

    df["pub_rec"] = pd.to_numeric(df["pub_rec"], errors="coerce")

    if strategy == "median":
        median_val = df["pub_rec"].median()
        df["pub_rec"] = df["pub_rec"].fillna(median_val)
    elif strategy == "negative_one":
        df["pub_rec"] = df["pub_rec"].fillna(-1)
    else:
        raise ValueError("Unsupported strategy. Use 'median' or 'negative_one'.")

    return df


def encode_purpose_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    Working with preprocessed 'purpose' field that's already
    lowercase and space-normalized.
    Adds one-hot encoding of 'purpose_cleaned' field
    """
    if "purpose" not in df.columns:
        return df

    # Already cleaned in preprocessing, so just handle rare categories
    purpose_counts = df["purpose"].value_counts()
    min_count = max(100, len(df) * 0.01)  # Dynamic threshold
    rare_purposes = purpose_counts[purpose_counts < min_count].index

    df["purpose_cleaned"] = df["purpose"].apply(
        lambda x: "other" if x in rare_purposes else x
    )

    dummies = pd.get_dummies(df["purpose_cleaned"], prefix="purpose", dummy_na=True)
    return pd.concat([df, dummies], axis=1)


def create_fico_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    FICO feature creation
    """
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_average"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

        # Additional validation
        df["fico_average"] = df["fico_average"].clip(lower=300, upper=850)

        # Create multiple risk band versions for different model types
        try:
            df["fico_risk_band"] = pd.qcut(
                df["fico_average"],
                q=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
                labels=["F", "E", "D", "C", "B", "A"],
            )
        except ValueError:
            # Handle case where there aren't enough unique values
            df["fico_risk_band"] = pd.cut(
                df["fico_average"],
                bins=[300, 580, 650, 700, 750, 800, 850],
                labels=["F", "E", "D", "C", "B", "A"],
            )

    return df


def create_fico_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-specific FICO score features.
    """
    if "fico_average" in df.columns:
        # Tree-based features - discretized bands
        df["fico_band_tree"] = pd.qcut(
            df["fico_average"], q=10, labels=False, duplicates="drop"
        )

        # Create binary high/low FICO indicators
        df["is_high_fico_tree"] = (
                df["fico_average"] > df["fico_average"].median()
        ).astype(int)

        # Regression features - normalized scores
        df["fico_normalized_reg"] = (df["fico_average"] - df["fico_average"].min()) / (
                df["fico_average"].max() - df["fico_average"].min()
        )

    return df


def create_loan_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from loan amount and installment relationship
    """
    if "loan_amnt" in df.columns and "installment" in df.columns:
        # Avoid division by zero
        df["loan_to_installment_ratio"] = df["loan_amnt"] / df["installment"].replace(
            0, np.nan
        )

        # Handle extreme values
        df["loan_to_installment_ratio"] = df["loan_to_installment_ratio"].clip(
            lower=df["loan_to_installment_ratio"].quantile(0.01),
            upper=df["loan_to_installment_ratio"].quantile(0.99),
        )

        try:
            df["loan_amount_band"] = pd.qcut(
                df["loan_amnt"], q=10, labels=False, duplicates="drop"
            )
        except ValueError:
            # Fall back to fixed bins if qcut fails
            df["loan_amount_band"] = pd.cut(df["loan_amnt"], bins=10, labels=False)

    return df


def create_credit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine credit history indicators into meaningful features
    """
    if "delinq_2yrs" in df.columns and "pub_rec" in df.columns:
        # Create composite derogatory indicator
        df["has_derogatory"] = ((df["delinq_2yrs"] > 0) | (df["pub_rec"] > 0)).astype(
            int
        )

        # Weight recent delinquencies more heavily
        if "mths_since_last_delinq" in df.columns:
            df["delinq_weight"] = df["delinq_2yrs"] * (
                    1 + 1 / df["mths_since_last_delinq"].clip(lower=1)
            )

    return df


def create_credit_history_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-specific features from credit history indicators.
    """
    # Tree-based model features
    if "delinq_2yrs" in df.columns:
        df["delinq_2yrs_tree"] = df["delinq_2yrs"].fillna(-1)

        # Create binary flags for tree models
        df["has_delinq_tree"] = (df["delinq_2yrs"] > 0).astype(int)
        df["has_recent_delinq_tree"] = (
                df["mths_since_last_delinq"].fillna(999) < 24
        ).astype(int)

    # Regression model features
    if "delinq_2yrs" in df.columns:
        imputer = SimpleImputer(strategy="median")
        df[["delinq_2yrs_reg"]] = imputer.fit_transform(df[["delinq_2yrs"]])

        # Create normalized delinquency score
        if "mths_since_last_delinq" in df.columns:
            df["delinq_severity_reg"] = df["delinq_2yrs"] / (
                df["mths_since_last_delinq"].clip(lower=1)
            )
            df["delinq_severity_reg"] = df["delinq_severity_reg"].fillna(
                df["delinq_severity_reg"].median()
            )

    return df


def evaluate_feature_engineering(df: pd.DataFrame) -> dict:
    """
    Feature evaluation with comprehensive metrics
    """
    metrics = {
        "missing_rates": df.isnull().mean().to_dict(),
        "zero_rates": (df == 0).mean().to_dict(),
    }

    # Basic quality metrics

    # Distribution metrics for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    metrics["distributions"] = {
        col: {
            "skew": df[col].skew(),
            "kurtosis": df[col].kurtosis(),
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "1%": df[col].quantile(0.01),
            "99%": df[col].quantile(0.99),
            "unique_count": df[col].nunique(),
        }
        for col in numeric_cols
    }

    # Categorical column analysis
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    metrics["categorical"] = {
        col: {
            "unique_count": df[col].nunique(),
            "top_categories": df[col].value_counts().nlargest(5).to_dict(),
        }
        for col in cat_cols
    }

    # Target correlation if available
    if "loan_status_binary" in df.columns:
        metrics["target_correlations"] = (
            df.corr()["loan_status_binary"].sort_values(ascending=False).to_dict()
        )

    return metrics


def create_model_specific_datasets(df: pd.DataFrame,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates separate datasets optimized for tree-based and regression models.

    This function identifies and properly handles model-specific features
    (those ending with _tree or _reg) and ensures critical features like
    home_ownership_ordinal are preserved in both output datasets.

    Args:
        df: DataFrame with all engineered features

    Returns:
        tuple: (tree_features_df, regression_features_df)
    """
    # Print input columns for debugging
    print(f"Input columns to create_model_specific_datasets: {df.columns.tolist()}")

    # Create copies for tree and regression datasets
    tree_features = df.copy()
    regression_features = df.copy()

    # Identify all model-specific columns
    tree_specific_cols = [col for col in df.columns if col.endswith('_tree')]
    reg_specific_cols = [col for col in df.columns if col.endswith('_reg')]

    # Define the base features that should be retained in both datasets
    # This is a more comprehensive approach than using a static mapping
    essential_columns = [
        # Target variable
        "loan_status_binary",

        # Categorical encodings
        "home_ownership_ordinal",
        "purpose_high_risk",

        # Important loan characteristics
        "loan_amnt",
        "term",
        "int_rate",
        "grade_encoded",
        "sub_grade_encoded",

        # Credit score features
        "fico_average",
        "fico_risk_band",

        # Income features
        "annual_inc_final",

        # DTI features
        "dti_final",

        # Credit history features
        "credit_age_months",
        "has_derogatory",
        "delinquency_score",

        # Loan characteristic features
        "loan_to_installment_ratio",

        # Credit utilization
        "revol_util",

        # Joint application features
        "is_joint_app",
        "verification_status_final",

        # Hardship features
        "has_hardship",
        "was_late_before_hardship",
    ]

    # Map base features to their model-specific versions
    model_specific_mappings = {
        # Tree models - generally use missing value indicators
        "tree": {
            "emp_length_clean": "emp_length_clean_tree",
            "revol_util": "revol_util_tree",
            "delinq_2yrs": "delinq_2yrs_tree",
            # Add other mappings as needed
        },
        # Regression models - generally use imputed values
        "regression": {
            "emp_length_clean": "emp_length_clean_reg",
            "revol_util": "revol_util_reg",
            "delinq_2yrs": "delinq_2yrs_reg",
            # Add other mappings as needed
        },
    }

    # Process tree features dataset
    # 1. Map model-specific features to their base names
    for base_col, tree_col in model_specific_mappings["tree"].items():
        if tree_col in df.columns:
            tree_features[base_col] = tree_features[tree_col]

    # 2. Keep only tree-specific features and essential columns
    tree_cols_to_keep = (
            [col for col in tree_features.columns if not col.endswith('_reg')] +
            essential_columns
    )
    tree_cols_to_keep = list(set(tree_cols_to_keep))  # Remove duplicates
    tree_features = tree_features[
        [col for col in tree_cols_to_keep if col in tree_features.columns]
    ]

    # Process regression features dataset
    # 1. Map model-specific features to their base names
    for base_col, reg_col in model_specific_mappings["regression"].items():
        if reg_col in df.columns:
            regression_features[base_col] = regression_features[reg_col]

    # 2. Keep only regression-specific features and essential columns
    reg_cols_to_keep = (
            [col for col in regression_features.columns if not col.endswith('_tree')] +
            essential_columns
    )
    reg_cols_to_keep = list(set(reg_cols_to_keep))  # Remove duplicates
    regression_features = regression_features[
        [col for col in reg_cols_to_keep if col in regression_features.columns]
    ]

    # Print output dataset columns for debugging
    print(f"Tree features columns: {tree_features.columns.tolist()}")
    print(f"Regression features columns: {regression_features.columns.tolist()}")

    # Verify home_ownership_ordinal is present in both datasets
    for dataset_name, dataset in [("Tree", tree_features), ("Regression", regression_features)]:
        if "home_ownership_ordinal" in dataset.columns:
            print(f"✅ home_ownership_ordinal is present in {dataset_name} dataset")
        else:
            print(f"❌ home_ownership_ordinal is missing from {dataset_name} dataset")

    return tree_features, regression_features


def create_home_ownership_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """
        Create an ordinal encoding for home ownership status that reflects
    risk level.

        Parameters:
            df (pd.DataFrame): Input DataFrame with 'home_ownership' column

        Returns:
            pd.DataFrame: DataFrame with new 'home_ownership_ordinal' feature
    """
    # Risk order: 'other' (highest risk) > 'rent' > 'mortgage' > 'own' (lowest risk)
    ownership_risk_map = {"other": 3, "rent": 2, "mortgage": 1, "own": 0}

    if "home_ownership" in df.columns:
        df["home_ownership_ordinal"] = df["home_ownership"].map(ownership_risk_map)
        # Fill any missing values with highest risk level
        df["home_ownership_ordinal"] = df["home_ownership_ordinal"].fillna(3)

    return df


def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from date columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame with date columns

    Returns:
        pd.DataFrame: DataFrame with new time-based features
    """
    # Loan age at time of default/repayment
    if "issue_d" in df.columns and "last_pymnt_d" in df.columns:
        # Ensure both columns are datetime
        if not pd.api.types.is_datetime64_dtype(df["issue_d"]):
            df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")

        if not pd.api.types.is_datetime64_dtype(df["last_pymnt_d"]):
            df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], errors="coerce")

        valid_dates = df["issue_d"].notna() & df["last_pymnt_d"].notna()
        df["loan_age_months"] = np.nan

        if valid_dates.any():
            df.loc[valid_dates, "loan_age_months"] = (
                    (df.loc[valid_dates, "last_pymnt_d"] - df.loc[valid_dates, "issue_d"])
                    .dt.days / 30
            ).clip(lower=0)

            # Binned version for tree models
            loan_age_bins = [0, 6, 12, 24, 36, 48, 60, float("inf")]
            loan_age_labels = [0, 1, 2, 3, 4, 5, 6]

            df["loan_age_bins"] = pd.cut(
                df["loan_age_months"],
                bins=loan_age_bins,
                labels=loan_age_labels,
                include_lowest=True,
            )

    # Seasonality features
    if "issue_d" in df.columns:
        if not pd.api.types.is_datetime64_dtype(df["issue_d"]):
            df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")

        valid_issue_dates = df["issue_d"].notna()
        df["issue_month"] = np.nan
        df["issue_quarter"] = np.nan
        df["issue_year"] = np.nan

        if valid_issue_dates.any():
            df.loc[valid_issue_dates, "issue_month"] = df.loc[valid_issue_dates, "issue_d"].dt.month
            df.loc[valid_issue_dates, "issue_quarter"] = df.loc[valid_issue_dates, "issue_d"].dt.quarter
            df.loc[valid_issue_dates, "issue_year"] = df.loc[valid_issue_dates, "issue_d"].dt.year

            # Economic cycle proxy (very simple)
            recession_years = [2008, 2009, 2020]
            df["recession_period"] = df["issue_year"].isin(recession_years).astype(int)

    return df


def create_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create payment-related features from loan details.

    Parameters:
        df (pd.DataFrame): Input DataFrame with loan payment info

    Returns:
        pd.DataFrame: DataFrame with new payment-related features
    """
    # Income to payment ratio (affordability metric)
    if "annual_inc_final" in df.columns and "installment" in df.columns:
        df["payment_to_income"] = (
                (df["installment"] * 12) / df["annual_inc_final"] * 100
        )
        df["payment_to_income"] = df["payment_to_income"].clip(upper=100)

        # Flag high payment burden
        df["high_payment_burden"] = (df["payment_to_income"] > 20).astype(int)

    # Interest payment burden
    if (
            "installment" in df.columns
            and "loan_amnt" in df.columns
            and "term" in df.columns
    ):
        # Total payments over loan life
        df["total_payments"] = df["installment"] * df["term"]
        # Total interest as percentage of loan amount
        df["interest_burden_pct"] = (
                                            (df["total_payments"] - df["loan_amnt"]) / df["loan_amnt"]
                                    ) * 100

    return df


def create_credit_inquiry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Create features related to credit inquiries and credit seeking
    behavior.

        Parameters:
            df (pd.DataFrame): Input DataFrame with inquiry information

        Returns:
            pd.DataFrame: DataFrame with new inquiry-related features
    """
    # Recent inquiry intensity
    if "inq_last_6mths" in df.columns and "inq_last_12m" in df.columns:
        # Calculate 6-month vs 12-month inquiry ratio
        df["recent_inquiry_intensity"] = df["inq_last_6mths"] / df["inq_last_12m"].clip(
            lower=1
        )
        df["recent_inquiry_intensity"] = df["recent_inquiry_intensity"].clip(upper=1)

        # Create a high recent inquiry flag
        df["high_recent_inquiries"] = (df["inq_last_6mths"] >= 3).astype(int)

    # Inquiries to accounts ratio (credit seeking success rate)
    if "inq_last_12m" in df.columns and "open_acc_6m" in df.columns:
        df["inq_to_open_acc_ratio"] = df["inq_last_12m"] / df["open_acc_6m"].clip(
            lower=1
        )
        df["inq_to_open_acc_ratio"] = df["inq_to_open_acc_ratio"].clip(upper=10)

    return df


def create_delinquency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that capture delinquency history and patterns.

    Parameters:
        df (pd.DataFrame): Input DataFrame with delinquency information.

    Returns:
        pd.DataFrame: DataFrame with new delinquency features.
    """
    delinq_columns = [
        "delinq_2yrs",
        "mths_since_last_delinq",
        "mths_since_recent_revol_delinq",
        "acc_now_delinq",
    ]

    # Initialize delinquency score
    df["delinquency_score"] = 0

    # Add points for delinquencies in the last 2 years
    if "delinq_2yrs" in df.columns:
        df["delinquency_score"] += df["delinq_2yrs"].clip(upper=5)

    # Subtract points for time since last delinquency (longer is better)
    if "mths_since_last_delinq" in df.columns:
        df["recent_delinq_bin"] = pd.cut(
            df["mths_since_last_delinq"].fillna(999),
            bins=[0, 6, 12, 24, float("inf")],
            labels=[3, 2, 1, 0],
            include_lowest=True,
        )
        df["delinquency_score"] += df["recent_delinq_bin"].astype(float)

    # Add extra weight for current delinquencies
    if "acc_now_delinq" in df.columns:
        df["delinquency_score"] += df["acc_now_delinq"].clip(upper=3) * 2

    # Cap the score at 10
    df["delinquency_score"] = df["delinquency_score"].clip(upper=10)

    return df


def create_debt_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that characterize the composition of the applicant's debt profile.

    Parameters:
        df (pd.DataFrame): Input DataFrame with account information.

    Returns:
        pd.DataFrame: DataFrame with new debt composition features.
    """
    # Installment to revolving debt ratio
    if "total_bal_il" in df.columns and "revol_bal_final" in df.columns:
        df["inst_to_revol_ratio"] = df["total_bal_il"] / df["revol_bal_final"].clip(
            lower=1
        )
        df["inst_to_revol_ratio"] = df["inst_to_revol_ratio"].clip(upper=10)

        df["debt_composition_type"] = pd.cut(
            df["inst_to_revol_ratio"],
            bins=[0, 0.5, 2, 5, float("inf")],
            labels=[
                "revolving_heavy",
                "balanced",
                "installment_heavy",
                "installment_only",
            ],
            include_lowest=True,
        )

    # Mortgage burden features
    if "mort_acc" in df.columns and "total_acc" in df.columns:
        df["mortgage_ratio"] = df["mort_acc"] / df["total_acc"].clip(lower=1)
        df["has_mortgage"] = (df["mort_acc"] > 0).astype(int)

    return df


def create_account_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that measure recent credit account activity and patterns.

    Parameters:
        df (pd.DataFrame): Input DataFrame with account activity information.

    Returns:
        pd.DataFrame: DataFrame with new account activity features.
    """

    # Calculate ratio of recently opened accounts
    if "open_acc" in df.columns and "acc_open_past_24mths" in df.columns:
        df["recent_acc_ratio"] = df["acc_open_past_24mths"] / df["open_acc"].clip(
            lower=1
        )
        df["recent_acc_ratio"] = df["recent_acc_ratio"].clip(upper=1)

        # Flag borrowers with rapid account acquisition
        df["rapid_acc_acquisition"] = (df["recent_acc_ratio"] > 0.5).astype(int)

    # Calculate active account ratio
    if "open_acc" in df.columns and "total_acc" in df.columns:
        df["active_acc_ratio"] = df["open_acc"] / df["total_acc"].clip(lower=1)
        df["active_acc_ratio"] = df["active_acc_ratio"].clip(upper=1)

    return df


def create_loan_purpose_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Create risk-based features for different loan purposes based on
    historical default rates.

        Parameters:
            df (pd.DataFrame): Input DataFrame with loan purpose information

        Returns:
            pd.DataFrame: DataFrame with purpose risk features (one-hot encoded)
    """
    # Risk mapping based on industry knowledge
    purpose_risk_map = {
        "debt_consolidation": 2,
        "credit_card": 2,
        "home_improvement": 1,
        "major_purchase": 2,
        "medical": 3,
        "car": 1,
        "small_business": 4,
        "moving": 3,
        "vacation": 4,
        "house": 1,
        "wedding": 3,
        "renewable_energy": 2,
        "educational": 3,
        "other": 3,
    }

    if "purpose" in df.columns:
        # Create risk score
        df["purpose_risk_score"] = df["purpose"].map(purpose_risk_map)
        df["purpose_risk_score"] = df["purpose_risk_score"].fillna(3)

        # Risk categories
        risk_category_map = {1: "low", 2: "medium", 3: "high", 4: "very_high"}
        df["purpose_risk_category"] = df["purpose_risk_score"].map(risk_category_map)

        # Create one-hot encoding for tree models
        df["purpose_high_risk"] = (df["purpose_risk_score"] >= 3).astype(int)

    return df


def create_combined_hardship_risk(df):
    """
    Optional: Combine hardship indicators into a single binary flag.
    """
    df["hardship_risk_flag"] = (
        (df["has_hardship"] == 1)
        & (df["was_late_before_hardship"] == 1)
        & (df["hardship_dpd_filled"] > 30)
    ).astype(int)
    return df