"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.8
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple


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


def create_credit_age_feature(df):
    """
    Creates a unified feature 'earliest_cr_line_final' combining individual and joint applicants' credit line information,
    and calculates credit age in months.

    Parameters:
        df (pd.DataFrame): DataFrame with 'earliest_cr_line', 'sec_app_earliest_cr_line', 'issue_d', and 'is_joint_app'.

    Returns:
        pd.DataFrame: Updated DataFrame with 'earliest_cr_line_final' and 'credit_age_months'.
    """
    df["earliest_cr_line_final"] = df["earliest_cr_line"]
    if "is_joint_app" in df.columns and "sec_app_earliest_cr_line" in df.columns:
        joint_mask = df["is_joint_app"] == 1
        df.loc[joint_mask, "earliest_cr_line_final"] = df.loc[
            joint_mask, "sec_app_earliest_cr_line"
        ]

    # Drop original columns
    # df.drop(
    #     columns=[
    #         col
    #         for col in ["earliest_cr_line", "sec_app_earliest_cr_line"]
    #         if col in df.columns
    #     ],
    #     inplace=True,
    # )

    # Calculate credit age in months
    if "issue_d" in df.columns and "earliest_cr_line_final" in df.columns:
        # Using pd.Timedelta for more robust datetime calculations
        df["credit_age_months"] = (
            df["issue_d"] - df["earliest_cr_line_final"]
        ).dt.total_seconds() / (30 * 24 * 60 * 60)
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
    Modified to work with preprocessed 'purpose' field that's already
    lowercase and space-normalized.
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

    return pd.get_dummies(df["purpose_cleaned"], prefix="purpose", dummy_na=True)


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


def create_model_specific_datasets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates separate datasets optimized for tree-based and regression models.

    Args:
        df: DataFrame with all engineered features

    Returns:
        tuple: (tree_features_df, regression_features_df)
    """
    tree_features = df.copy()
    regression_features = df.copy()

    # Map features to their model-specific versions
    feature_mapping = {
        "tree": {
            "emp_length_clean": "emp_length_clean_tree",
            "revol_util": "revol_util_tree",
        },
        "regression": {
            "emp_length_clean": "emp_length_clean_reg",
            "revol_util": "revol_util_reg",
        },
    }

    # Update tree dataset
    for orig, tree_col in feature_mapping["tree"].items():
        if tree_col in df.columns:
            tree_features[orig] = tree_features[tree_col]
            tree_features = tree_features.drop(columns=[tree_col])

    # Update regression dataset
    for orig, reg_col in feature_mapping["regression"].items():
        if reg_col in df.columns:
            regression_features[orig] = regression_features[reg_col]
            regression_features = regression_features.drop(columns=[reg_col])

    return tree_features, regression_features
