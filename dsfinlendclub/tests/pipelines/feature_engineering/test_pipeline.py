"""
  Unit tests for the feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from kedro.config import OmegaConfigLoader
from pathlib import Path
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets.pandas import CSVDataset, ParquetDataset
from kedro.runner import SequentialRunner
from dsfinlendclub.pipelines.feature_engineering.pipeline import create_pipeline
from ydata_profiling import ProfileReport

# Load configurations
conf_path = Path("conf")
conf_loader = OmegaConfigLoader(conf_source=str(conf_path), env="local")
params = conf_loader["parameters"]

# Load configurations
conf_loader = OmegaConfigLoader(conf_source=str(conf_path), env="local")
catalog_conf = conf_loader["catalog"]  # Load the catalog config

# Get parameters for testing
columns_to_drop = params["columns_to_drop"]
pub_rec_strategy = params["pub_rec_strategy"]


def test_feature_engineering_pipeline():
    """
    Test the complete feature engineering pipeline with a synthetic dataset
    that matches the output format from the data_processing pipeline.
    """
    # Create sample processed data matching the output of data_processing
    # Note: All string values are lowercase, matching the preprocessing transformations
    intermediate_data = pd.DataFrame({
        # Core loan features
        "loan_amnt": [10000, 15000, 20000, 5000, 30000],
        "funded_amnt": [10000, 15000, 20000, 5000, 30000],
        "installment": [300.45, 450.67, 600.89, 150.25, 900.75],

        # Income and DTI features
        "annual_inc": [50000, 75000, 100000, 35000, 150000],
        "annual_inc_joint": [80000, np.nan, 150000, np.nan, 200000],
        "dti": [15.5, 20.3, 25.7, 12.8, 18.4],
        "dti_joint": [18.2, np.nan, 22.4, np.nan, 19.5],

        # Credit score features
        "fico_range_low": [680, 700, 720, 650, 750],
        "fico_range_high": [690, 710, 730, 660, 760],

        # Loan terms - note: term is an integer after preprocessing
        "term": [36, 36, 60, 36, 60],  # Converted from "XX months" in preprocessing
        "int_rate": [10.5, 12.3, 15.7, 9.8, 7.2],  # Converted from percentage string
        "grade": ["a", "b", "c", "d", "a"],  # Lowercase
        "sub_grade": ["a3", "b2", "c1", "d4", "a1"],  # Lowercase

        # Employment - preprocessing converts to numeric in feature engineering
        "emp_length": ["1 year", "5 years", "10+ years", "< 1 year", "8 years"],

        # Dates - preprocessing converts to datetime
        "issue_d": pd.to_datetime(
            pd.Series(["Jan-2019", "Jan-2019", "Feb-2020", "Mar-2018", "Dec-2020"]), format="%b-%Y"),
        "earliest_cr_line": pd.to_datetime(
            pd.Series(["Jan-2010", "Jan-2008", "Feb-2005", "May-2012", "Jun-2000"]), format="%b-%Y"),
        "last_pymnt_d": pd.to_datetime(
            pd.Series(["Jun-2021", "Jul-2021", "Aug-2021", "Sep-2020", np.nan]), format="%b-%Y"),
        "sec_app_earliest_cr_line": pd.to_datetime(
            pd.Series([np.nan, np.nan, "Mar-2007", np.nan, "Jan-2005"]), format="%b-%Y"),

        # Categorical features - lowercase and spaces instead of underscores
        "purpose": ["debt consolidation", "credit card", "home improvement", "medical", "small business"],
        "home_ownership": ["rent", "own", "mortgage", "other", "own"],

        # Credit history
        "delinq_2yrs": [0, 1, 0, 2, 0],
        "inq_last_6mths": [1, 2, 0, 3, 1],
        "inq_last_12m": [2, 3, 1, 5, 2],
        "mths_since_last_delinq": [np.nan, 24, np.nan, 12, 36],
        "open_acc": [5, 8, 12, 4, 10],
        "pub_rec": [0, 0, 1, 0, 0],
        "revol_util": [45.2, 62.1, 33.5, 78.6, 28.3],  # Numeric, not percentage string
        "total_acc": [15, 20, 25, 8, 30],
        "open_acc_6m": [1, 2, 0, 2, 1],
        "acc_open_past_24mths": [3, 4, 2, 3, 1],

        # Joint application features
        "is_joint_app": [0, 0, 1, 0, 1],  # Binary flag
        "verification_status": ["verified", "not verified", "source verified", "verified", "not verified"],  # Lowercase
        "verification_status_joint": [np.nan, np.nan, "verified", np.nan, "source verified"],  # Lowercase

        # Hardship features
        "hardship_flag": ["n", "y", "n", "y", "n"],  # Lowercase
        "hardship_dpd": [np.nan, 30, np.nan, 15, np.nan],
        "hardship_loan_status": [np.nan, "late (31-120 days)", np.nan, "late (16-30 days)", np.nan],  # Lowercase

        # Additional credit metrics
        "revol_bal": [15000, 20000, 25000, 8000, 35000],
        "revol_bal_joint": [np.nan, np.nan, 35000, np.nan, 50000],
        "mths_since_last_record": [np.nan, 24, 36, np.nan, 60],
        "initial_list_status": ["w", "f", "w", "f", "w"],  # Lowercase
        "tot_coll_amt": [0, 500, 0, 1200, 0],
        "tot_cur_bal": [45000, 60000, 80000, 25000, 120000],
        "total_rev_hi_lim": [25000, 35000, 50000, 15000, 70000],
        "mths_since_recent_revol_delinq": [np.nan, 18, np.nan, 6, 48],
        "mths_since_recent_bc": [12, 8, 24, 5, 18],
        "mths_since_recent_inq": [3, 1, 5, 0, 4],

        # More account details
        "open_il_12m": [1, 2, 0, 3, 0],
        "open_il_24m": [2, 3, 1, 4, 1],
        "total_bal_il": [20000, 30000, 40000, 15000, 50000],
        "open_rv_12m": [1, 2, 0, 2, 1],
        "open_rv_24m": [2, 4, 1, 3, 2],
        "acc_now_delinq": [0, 0, 0, 1, 0],
        "mort_acc": [0, 1, 2, 0, 3],
        "num_actv_bc_tl": [2, 3, 4, 1, 5],
        "num_actv_rev_tl": [3, 4, 5, 2, 6],
        "num_rev_accts": [7, 9, 12, 5, 15],
        "pub_rec_bankruptcies": [0, 0, 0, 1, 0],
        "tax_liens": [0, 0, 0, 0, 0],
        "num_tl_op_past_12m": [2, 3, 1, 4, 1],
        "mths_since_last_major_derog": [np.nan, 36, np.nan, 12, 60],

        # Target variable already processed to binary
        "loan_status_binary": [0, 1, 0, 1, 0],

        # Loan status is preserved but already converted to lowercase
        "loan_status": ["fully paid", "charged off", "fully paid", "default", "fully paid"],
    })

    # Create a catalog that combines memory and filesystem datasets
    catalog_dict = {
        # Input data in memory
        "intermediate_data": MemoryDataset(intermediate_data),

        # Parameters in memory
        "params:columns_to_drop": MemoryDataset(params["columns_to_drop"]),
        "params:pub_rec_strategy": MemoryDataset(params["pub_rec_strategy"]),

        # # Intermediate datasets with persistent storage
        # "data_with_home_ownership_ordinal": CSVDataset(filepath="data/02_intermediate/test_home_ownership_ordinal.csv"),
        # "data_with_hardship_flag": CSVDataset(filepath="data/02_intermediate/test_hardship_flag.csv"),

        # Final outputs with persistent storage
        "engineered_features": ParquetDataset(filepath="data/04_feature/test_engineered_features.pq"),
        "tree_features": ParquetDataset(
            filepath="data/04_feature/test_tree_features.pq"),
        "regression_features": ParquetDataset(
            filepath="data/04_feature/test_regression_features.pq"),
    }

    # Create the catalog
    catalog = DataCatalog(catalog_dict)

    # Create and run pipeline
    pipeline = create_pipeline()
    runner = SequentialRunner()
    outputs = runner.run(pipeline, catalog)

    # Get engineered features
    engineered = catalog.load("engineered_features")
    tree_features = catalog.load("tree_features")
    regression_features = catalog.load("regression_features")
    # engineered = outputs["engineered_features"]
    # tree_features = outputs["tree_features"]
    # regression_features = outputs["regression_features"]

    # -------- Basic Validation Tests --------
    # Verify dataset types
    assert isinstance(engineered, pd.DataFrame), "Engineered features should be a DataFrame"
    assert isinstance(tree_features, pd.DataFrame), "Tree features should be a DataFrame"
    assert isinstance(regression_features, pd.DataFrame), "Regression features should be a DataFrame"

    # Verify row count preservation
    assert len(engineered) == len(intermediate_data), "Row count should be preserved"
    assert len(tree_features) == len(intermediate_data), "Row count should be preserved for tree features"
    assert len(regression_features) == len(intermediate_data), "Row count should be preserved for regression features"

    # -------- Categorical Encoding Tests --------
    # Test home ownership ordinal encoding
    assert "home_ownership_ordinal" in engineered.columns, "home_ownership_ordinal is missing"
    assert engineered["home_ownership_ordinal"].between(0, 3).all(), "home_ownership_ordinal should be between 0-3"

    # Test purpose encoding - ensure it works with lowercase and spaces
    assert any(col.startswith("purpose_") for col in engineered.columns), "Purpose encoding columns missing"

    # -------- Joint Application Feature Tests --------
    # Test joint income feature
    assert "annual_inc_final" in engineered.columns, "annual_inc_final is missing"
    assert engineered.loc[engineered["is_joint_app"] == 1, "annual_inc_final"].equals(
        engineered.loc[engineered["is_joint_app"] == 1, "annual_inc_joint"]), "annual_inc_final incorrect for joint applications"

    # Test joint DTI feature
    assert "dti_final" in engineered.columns, "dti_final is missing"
    assert engineered.loc[engineered["is_joint_app"] == 1, "dti_final"].equals(
          engineered.loc[engineered["is_joint_app"] == 1, "dti_joint"]), "dti_final incorrect for joint applications"

    # -------- Credit Score Feature Tests --------
    # Test FICO score features
    assert "fico_average" in engineered.columns, "fico_average is missing"
    assert "fico_risk_band" in engineered.columns, "fico_risk_band is missing"
    assert engineered["fico_average"].between(300, 850).all(), "fico_average should be between 300-850"

    # -------- Credit History Feature Tests --------
    # Test credit age feature
    assert "credit_age_months" in engineered.columns, "credit_age_months is missing"
    assert (engineered["credit_age_months"] >= 0).all(), "credit_age_months should be non-negative"

    # Test hardship features - check it works with lowercase 'y'/'n'
    assert "has_hardship" in engineered.columns, "has_hardship is missing"
    assert engineered["has_hardship"].isin([0, 1]).all(), "has_hardship should be binary"
    assert engineered.loc[intermediate_data["hardship_flag"] == "y", "has_hardship"].sum() == 2, "has_hardship should be 1 when hardship_flag is 'y'"

    # -------- Time-based Feature Tests --------
    # Check if time features were created
    if "issue_year" in engineered.columns:  # This checks if the new suggested function was implemented
        assert "issue_month" in engineered.columns, "issue_month is missing"
        assert "issue_quarter" in engineered.columns, "issue_quarter is missing"
        assert engineered["issue_month"].between(1, 12).all(), "issue_month should be between 1-12"
        assert engineered["issue_quarter"].between(1, 4).all(), "issue_quarter should be between 1-4"

    # -------- Payment Feature Tests --------
    # Check payment features if implemented
    if "payment_to_income" in engineered.columns:  # This checks if the new suggested function was implemented
        assert (engineered["payment_to_income"] >= 0).all(), "payment_to_income should be non-negative"
        assert engineered["payment_to_income"].max() <= 100, "payment_to_income should be capped at 100"

    # -------- Loan Amount Feature Tests --------
    # Test loan amount features
    assert "loan_to_installment_ratio" in engineered.columns, "loan_to_installment_ratio is missing"
    assert engineered["loan_to_installment_ratio"].between(engineered["loan_to_installment_ratio"].quantile(0.01), engineered["loan_to_installment_ratio"].quantile(0.99)).all(), "loan_to_installment_ratio has outliers"

    # -------- Credit Utilization Feature Tests --------
    # Test revol_util features
    assert "revol_util_tree" in engineered.columns, "revol_util_tree is missing"
    assert "revol_util_reg" in engineered.columns, "revol_util_reg is missing"
    assert engineered["revol_util_reg"].notna().all(), "revol_util_reg should not have NaNs"

    # -------- Model-specific Feature Tests --------
    # Test tree-based model features
    assert "emp_length_clean_tree" in tree_features.columns, "emp_length_clean_tree is missing in tree features"
    assert tree_features["emp_length_clean_tree"].notna().all(), "emp_length_clean_tree should not have NaNs"

    # Test regression model features
    assert "emp_length_clean_reg" in regression_features.columns, "emp_length_clean_reg is missing in regression " \
                                                                  "features"
    assert regression_features["emp_length_clean_reg"].notna().all(), "emp_length_clean_reg should not have NaNs"

    # -------- Test Hardship Features for Lowercase --------
    # Ensure the hardship loan status late detection works with lowercase
    assert "was_late_before_hardship" in engineered.columns, "was_late_before_hardship is missing"
    assert engineered.loc[intermediate_data["hardship_loan_status"].str.contains("late", case=False, na=False)].shape[0] == 2, "Two rows should have 'late' in hardship_loan_status"
    assert engineered.loc[intermediate_data["hardship_loan_status"].str.contains("late", case=False, na=False), "was_late_before_hardship"].sum() == 2, "was_late_before_hardship should be 1 for all rows with 'late' in hardship_loan_status"

    # -------- Test Initial List Status with Lowercase --------
    assert "initial_list_status_flag" in engineered.columns, "initial_list_status_flag is missing"
    assert engineered.loc[intermediate_data["initial_list_status"] == "w", "initial_list_status_flag"].sum() == 3, "initial_list_status_flag should be 1 for rows with 'w'"

    # -------- Test Verification Status with Lowercase --------
    if "verification_status_final" in engineered.columns:
        for idx in intermediate_data.index:
            if intermediate_data.loc[idx, "is_joint_app"] == 1:
                expected = intermediate_data.loc[idx, "verification_status_joint"]
            else:
                expected = intermediate_data.loc[idx, "verification_status"]

            assert engineered.loc[idx, "verification_status_final"] == expected, f"verification_status_final incorrect for row {idx}"

    # -------- Evaluation Test --------
    # Test feature evaluation metrics
    metrics = outputs["feature_evaluation_metrics"]
    assert isinstance(metrics, dict), "Evaluation metrics should be a dictionary"
    assert "missing_rates" in metrics, "missing_rates is missing in evaluation metrics"
    assert "distributions" in metrics, "distributions is missing in evaluation metrics"

    # Test tree feature evaluation
    tree_metrics = outputs["tree_features_evaluation"]
    assert isinstance(tree_metrics, dict), "Tree evaluation metrics should be a dictionary"

    # Test regression feature evaluation
    reg_metrics = outputs["regression_features_evaluation"]
    assert isinstance(reg_metrics, dict), "Regression evaluation metrics should be a dictionary"

    # -------- Additional Tests for Consistency --------
    # Verify that all engineered features are numeric or categorical (no object types)
    object_cols = engineered.select_dtypes(include=['object']).columns
    assert len(object_cols) <= 3, f"Too many object columns remain: {list(object_cols)}"

    # Check for any remaining nulls in critical columns
    critical_cols = ["loan_amnt", "term", "int_rate", "annual_inc_final", "dti_final", "fico_average"]
    for col in critical_cols:
        if col in engineered.columns:
            assert engineered[col].notna().all(), f"{col} should not have nulls"

    # -------- Generate Reports for Visual Inspection --------
    # Save the reports for manual review of distributions
    ProfileReport(intermediate_data, title="Pre-Feature Engineering").to_file("data/08_reporting"
                                                                              "/pre_feature_engineering.html")
    ProfileReport(engineered, title="Post-Feature Engineering").to_file("data/08_reporting/post_feature_engineering"
                                                                        ".html")

    # Return success message
    print("✅ Feature engineering pipeline test completed successfully")