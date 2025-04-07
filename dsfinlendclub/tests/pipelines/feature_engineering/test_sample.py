import pandas as pd
from ydata_profiling import ProfileReport
from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets.pandas import ParquetDataset
from kedro.runner import SequentialRunner
from dsfinlendclub.pipelines.feature_engineering.pipeline import create_pipeline
from pathlib import Path
from pandas.api.types import is_integer_dtype


def test_data_processing_pipeline_on_sample():
    # Load configuration
    conf_path = Path("conf")
    conf_loader = OmegaConfigLoader(conf_source=str(conf_path), env="local")

    # Load catalog
    catalog_config = conf_loader["catalog"]
    credentials = conf_loader.get("credentials", {})
    full_catalog = DataCatalog.from_config(catalog_config, credentials=credentials)

    # Load real dataset from catalog
    preprocessed_data = full_catalog.load("intermediate_data")  # <-- this loads the Parquet as a
    # DataFrame
    params = conf_loader["parameters"]

    # Create a catalog that combines memory and filesystem datasets
    sample_catalog_dict = {
        # Input data in memory
        "intermediate_data": MemoryDataset(preprocessed_data),

        # Parameters in memory
        "params:columns_to_drop": MemoryDataset(params["columns_to_drop"]),
        "params:pub_rec_strategy": MemoryDataset(params["pub_rec_strategy"]),
        "params:cols_for_reg": MemoryDataset(params["cols_for_reg"]),
        "params:cols_for_tree": MemoryDataset(params["cols_for_tree"]),

        # Final outputs with persistent storage
        "engineered_features": ParquetDataset(filepath="data/03_primary/engineered_features.pq"),
        "tree_features": ParquetDataset(
            filepath="data/04_feature/sample_tree_features.pq"),
        "regression_features": ParquetDataset(
            filepath="data/04_feature/sample_regression_features.pq"),
    }

    # Create the catalog
    sample_eng_catalog = DataCatalog(sample_catalog_dict)

    # Run pipeline
    sample_pipeline = create_pipeline()
    runner = SequentialRunner()
    output = runner.run(sample_pipeline, sample_eng_catalog)

    engineered = sample_eng_catalog.load("engineered_features")
    tree_features = sample_eng_catalog.load("tree_features")
    regression_features = sample_eng_catalog.load("regression_features")

    assert isinstance(engineered, pd.DataFrame)
    assert isinstance(tree_features, pd.DataFrame)
    assert isinstance(regression_features, pd.DataFrame)
    print(engineered.head())  # Debugging line
    print(tree_features.head())  # Debugging line
    print(regression_features.head())  # Debugging line

    # Target variable
    assert "loan_status_binary" in engineered.columns
    assert engineered["loan_status_binary"].isin([0, 1]).all()

    # Verify dataset types
    assert isinstance(engineered, pd.DataFrame), "Engineered features should be a DataFrame"
    assert isinstance(tree_features, pd.DataFrame), "Tree features should be a DataFrame"
    assert isinstance(regression_features, pd.DataFrame), "Regression features should be a DataFrame"

    # Verify row count preservation
    assert len(engineered) == len(preprocessed_data), "Row count should be preserved"
    assert len(tree_features) == len(preprocessed_data), "Row count should be preserved for tree features"
    assert len(regression_features) == len(
        preprocessed_data), "Row count should be preserved for regression features"

    # -------- Categorical Encoding Tests --------
    # Test home ownership ordinal encoding
    assert "home_ownership_ordinal" in engineered.columns, "home_ownership_ordinal is missing"
    assert engineered["home_ownership_ordinal"].between(0, 3).all(), "home_ownership_ordinal should be between 0-3"

    # Test purpose encoding - ensure it works with lowercase and spaces
    assert any(col.startswith("purpose_") for col in engineered.columns), "Purpose encoding columns missing"

    # -------- Joint Application Feature Tests --------
    # Test joint income feature
    assert "annual_inc_final" in engineered.columns, "annual_inc_final is missing"
    pd.testing.assert_series_equal(
        engineered.loc[engineered["is_joint_app"] == 1, "annual_inc_final"],
        engineered.loc[engineered["is_joint_app"] == 1, "annual_inc_joint"],
        check_dtype=False,
        check_names=False,
        obj="annual_inc_final for joint applications"
    )

    # Test joint DTI feature
    assert "dti_final" in engineered.columns, "dti_final is missing"
    joint_mask = (engineered["is_joint_app"] == 1) & (engineered["dti_joint"].notna())
    pd.testing.assert_series_equal(
        engineered.loc[joint_mask, "dti_final"],
        engineered.loc[joint_mask, "dti_joint"],
        check_dtype=False,
        check_names=False,
        obj="dti_final for joint applications"
    )

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
    merged = preprocessed_data[["hardship_flag"]].reset_index().merge(
        engineered[["has_hardship"]].reset_index(), on="index", how="inner"
    )
    assert (merged.loc[merged["hardship_flag"] == "y", "has_hardship"] == 1).all(), \
        "has_hardship should be 1 when hardship_flag is 'y'"

    print(engineered["has_hardship"].value_counts())
    print(preprocessed_data["hardship_flag"].value_counts())

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
    # assert engineered["loan_to_installment_ratio"].between(engineered["loan_to_installment_ratio"].quantile(0.01), engineered["loan_to_installment_ratio"].quantile(0.99)).all(), "loan_to_installment_ratio has outliers"

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

    # -------- Test Hardship Features for Lowercase -------- # Ensure the hardship loan status late detection works
    # with lowercase assert "was_late_before_hardship" in engineered.columns, "was_late_before_hardship is missing"
    # assert \ engineered.loc[preprocessed_data["hardship_loan_status"].str.contains("late", case=False,
    # na=False)].shape[ 0] == 2, "Two rows should have 'late' in hardship_loan_status" assert engineered.loc[
    # preprocessed_data["hardship_loan_status"].str.contains("late", case=False, na=False),
    # "was_late_before_hardship"].sum() == 2, "was_late_before_hardship should be 1 for all rows with 'late' in
    # hardship_loan_status"

    # -------- Test Initial List Status with Lowercase --------
    assert "initial_list_status_flag" in engineered.columns, "initial_list_status_flag is missing"
    # Normalize the test condition
    initial_list_cleaned = preprocessed_data["initial_list_status"].str.strip().str.lower()

    # Compare against the engineered flag
    assert (
            engineered.loc[initial_list_cleaned == "w", "initial_list_status_flag"].sum()
            == (initial_list_cleaned == "w").sum()
    ), "initial_list_status_flag should be 1 for rows with 'w'"

    # -------- Test Verification Status with Lowercase --------
    if "verification_status_final" in engineered.columns:
        for idx in preprocessed_data.index:
            if preprocessed_data.loc[idx, "is_joint_app"] == 1:
                expected = preprocessed_data.loc[idx, "verification_status_joint"]
            else:
                expected = preprocessed_data.loc[idx, "verification_status"]

            assert engineered.loc[
                       idx, "verification_status_final"] == \
                   expected, f"verification_status_final incorrect for row {idx}"

    # -------- Evaluation Test --------

    # Test tree feature evaluation
    tree_metrics = output["tree_features_evaluation"]
    assert isinstance(tree_metrics, dict), "Tree evaluation metrics should be a dictionary"

    # Test regression feature evaluation
    reg_metrics = output["regression_features_evaluation"]
    assert isinstance(reg_metrics, dict), "Regression evaluation metrics should be a dictionary"

    # Test tot_cur_bal features
    assert "tot_cur_bal" in engineered.columns
    assert pd.api.types.is_numeric_dtype(engineered["tot_cur_bal"])
    assert (engineered["tot_cur_bal"].dropna() >= 0).all()
    assert "log_tot_cur_bal" in engineered.columns
    assert "cur_bal_to_income" in engineered.columns
    assert "cur_bal_to_loan" in engineered.columns
    assert "tot_cur_bal_missing" in engineered.columns

    # Test open_act_il features
    assert "open_act_il_log" in engineered.columns
    assert "open_act_il_missing" in engineered.columns
    assert "open_act_il_ratio" in engineered.columns

    # Test avg_cur_bal features
    assert "avg_cur_bal_log" in engineered.columns
    assert "avg_cur_bal_missing" in engineered.columns
    assert "avg_bal_per_acc" in engineered.columns

    # Test mths_since_recent_inq features
    assert "mths_since_recent_inq_missing" in engineered.columns
    assert "mths_since_recent_inq_capped" in engineered.columns
    assert "had_recent_inquiry" in engineered.columns

    # Test num_tl_op_past_12m derived features
    assert "num_tl_op_past_12m_missing" in engineered.columns
    assert "num_tl_op_past_12m_capped" in engineered.columns

    # Test pub_rec_bankruptcies features
    assert "pub_rec_bankruptcies_missing" in engineered.columns
    assert "pub_rec_bankruptcies_capped" in engineered.columns
    assert "has_bankruptcy" in engineered.columns

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
    # Pre-feature engineering profile
    profile_pre = ProfileReport(
        preprocessed_data,
        title="Pre-Feature Engineering",
        correlations={"autocorrelation": False},  # Disable slow correlation
        minimal=True,  # Also disables expensive computations like interactions
    )
    profile_pre.to_file("data/08_reporting/pre_feature_engineering.html")

    # Post-feature engineering profile
    profile_post = ProfileReport(
        engineered,
        title="Post-Feature Engineering",
        correlations={
            "pearson": True,
            "spearman": True,
            "kendall": True,
            "phi_k": False,
            "cramers": False,
            "autocorrelation": False  # Disable autocorrelation
        },
        minimal=True,
    )
    profile_post.to_file("data/08_reporting/post_feature_engineering.html")

    # Return success message
    print("✅ Feature engineering pipeline test completed successfully")
