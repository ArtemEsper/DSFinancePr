import pandas as pd
import numpy as np
from kedro.config import OmegaConfigLoader
from pathlib import Path
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from dsfinlendclub.pipelines.feature_engineering.pipeline import create_pipeline
from ydata_profiling import ProfileReport

conf_path = Path("conf")
conf_loader = OmegaConfigLoader(conf_source=str(conf_path), env="local")
params = conf_loader["parameters"]

columns_to_drop = params["columns_to_drop"]
pub_rec_strategy = params["pub_rec_strategy"]


def test_feature_engineering_pipeline():
    # Create sample processed data matching the output of data_processing
    intermediate_data = pd.DataFrame({
        # Core loan features
        "loan_amnt": [10000, 15000, 20000],
        "funded_amnt": [10000, 15000, 20000],
        "installment": [300.45, 450.67, 600.89],

        # Income and DTI features
        "annual_inc": [50000, 75000, 100000],
        "annual_inc_joint": [80000, np.nan, 150000],
        "dti": [15.5, 20.3, 25.7],
        "dti_joint": [18.2, np.nan, 22.4],

        # Credit score features
        "fico_range_low": [680, 700, 720],
        "fico_range_high": [690, 710, 730],

        # Loan terms
        "term": [36, 36, 60],
        "int_rate": [10.5, 12.3, 15.7],
        "grade": ["A", "B", "C"],
        "sub_grade": ["A3", "B2", "C1"],

        # Employment
        "emp_length": ["1 year", "5 years", "10+ years"],

        # Dates
        # Dates
        "issue_d": pd.to_datetime(
            pd.Series(["Jan-2019", "Jan-2019", "Feb-2020"]),
            format="%b-%Y"
        ),
        "earliest_cr_line": pd.to_datetime(
            pd.Series(["Jan-2019", "Jan-2019", "Feb-2020"]),
            format="%b-%Y"
        ),

        # Categorical features
        "purpose": ["debt_consolidation", "credit_card", "home_improvement"],
        "home_ownership": ["rent", "own", "mortgage"],

        # Credit history
        "delinq_2yrs": [0, 1, 0],
        "inq_last_6mths": [1, 2, 0],
        "open_acc": [5, 8, 12],
        "pub_rec": [0, 0, 1],
        "revol_util": [45.2, 62.1, 33.5],

        # Joint application features
        "is_joint_app": [0, 0, 1],
        "verification_status": ["verified", "not verified", "source verified"],
        "verification_status_joint": [np.nan, np.nan, "verified"],

        # Hardship features
        "hardship_flag": ["N", "Y", "N"],
        "hardship_dpd": [np.nan, 30, np.nan],
        "hardship_loan_status": [np.nan, "Late (31-120 days)", np.nan],

        # Additional credit metrics
        "revol_bal": [15000, 20000, 25000],
        "revol_bal_joint": [np.nan, np.nan, 35000],
        "mths_since_last_record": [np.nan, 24, 36],
        "mths_since_last_major_derog": [np.nan, 12, 24]
    })

    # Create data catalog
    catalog = DataCatalog({
        "processed_data": MemoryDataset(intermediate_data),
        "params:columns_to_drop": MemoryDataset(params["columns_to_drop"]),
        "params:pub_rec_strategy": MemoryDataset(params["pub_rec_strategy"]),

    })

    # Create and run pipeline
    pipeline = create_pipeline()
    runner = SequentialRunner()
    outputs = runner.run(pipeline, catalog)

    # Get engineered features
    engineered = outputs["engineered_features"]

    # Basic validation
    assert isinstance(engineered, pd.DataFrame)

    # Test purpose encoding
    assert "purpose_debt_consolidation" in engineered.columns
    assert engineered["purpose_debt_consolidation"].isin([0, 1]).all()

    # Test homeownership encoding
    assert "home_ownership_ordinal" in engineered.columns
    assert engineered["home_ownership_ordinal"].between(0, 3).all()

    # Test FICO score features
    assert "fico_average" in engineered.columns
    assert "fico_risk_band" in engineered.columns
    assert engineered["fico_average"].between(300, 850).all()

    # Test joint features
    assert "annual_inc_final" in engineered.columns
    assert "dti_final" in engineered.columns
    assert engineered.loc[engineered["is_joint_app"] == 1, "annual_inc_final"].equals(
        engineered.loc[engineered["is_joint_app"] == 1, "annual_inc_joint"]
    )

    # Test credit history features
    assert "has_derogatory" in engineered.columns
    assert engineered["has_derogatory"].isin([0, 1]).all()

    # Test loan amount features
    assert "loan_to_installment_ratio" in engineered.columns
    assert engineered["loan_to_installment_ratio"].between(
        engineered["loan_to_installment_ratio"].quantile(0.01),
        engineered["loan_to_installment_ratio"].quantile(0.99)
    ).all()

    # Test evaluation metrics
    metrics = outputs["feature_evaluation_metrics"]
    assert isinstance(metrics, dict)
    assert "missing_rates" in metrics
    assert "distributions" in metrics

    assert isinstance(outputs["tree_features"], pd.DataFrame)
    assert isinstance(outputs["regression_features"], pd.DataFrame)

    # test categorical encoding outputs
    assert any(col.startswith("purpose_") for col in engineered.columns)

    # Validate ranges of engineered values
    assert engineered["income_log_reg"].notna().all()
    assert (engineered["term_normalized_reg"] >= 0).all()
    assert (engineered["fico_normalized_reg"] >= 0).all()

    # Generate reports
    ProfileReport(intermediate_data, title="Pre-Feature Engineering").to_file(
        "data/08_reporting/pre_feature_engineering.html"
    )
    ProfileReport(engineered, title="Post-Feature Engineering").to_file(
        "data/08_reporting/post_feature_engineering.html"
    )
