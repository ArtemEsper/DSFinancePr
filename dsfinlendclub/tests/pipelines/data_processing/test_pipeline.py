import pandas as pd
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from kedro.io import MemoryDataset
from dsfinlendclub.pipelines.data_processing.pipeline import create_pipeline
from ydata_profiling import ProfileReport


# - title
# - zip_code
# - emp_title
# - policy_code
# - pymnt_plan
# - application_type
# - funded_amnt_inv
# - next_pymnt_d
# - Unnamed: 0.1
# - Unnamed: 0

def test_data_processing_pipeline_runs():
    raw_data = pd.DataFrame({
        "id": [1, 1, 2],  # has to be deduplicated and deleted
        "member_id": [1, 3, 2, 5],  # has to be deleted
        "url": ["https://lendingclub.com/browse/loanDetail.action?loan_id=167338079",
                "https://lendingclub.com/browse/loanDetail.action?loan_id=71016917"],  # has to be deleted
        "title": ["Credit card refinancing", "Debt consolidation", "Home improvement"],  # has to be deleted
        "zip_code": ["115xx", "116xx", "117xx"],  # has to be deleted
        "emp_title": ["cashier", "ABM", ""],
        "loan_amnt": [1000, 1000, 2000],
        "annual_inc": [50000, 50000, 60000],
        "dti": [15.0, 15.0, 20.0],
        "int_rate": ["10.5%", "10.5%", "12.3%"],
        "term": ["36 months", "36 months", "60 months"],
        "loan_status": ["Fully Paid", "Fully Paid", "Charged Off"],
        "emp_length": ["1 year", "n/a", "10+ years"],
        "issue_d": ["Jan-2019", "Jan-2019", "Feb-2020"],
        "earliest_cr_line": ["Jan-2019", "Jan-2019", "Feb-2020"],
        "last_credit_pull_d": ["Jan-2019", "Jan-2019", "Feb-2020"],
        "purpose": ["credit_card", "debt_consolidation", "home_improvement"]
    })

    catalog = DataCatalog({
        "raw_data": MemoryDataset(raw_data),
        "params:admin_columns_to_drop": MemoryDataset(["some_unused_column"])
    })

    pipeline = create_pipeline()
    runner = SequentialRunner()
    output = runner.run(pipeline, catalog)

    assert "data_processing_output" in output
    processed = output["data_processing_output"]
    assert isinstance(processed, pd.DataFrame)

    # id
    assert processed["id"].nunique() == 2
    assert processed.shape[0] == 2

    # loan_amnt
    assert "loan_amnt" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed['loan_amnt'])
    assert (processed['loan_amnt'] > 0).all()

    # annual_inc
    assert "annual_inc" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed['annual_inc'])
    assert (processed['annual_inc'] > 0).all()
    assert processed["annual_inc"].max() < 1000000  # or a threshold based on EDA

    # dti
    assert "dti" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["dti"])
    assert (processed["dti"] >= 0).all()
    assert (processed["dti"] <= 100).all()  # Adjust threshold if needed

    # int_rate
    assert processed["int_rate"].dtype == float

    # term
    assert processed["term"].dtype == object  # (technically str in pandas)
    assert set(processed["term"].unique()) == {"36 months", "60 months"}

    # emp_length
    assert "n/a" not in processed["emp_length"].fillna("").values

    # issue_d
    assert pd.api.types.is_datetime64_any_dtype(processed['issue_d'])
    assert processed['issue_d'].notnull().all()  # or .any() if you expect nulls

    # earliest_cr_line
    assert pd.api.types.is_datetime64_any_dtype(processed['earliest_cr_line'])
    assert processed['earliest_cr_line'].notnull().all()  # or .any() if you expect nulls

    # last_credit_pull_d
    assert pd.api.types.is_datetime64_any_dtype(processed['last_credit_pull_d'])
    assert processed['last_credit_pull_d'].notnull().all()  # or .any() if you expect nulls

    # loan_status
    assert "loan_status" in processed.columns  # Column must exist
    expected_statuses = {"Fully Paid", "Charged Off", "Default"}
    actual_statuses = set(processed["loan_status"].dropna().unique())
    assert actual_statuses.issubset(expected_statuses)

    # purpose
    assert processed["purpose"].str.islower().all()
    assert processed["purpose"].str.contains("_").sum() == 0

    print(processed.head())

    ProfileReport(raw_data, title="Raw Data").to_file("raw_report.html")
    ProfileReport(processed, title="Processed Data").to_file("processed_report.html")
