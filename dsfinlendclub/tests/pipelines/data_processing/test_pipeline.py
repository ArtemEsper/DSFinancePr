import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from dsfinlendclub.pipelines.data_processing.pipeline import create_pipeline
from ydata_profiling import ProfileReport


# - emp_title - decode first
# - application_type (?) use it for a flag

def test_data_processing_pipeline_runs():
    raw_data = pd.DataFrame({
        "id": [1, 1, 2],  # has to be deduplicated and deleted
        "member_id": [1, 3, 2],  # has to be deleted
        "url": ["https://lendingclub.com/browse/loanDetail.action?loan_id=167338079",
                "https://lendingclub.com/browse/loanDetail.action?loan_id=71016917",
                "https://lendingclub.com/browse/loanDetail.action?loan_id=71016917"],  # has to be deleted
        "title": ["Credit card refinancing", "Debt consolidation", "Home improvement"],  # has to be deleted
        "zip_code": ["115xx", "116xx", "117xx"],  # has to be deleted
        "policy_code": [1.0, 1.0, 1.0],  # has to be deleted
        "pymnt_plan": ["n", "n", "n"],  # has to be deleted
        "funded_amnt_inv": [1000, 1000, 2000],  # has to be deleted
        "next_pymnt_d": ["Jan-2019", "Jan-2019", "Feb-2020"],  # has to be deleted
        "Unnamed: 0.1": [123, 124, 125],  # has to be deleted
        "Unnamed: 0": [126, 126, 130],  # has to be deleted
        "loan_amnt": [1000, 1000, 2000],
        "annual_inc": [50000, 50000, 60000],
        "dti": [15.0, 15.0, 20.0],
        "int_rate": ["10.5%", "10.5%", "12.3%"],
        "term": ["36 months", "36 months", "60 months"],
        "emp_length": ["1 year", "n/a", "10+ years"],
        "issue_d": ["Jan-2019", "Jan-2019", "Feb-2020"],
        "earliest_cr_line": ["Jan-2019", "Jan-2019", "Feb-2020"],
        "purpose": ["credit_card", "debt_consolidation", "home_improvement"],
        "home_ownership": ["MORTGAGE", "ANY", "NONE"],
        "loan_status": ["Fully Paid", "Charged Off", "Issued"],
        "addr_state": ["CA", "TX", "FL"],
        "revol_util": ["47%", "45%", "49%"],
        "initial_list_status": ["w", "W", "f"],
        "last_pymnt_d": ["Jan-2019", "Jan-2019", "Feb-2020"],
        "last_credit_pull_d": ["Jan-2019", "Jan-2019", "Feb-2020"],
        "application_type": ["Individual", "Individual", "Joint App"],
        "verification_status_joint": ["Not Verified", "Source Verified", "Verified"],
        "sec_app_earliest_cr_line": ["Jan-2019", "Jan-2019", "Feb-2020"],
        "hardship_flag": ["Y", "N", "N"],
        "hardship_type": ["ST0650PV01", "ST0650PV02", "ST0650PV03"],
        "hardship_reason": ["Job loss", None, "Medical expenses"],
        "hardship_start_date": ["Jan-2020", None, "Mar-2020"],
        "hardship_end_date": ["Apr-2020", None, "Jun-2020"],
        "hardship_amount": [1000.0, None, 1500.0],
        "hardship_length": [3, None, 4],
        "deferral_term": [2, None, 3],
        "hardship_loan_status": ["Late (31-120 days)", None, "Current"],
        "hardship_payoff_balance_amount": [5000.0, None, 3200.0],
        "hardship_last_payment_amount": [150.0, None, 200.0],
        "orig_projected_additional_accrued_interest": [75.3, None, 45.0],
        "payment_plan_start_date": ["2020-03-01", None, "2020-06-15"],
        "debt_settlement_flag": ["N", "Y", "Y"]
    })

    catalog = DataCatalog({
        "raw_data": MemoryDataset(raw_data),
        "params:admin_columns_to_drop": MemoryDataset([
            "id", "member_id", "url", "title", "zip_code", "policy_code",
            "pymnt_plan", "funded_amnt_inv", "next_pymnt_d", "Unnamed: 0.1", "Unnamed: 0", "loan_status",
            "addr_state", "last_pymnt_d", "last_credit_pull_d", "hardship_type", "hardship_reason",
            "hardship_start_date", "hardship_end_date", "hardship_amount", "hardship_length", "deferral_term",
            "hardship_loan_status", "hardship_payoff_balance_amount", "hardship_last_payment_amount",
            "hardship_payoff_balance_amount", "hardship_last_payment_amount",
            "orig_projected_additional_accrued_interest", "payment_plan_start_date", "debt_settlement_flag"
        ]),
        "dedup_flag": MemoryDataset(),  # capture intermediate result
    })

    pipeline = create_pipeline()
    runner = SequentialRunner()
    output = runner.run(pipeline, catalog)

    processed = output["data_processing_output"]
    # Get flag instead of full deduped dataframe
    deduped_success = output["dedup_flag"]

    # Check outputs exist
    assert "data_processing_output" in output

    # Check deduplication BEFORE 'id' is dropped
    assert deduped_success is True

    assert isinstance(processed, pd.DataFrame)

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
    assert processed["term"].dtype in [int, "int32", "int64"]
    assert set(processed["term"].unique()) <= {36, 60}, "Unexpected values in 'term'"

    # emp_length
    assert "n/a" not in processed["emp_length"].fillna("").values

    # issue_d
    assert pd.api.types.is_datetime64_any_dtype(processed['issue_d'])
    assert processed['issue_d'].notnull().all()  # or .any() if you expect nulls

    # earliest_cr_line
    assert pd.api.types.is_datetime64_any_dtype(processed['earliest_cr_line'])
    assert processed['earliest_cr_line'].notnull().all()  # or .any() if you expect nulls

    # loan_status
    # Check only 3 valid statuses remain
    valid_statuses = {"Fully Paid", "Charged Off", "Default"}
    assert set(processed["loan_status"].unique()).issubset(valid_statuses)
    assert "loan_status_binary" in processed.columns
    assert set(processed["loan_status_binary"].unique()).issubset({0, 1})  # Check if binary column was created

    # purpose
    assert processed["purpose"].str.islower().all()
    assert processed["purpose"].str.contains("_").sum() == 0

    # home_ownership
    assert "home_ownership" in processed.columns
    assert processed["home_ownership"].str.islower().all()
    assert processed["home_ownership"].str.strip().eq(processed["home_ownership"]).all()
    assert not processed["home_ownership"].isin(["none", "any", "unknown"]).any()
    expected_categories = {"rent", "own", "mortgage", "other"}
    assert set(processed["home_ownership"].dropna().unique()).issubset(expected_categories)

    # revol_util
    assert "revol_util" in processed.columns
    assert pd.api.types.is_float_dtype(processed["revol_util"])
    assert processed["revol_util"].dropna().between(0, 200).all()

    # initial_list_status
    assert "initial_list_status" in processed.columns
    assert processed["initial_list_status"].str.strip().eq(processed["initial_list_status"]).all()
    assert processed["initial_list_status"].isin(["f", "w"]).all()

    # sec_app_earliest_cr_line
    if "sec_app_earliest_cr_line" in processed.columns:
        assert pd.api.types.is_datetime64_any_dtype(processed["sec_app_earliest_cr_line"])

    # fields to delete
    for col in ["id", "member_id", "url", "title", "zip_code", "policy_code",
                "pymnt_plan", "funded_amnt_inv", "next_pymnt_d", "Unnamed: 0.1", "Unnamed: 0", "loan_status",
                "addr_state", "last_pymnt_d", "last_credit_pull_d", "hardship_type", "hardship_reason",
                "hardship_start_date", "hardship_end_date", "hardship_amount", "hardship_length", "deferral_term",
                "hardship_loan_status", "hardship_payoff_balance_amount", "hardship_last_payment_amount",
                "hardship_payoff_balance_amount", "hardship_last_payment_amount",
                "orig_projected_additional_accrued_interest", "payment_plan_start_date", "debt_settlement_flag"]:
        assert col not in processed.columns, f"{col} was not dropped"

    # check if other fields exist
    for col in ["loan_amnt", "annual_inc", "dti", "term", "int_rate", "emp_length", "issue_d",
                "purpose", "home_ownership", "application_type", "verification_status_joint", "hardship_flag"]:
        assert col in processed.columns, f"Expected column {col} is missing"

    print(processed.head())

    ProfileReport(raw_data, title="Raw Data").to_file("data/08_reporting/raw_report.html")
    ProfileReport(processed, title="Processed Data").to_file("data/08_reporting/processed_report.html")
