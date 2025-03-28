import pandas as pd
from ydata_profiling import ProfileReport
from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets.pandas import ParquetDataSet
from kedro.runner import SequentialRunner
from dsfinlendclub.pipelines.data_processing.pipeline import create_pipeline
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
    sample = full_catalog.load("sample")  # <-- this loads the CSV as a DataFrame
    params = conf_loader["parameters"]
    columns_to_drop = params["admin_columns_to_drop"]

    # Construct test-specific catalog with MemoryDatasets
    test_catalog = DataCatalog({
        "raw_data": MemoryDataset(sample),
        "params:admin_columns_to_drop": MemoryDataset(columns_to_drop),
        # "intermediate_data": ParquetDataSet(filepath="data/02_intermediate/preprocessed_sample_data.pq"),
        "dedup_flag": MemoryDataset(),  # will be filled by the pipeline
    })

    # Run pipeline
    pipeline = create_pipeline()
    runner = SequentialRunner()
    output = runner.run(pipeline, test_catalog)

    processed = output["intermediate_data"]
    assert isinstance(processed, pd.DataFrame)
    print(processed.head())  # Debugging line

    # Get deduplication flag
    deduped_success = output["dedup_flag"]
    # Check deduplication BEFORE 'id' is dropped
    assert deduped_success is True

    # Check outputs exist
    assert "intermediate_data" in output

    # check if fields exist
    for col in ["loan_amnt", "annual_inc", "dti", "term", "int_rate", "emp_length", "issue_d",
                "purpose", "home_ownership",
                "hardship_flag", "sec_app_earliest_cr_line", "earliest_cr_line", "hardship_loan_status",
                "initial_list_status", "revol_util", "annual_inc_joint", "dti_joint", "revol_bal", "revol_bal_joint",
                "hardship_dpd", "mths_since_last_record", "is_joint_app", "funded_amnt", "installment", "grade",
                "sub_grade", "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
                "mths_since_last_delinq", "open_acc", "pub_rec", "total_acc", "open_acc_6m",
                "mths_since_last_major_derog"]:
        assert col in processed.columns, f"Expected column {col} is missing"

    # fields to delete
    for col in columns_to_drop:
        assert col not in processed.columns, f"{col} was not dropped"

    # loan_amnt
    assert pd.api.types.is_numeric_dtype(processed['loan_amnt'])
    assert (processed['loan_amnt'] > 0).all()
    print(processed["loan_amnt"].head())

    # pub_rec_bankruptcies
    assert "pub_rec_bankruptcies" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["pub_rec_bankruptcies"])
    assert (processed["pub_rec_bankruptcies"].dropna() >= 0).all()

    # tax_liens
    assert "tax_liens" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["tax_liens"])
    assert (processed["tax_liens"].dropna() >= 0).all()

    # mths_since_recent_bc_dlq
    assert "mths_since_recent_bc_dlq" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["mths_since_recent_bc_dlq"])
    # allow NaNs for this field
    assert processed["mths_since_recent_bc_dlq"].dropna().between(0, 300).all()

    # mths_since_recent_inq
    assert "mths_since_recent_inq" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["mths_since_recent_inq"])
    assert (processed["mths_since_recent_inq"].dropna() >= 0).all()

    # mths_since_recent_revol_delinq
    assert "mths_since_recent_revol_delinq" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["mths_since_recent_revol_delinq"])
    # allow NaNs, check value range if present
    assert processed["mths_since_recent_revol_delinq"].dropna().between(0, 300).all()

    # avg_cur_bal
    assert "avg_cur_bal" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["avg_cur_bal"])
    assert (processed["avg_cur_bal"].dropna() >= 0).all()

    # num_tl_op_past_12m
    assert "num_tl_op_past_12m" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["num_tl_op_past_12m"])
    assert (processed["num_tl_op_past_12m"].dropna() >= 0).all()

    # acc_now_delinq
    assert "acc_now_delinq" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["acc_now_delinq"])
    assert (processed["acc_now_delinq"].dropna() >= 0).all()

    # tot_coll_amt
    assert "tot_coll_amt" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["tot_coll_amt"])
    assert (processed["tot_coll_amt"].dropna() >= 0).all()

    # tot_cur_bal
    assert "tot_cur_bal" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["tot_cur_bal"])
    assert (processed["tot_cur_bal"].dropna() >= 0).all()

    # open_act_il
    assert "open_act_il" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["open_act_il"])
    assert (processed["open_act_il"].dropna() >= 0).all()

    # open_il_12m
    assert "open_il_12m" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["open_il_12m"])
    assert (processed["open_il_12m"].dropna() >= 0).all()

    # open_acc_6m
    assert pd.api.types.is_numeric_dtype(processed["open_acc_6m"])
    assert (processed["open_acc_6m"].dropna() >= 0).all()

    # is_joint_app
    assert pd.api.types.is_integer_dtype(processed["is_joint_app"]), "'is_joint_app' should be an integer (0/1)"
    # Values are only 0 or 1
    assert set(processed["is_joint_app"].dropna().unique()) <= {0, 1}, "Unexpected values in 'is_joint_app'"

    # annual_inc
    assert pd.api.types.is_numeric_dtype(processed['annual_inc'])
    assert (processed['annual_inc'] > 0).all()

    # annual_inc_joint
    assert pd.api.types.is_numeric_dtype(processed["annual_inc_joint"])
    assert (processed['annual_inc_joint'].dropna() > 0).all()
    assert not processed['annual_inc_joint'].dropna().empty

    # Check column exists before validating its values
    if "annual_inc" in processed.columns and not processed["annual_inc"].empty:
        annual_inc_cap = sample["annual_inc"].quantile(0.99)
        assert processed["annual_inc"].max() <= annual_inc_cap

    if "annual_inc_joint" in processed.columns and not processed["annual_inc_joint"].dropna().empty:
        annual_inc_joint_cap = sample["annual_inc_joint"].quantile(0.99)
        assert processed["annual_inc_joint"].dropna().max() <= annual_inc_joint_cap

    # dti
    assert pd.api.types.is_numeric_dtype(processed["dti"])
    assert (processed["dti"] >= 0).all()
    assert (processed["dti"] <= 100).all()  # Adjust threshold if needed

    # int_rate
    assert processed["int_rate"].dtype == float

    # term
    assert is_integer_dtype(processed["term"]), "'term' should be an integer type"
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
    assert "loan_status_binary" in processed.columns
    assert processed["loan_status_binary"].dropna().isin([0, 1]).all()  # Check binary target created

    # purpose
    assert processed["purpose"].str.islower().all()
    assert processed["purpose"].str.contains("_").sum() == 0

    # home_ownership
    assert processed["home_ownership"].str.islower().all()
    assert processed["home_ownership"].str.strip().eq(processed["home_ownership"]).all()
    assert not processed["home_ownership"].isin(["none", "any", "unknown"]).any()
    expected_categories = {"rent", "own", "mortgage", "other"}
    assert set(processed["home_ownership"].dropna().unique()).issubset(expected_categories)

    # revol_util
    if "revol_util" in processed.columns:
        print(processed["revol_util"].head())  # Debug
        assert pd.api.types.is_float_dtype(processed["revol_util"]), "Expected revol_util to be float"
        assert processed["revol_util"].dropna().between(0, 100).all()
    else:
        raise AssertionError("Expected 'revol_util' column is missing in processed dataset")

    assert processed["revol_util"].dropna().between(0, 100).all()

    # initial_list_status
    assert processed["initial_list_status"].str.strip().eq(processed["initial_list_status"]).all()
    assert processed["initial_list_status"].isin(["f", "w"]).all()

    # sec_app_earliest_cr_line
    if "sec_app_earliest_cr_line" in processed.columns:
        assert pd.api.types.is_datetime64_any_dtype(processed["sec_app_earliest_cr_line"])

    # dti_joint
    assert pd.api.types.is_numeric_dtype(processed["dti_joint"])
    assert processed["dti_joint"].dropna().between(0, 100).all()

    # verification_status and verification_status_joint

    expected_values = {"not verified", "source verified", "verified"}
    for col in ["verification_status", "verification_status_joint"]:
        if col in processed.columns:
            invalid_values = processed[col].dropna()[~processed[col].dropna().isin(expected_values)].unique()
            print(f"Unexpected values in {col}: {invalid_values}")
            assert processed[col].dropna().isin(expected_values).all()

    # revol_bal and revol_bal_joint
    for col in ["revol_bal", "revol_bal_joint"]:
        if col in processed.columns:
            assert pd.api.types.is_numeric_dtype(processed[col])
            # This assumes the capping worked in the pipeline
            assert processed[col].max() <= sample[col].quantile(0.99)

    # hardship_dpd
    assert pd.api.types.is_numeric_dtype(processed["hardship_dpd"])

    # mths_since_last_record
    assert pd.api.types.is_numeric_dtype(processed["mths_since_last_record"])
    # Just check that no exceptions occur due to mixed types
    processed["mths_since_last_record"].dropna().apply(lambda x: isinstance(x, (int, float)))
    if "mths_since_last_record" in processed.columns and not processed["mths_since_last_record"].empty:
        # Validate type and expect NaNs only if column exists and has rows
        assert pd.api.types.is_numeric_dtype(processed["mths_since_last_record"])
        assert processed["mths_since_last_record"].isna().any(), "Expected some NaNs in mths_since_last_record"
    else:
        print("Column 'mths_since_last_record' is missing or empty — skipping NaN test.")

    # assert processed["mths_since_last_record"].isna().any()  # Expect NaNs at this point

    # funded_amnt
    assert pd.api.types.is_numeric_dtype(processed["funded_amnt"])
    assert (processed["funded_amnt"] > 0).all(), "'funded_amnt' should be positive"

    # installment
    assert pd.api.types.is_numeric_dtype(processed["installment"])
    assert (processed["installment"] > 0).all(), "'installment' should be positive"

    # grade
    assert processed["grade"].dropna().str.isalpha().all(), "'grade' should contain alphabetic values"
    assert processed["grade"].str.len().eq(1).all(), "'grade' should be a single letter"

    # sub_grade
    assert processed["sub_grade"].dropna().str.len().between(2, 3).all(), "'sub_grade' should be 2-3 characters long"
    assert processed["sub_grade"].str[0].isin(processed["grade"]).all(), "'sub_grade' prefix should match 'grade'"

    # delinq_2yrs
    assert pd.api.types.is_numeric_dtype(processed["delinq_2yrs"])
    assert (processed["delinq_2yrs"] >= 0).all(), "'delinq_2yrs' should be non-negative"

    # fico_range_low
    assert pd.api.types.is_numeric_dtype(processed["fico_range_low"])
    assert (processed["fico_range_low"] >= 300).all()

    # fico_range_high
    assert pd.api.types.is_numeric_dtype(processed["fico_range_high"])
    assert (processed["fico_range_high"] >= processed["fico_range_low"]).all()

    # inq_last_6mths
    assert pd.api.types.is_numeric_dtype(processed["inq_last_6mths"])
    assert (processed["inq_last_6mths"] >= 0).all()

    # mths_since_last_delinq
    assert pd.api.types.is_float_dtype(processed["mths_since_last_delinq"])
    assert processed["mths_since_last_delinq"].isna().sum() >= 0  # allow missing

    # open_acc
    assert pd.api.types.is_numeric_dtype(processed["open_acc"])
    assert (processed["open_acc"] >= 0).all()

    # pub_rec
    assert pd.api.types.is_numeric_dtype(processed["pub_rec"])
    assert processed["pub_rec"].dropna().ge(0).all()  # All non-missing values >= 0

    # total_acc
    assert pd.api.types.is_numeric_dtype(processed["total_acc"])
    assert (processed["total_acc"] >= 0).all()

    # mths_since_last_major_derog
    assert pd.api.types.is_numeric_dtype(processed["mths_since_last_major_derog"])
    assert processed["mths_since_last_major_derog"].isna().sum() == 0
    assert processed["mths_since_last_major_derog"].between(0, 999).all()

    print(processed.head())

    ProfileReport(sample, title="Sample").to_file("data/08_reporting/sample_report.html")
    ProfileReport(processed, title="Processed Sample Data").to_file("data/08_reporting/processed_sample_report.html")
